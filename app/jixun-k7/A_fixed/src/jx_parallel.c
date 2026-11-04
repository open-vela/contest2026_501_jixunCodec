#define _GNU_SOURCE
#include "jx_neon.h"
#ifdef JX_NEON_ENABLED
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <limits.h>
#include <stdio.h>
/* Two bounded compute helpers use the radio cores at a lower priority.
 * Output partitions are independent: each retains the reference MAC order. */
struct completion {int remaining;pthread_cond_t done;};
struct job {void (*fn)(void *,int,int);void *arg;int first,end;struct completion *owner;};
static pthread_mutex_t lock=PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready=PTHREAD_COND_INITIALIZER;
static struct job queue[16];
static pthread_t workers[2];
static int head,tail,queued,stopping,started;
static void *worker(void *arg) {
    int cpu=(int)(intptr_t)arg;
    if(sysconf(_SC_NPROCESSORS_ONLN)>=8)cpu+=6;
    cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
    int affinity=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
#ifdef __NuttX__
    struct sched_param priority={.sched_priority=140};
    int schedule=pthread_setschedparam(pthread_self(),SCHED_RR,&priority);
    fprintf(stderr,"[native codec_helper cpu=%d affinity_rc=%d rr_priority=140 sched_rc=%d]\n",cpu,affinity,schedule);
#else
    (void)affinity;
#endif
    for(;;) {
        pthread_mutex_lock(&lock);
        while(!queued&&!stopping)pthread_cond_wait(&ready,&lock);
        if(!queued&&stopping){pthread_mutex_unlock(&lock);break;}
        struct job job=queue[head];head=(head+1)%16;queued--;
        pthread_mutex_unlock(&lock);
        job.fn(job.arg,job.first,job.end);
        pthread_mutex_lock(&lock);
        if(!--job.owner->remaining)pthread_cond_signal(&job.owner->done);
        pthread_mutex_unlock(&lock);
    }
    return NULL;
}
int jx_parallel_start(void) {
    if(started)return started;
    head=tail=queued=stopping=0;
    pthread_attr_t attr;if(pthread_attr_init(&attr))return 0;
    size_t stack=65536;
#ifdef PTHREAD_STACK_MIN
    if(stack<(size_t)PTHREAD_STACK_MIN)stack=PTHREAD_STACK_MIN;
#endif
    if(pthread_attr_setstacksize(&attr,stack)){pthread_attr_destroy(&attr);return 0;}
    for(int i=0;i<2;i++) {
        if(pthread_create(&workers[i],&attr,worker,(void *)(intptr_t)i))break;
        started++;
    }
    pthread_attr_destroy(&attr);return started;
}
void jx_parallel_stop(void) {
    pthread_mutex_lock(&lock);stopping=1;pthread_cond_broadcast(&ready);pthread_mutex_unlock(&lock);
    for(int i=0;i<started;i++)pthread_join(workers[i],NULL);
    started=0;
}
void jx_parallel_run(void (*fn)(void *,int,int),void *arg,int units,int quantum,size_t work) {
    int blocks=units/quantum,parts=started+1;
    if(!started||blocks<parts||work<24000){fn(arg,0,units);return;}
    struct completion completion={.remaining=started};
    if(pthread_cond_init(&completion.done,NULL)){fn(arg,0,units);return;}
    pthread_mutex_lock(&lock);
    if(queued+started>16) {
        pthread_mutex_unlock(&lock);pthread_cond_destroy(&completion.done);fn(arg,0,units);return;
    }
    for(int i=1;i<parts;i++) {
        int first=blocks*i/parts*quantum,end=i==parts-1?units:blocks*(i+1)/parts*quantum;
        queue[tail]=(struct job){fn,arg,first,end,&completion};tail=(tail+1)%16;queued++;
    }
    pthread_cond_broadcast(&ready);pthread_mutex_unlock(&lock);
    fn(arg,0,blocks/parts*quantum);
    pthread_mutex_lock(&lock);
    while(completion.remaining)pthread_cond_wait(&completion.done,&lock);
    pthread_mutex_unlock(&lock);pthread_cond_destroy(&completion.done);
}
#endif
