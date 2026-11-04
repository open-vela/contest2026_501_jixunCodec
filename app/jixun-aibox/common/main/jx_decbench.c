/* Decoder-only fixture benchmark: no encoder, radio, microphone or playback. */
#include <nuttx/config.h>
#include "stream_codec.h"
#include "nn_ops.h"
#include "jxprof.h"
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "decode_fixture.h"

extern int jx_pf_session_begin(int);
extern void jx_pf_session_end(void);
extern void jx_i8_options_begin(void);
extern void jx_i8_options_end(void);

struct decode_job { int dual, step, rc; };
static double decode_ms(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec*1000.0+(double)t.tv_nsec/1e6;
}
struct detail { char name[96]; double ms; };
static struct detail details[96];static int detail_n;static double detail_last;
static void detail_tap(void *ud,const char *name,const float *data,int c,int t,int last) {
    (void)ud;(void)data;(void)c;(void)t;(void)last;
    double now=decode_ms();int i;
    for(i=0;i<detail_n;i++)if(!strcmp(details[i].name,name))break;
    if(i==detail_n&&detail_n<96){snprintf(details[i].name,96,"%s",name);detail_n++;}
    if(i<96)details[i].ms+=now-detail_last;
    detail_last=now;
}
static void *decode_run(void *arg) {
    struct decode_job *job=arg;
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(1,&cpus);
    int pin=sched_setaffinity(0,sizeof(cpus),&cpus);
    jx_i8_options_begin();jx_wq_init();jx_prof_reset();
    void *pressure=NULL;
    const char *stress=getenv("JX_D_PRESSURE");
    if(stress && stress[0]=='1') {
        pressure=malloc(768u*1024u);
        if(!pressure){job->rc=6;jx_i8_options_end();return NULL;}
        memset(pressure,0xa5,768u*1024u);
        printf("[DB_PRESSURE] reserved=786432 bytes for simulated receive/play buffers\n");
    }
    SDecState *dec=sdec_new_chunk(job->step);
    int parallel=jx_pf_session_begin(job->dual);
    printf("[DB_SETUP] pin=%d parallel=%d step=%d window=%d depth=%d\n",pin,parallel,job->step,EN_WINDOW,EN_DEC_DEPTH);
    const char *detail=getenv("JX_D_DETAIL");
    int detailed=detail&&detail[0]=='1';
    if(detailed){memset(details,0,sizeof(details));detail_n=0;g_stap=detail_tap;}
    uint64_t hash=1469598103934665603ull;
    float pcm[16*96];int samples=0;
    double start=decode_ms(),work=0,max_step=0;
    for(int off=0;off<(int)(sizeof(decode_fixture)/sizeof(decode_fixture[0]));off+=job->step) {
        int nt=(int)(sizeof(decode_fixture)/sizeof(decode_fixture[0]))-off;
        if(nt>job->step)nt=job->step;
        double before=decode_ms();detail_last=before;int got=sdec_step(dec,decode_fixture+off,nt,pcm);
        double duration=decode_ms()-before;work+=duration;
        if(duration>max_step)max_step=duration;
        if(got!=nt*96){job->rc=1;break;}
        const unsigned char *bytes=(const unsigned char *)pcm;
        for(size_t i=0;i<(size_t)got*sizeof(float);i++)hash=(hash^bytes[i])*1099511628211ull;
        const char *dump=getenv("JX_D_DUMP");
        if(dump&&dump[0]=='1') {
            printf("[DB_PCM] %d ",samples);
            for(size_t i=0;i<(size_t)got*sizeof(float);i++)printf("%02x",bytes[i]);
            printf("\n");
        }
        samples+=got;
    }
    double wall=decode_ms()-start;
    if(parallel>0)jx_pf_session_end();
    printf("[DB_RESULT] samples=%d work_ms=%.1f wall_ms=%.1f rtf=%.4f max_step_ms=%.1f arena_peak=%zu hash=%016llx rc=%d\n",
           samples,work,wall,work*16.0/(double)samples,max_step,sdec_peak(dec)*sizeof(float),(unsigned long long)hash,job->rc);
    if(detailed){g_stap=NULL;for(int i=0;i<detail_n;i++)printf("[D_DETAIL] %s ms=%.1f\n",details[i].name,details[i].ms);}
    jx_prof_dump();sdec_free(dec);free(pressure);jx_i8_options_end();return NULL;
}
int jx_cmd_decbench(int argc,char **argv) {
    struct decode_job job={0,8,0};
    if(argc>2)job.dual=atoi(argv[2])!=0;
    if(argc>3)job.step=atoi(argv[3]);
    if(job.step<1||job.step>16)return 2;
    pthread_t thread;pthread_attr_t attr;
    if(pthread_attr_init(&attr)!=0)return 3;
    if(pthread_attr_setstacksize(&attr,262144)!=0){pthread_attr_destroy(&attr);return 4;}
    int rc=pthread_create(&thread,&attr,decode_run,&job);
    pthread_attr_destroy(&attr);if(rc!=0)return 5;
    pthread_join(thread,NULL);return job.rc;
}
