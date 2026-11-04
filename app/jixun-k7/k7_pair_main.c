/* Native node supervisor for the existing JR admission + JX voice stages.
 * One task owns the radio lifecycle. Request files preserve the Linux wire
 * implementation's control seam; Python/systemd are not required. */
#include <nuttx/config.h>
#include <nuttx/sched.h>
#include <sys/stat.h>
#include <sys/mount.h>
#include <sys/statfs.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "rk3576_gpio.h"
extern int audioctl_main(int,char**);
extern int k7_access_main(int,char**);
extern int k7_voice_main(int,char**);
static pthread_mutex_t control=PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t shutdown_node;
static pid_t node_pid=-1;
static char node_role[2]="a";
#define STATE "/tmp/k7_pair"
static int ensure_tmp(void){
 struct statfs fs;
 if(!statfs("/tmp",&fs)&&fs.f_type==TMPFS_MAGIC)return 0;
 if(mkdir("/tmp",0777)<0&&errno!=EEXIST)return -errno;
 if(!mount(NULL,"/tmp","tmpfs",0,NULL))return 0;
 int error=errno;
 /* Board storage startup may have mounted it between the check and mount. */
 if(!statfs("/tmp",&fs)&&fs.f_type==TMPFS_MAGIC)return 0;
 return -error;
}
static int command(const char *value){
 FILE *f=fopen(STATE "/request.tmp","w");if(!f)return -errno;
 if(fprintf(f,"%s\n",value)<0){fclose(f);return -EIO;}
 if(fclose(f))return -errno;return rename(STATE "/request.tmp",STATE "/request");
}
static int field(const char *json,const char *name,char *out,size_t n){
 char key[80];snprintf(key,sizeof(key),"\"%s\":",name);const char *v=strstr(json,key);if(!v)return -EINVAL;
 v+=strlen(key);while(*v==' ')v++;char *end;strtoll(v,&end,10);if(end==v||end-v>=(int)n)return -EINVAL;
 memcpy(out,v,end-v);out[end-v]=0;return 0;
}
static void current(const char *stage,int rc){
 FILE*f=fopen(STATE "/current.json.tmp","w");if(!f)return;
 fprintf(f,"{\"role\":\"%c\",\"program\":\"%s\",\"pid\":%d,\"last_rc\":%d,\"peer_transport\":\"LoRa_only\"}\n",node_role[0]=='a'?'A':'B',stage,(int)getpid(),rc);
 if(!fclose(f))rename(STATE "/current.json.tmp",STATE "/current.json");
}
static int node_task(int argc,char **argv){
 (void)argc;(void)argv;int rc=0;struct gpio_dev_s *fan=NULL;
 setenv("PAIR_ROLE",node_role,1);setenv("PAIR_STATE_DIR",STATE,1);
 setenv("PAIR_LEGACY_STATUS",STATE "/legacy_status.json",1);
 setenv("JX_K7_SF","8",1);setenv("JX_K7_CONT_ACK","0",1);setenv("TX_DBM","22",1);
 if(!rk3576_gpio_get(GPIO_PORT2|GPIO_PIN_B3,&fan)){
  rk3576_gpio_set_af(fan,0);rk3576_gpio_set_mode(fan,RK3576_GPIO_OUTPUT);rk3576_gpio_write_bit(fan,true);
 }
 char *volume[]={"audioctl","volume","80",NULL};
 char *gain[]={"audioctl","mic","gain","75",NULL};
 int vr=audioctl_main(3,volume),gr=audioctl_main(4,gain);
 fprintf(stderr,"[native audio_controls volume80_rc=%d mic_gain75_rc=%d]\n",vr,gr);
 while(!shutdown_node){
  char *access[]={"k7_access","315360000",NULL};current("k7_access",rc);
  rc=k7_access_main(2,access);if(shutdown_node)break;
  if(rc!=42){fprintf(stderr,"[native access_retry rc=%d]\n",rc);sleep(1);continue;}
  char json[768],anchor[32],offset[32];FILE *f=fopen(STATE "/call_session.json","r");
  size_t n=f?fread(json,1,sizeof(json)-1,f):0;if(f)fclose(f);json[n]=0;
  if(!n||field(json,"master_anchor_us",anchor,sizeof(anchor))||field(json,"master_minus_local_us",offset,sizeof(offset))){rc=-EINVAL;continue;}
  setenv("PAIR_ANCHOR_US",anchor,1);setenv("PAIR_CLOCK_OFFSET_US",offset,1);
  char *voice[]={"k7_voice","duplex","315360000","native","1.0",node_role,NULL};current("k7_voice",rc);
  rc=k7_voice_main(6,voice);fprintf(stderr,"[native call_finished rc=%d]\n",rc);
  unlink(STATE "/call_session.json");rename(STATE "/call.json",STATE "/previous_call.json");
 }
 current("stopped",rc);
 if(fan)rk3576_gpio_put(fan);
 pthread_mutex_lock(&control);node_pid=-1;pthread_mutex_unlock(&control);return rc;
}
static void print_file(const char *name){char path[128],buf[256];snprintf(path,sizeof(path),STATE "/%s",name);FILE*f=fopen(path,"r");if(!f)return;printf("%s: ",name);while(fgets(buf,sizeof(buf),f))fputs(buf,stdout);fclose(f);}
int k7_pair_main(int argc,char **argv){
 if(argc<2){puts("k7_pair node a|b | start | stop | toggle | status | shutdown");return 1;}
 pthread_mutex_lock(&control);int rc=0;
 if(!strcmp(argv[1],"node")){
  if(argc!=3||(strcmp(argv[2],"a")&&strcmp(argv[2],"b"))){rc=1;goto done;}
  if(node_pid>0){fprintf(stderr,"node already running pid=%d\n",(int)node_pid);rc=1;goto done;}
  int tmp_rc=ensure_tmp();if(tmp_rc<0){errno=-tmp_rc;perror("prepare /tmp");rc=1;goto done;}
  mkdir(STATE,0755);unlink(STATE "/request");node_role[0]=argv[2][0];shutdown_node=0;
  node_pid=task_create("k7_pair_node",100,65536,node_task,NULL);rc=node_pid<0?1:0;
 }else if(!strcmp(argv[1],"status")){
  print_file("current.json");print_file("access.json");print_file("call.json");
 }else if(!strcmp(argv[1],"shutdown")){
  shutdown_node=1;command("stop");if(node_pid>0)kill(node_pid,SIGTERM);
 }else if(!strcmp(argv[1],"start")||!strcmp(argv[1],"stop")||!strcmp(argv[1],"toggle")){
  if(node_pid<=0){fprintf(stderr,"start node a|b first\n");rc=1;}else rc=command(argv[1])?1:0;
 }else rc=1;
done:pthread_mutex_unlock(&control);return rc;
}
