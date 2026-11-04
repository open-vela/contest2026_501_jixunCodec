/* Native K7 touchscreen console. LVGL is confined to this one low-priority
 * task; all radio lifecycle changes go through the existing node supervisor.
 * Physical panel/touch are portrait. Both use the original Linux LEFT mapping.
 */
#include <nuttx/config.h>
#include <nuttx/input/touchscreen.h>
#include <nuttx/video/fb.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <lvgl/lvgl.h>

LV_FONT_DECLARE(jx_font_24);
LV_FONT_DECLARE(jx_font_32);
extern int k7_pair_main(int,char **);
extern int audioctl_main(int,char **);
#define W 1280
#define H 720
#define BG 0x10171c
#define CARD 0x192329
#define MUTED 0x91a3ad
#define GREEN 0x73d8ac
#define WHITE 0xeef3f5
#define RED 0xf17b79

enum {UNAVAILABLE,JOINING,READY,CONNECTING,INCALL,ENDING};
static const char *phases[]={"unavailable","joining","ready","connecting","in_call","ending"};
static const char *phase_text[]={"节点未启动","正在入网","已入网 · 待机","正在接通","通话中","正在结束"};
static const char *tabs[]={"通信总览","节点接入","业务信道","实时通话","链路监控","事件记录","设置"};
static const char *tips[]={"等待节点启动","正在通过 LoRa 与对端建立连接","点击开始，双方可持续通话","正在与对端协商，请稍候","双方可直接说话，点击按钮结束","正在结束通话并恢复入网"};
static pthread_mutex_t command_lock=PTHREAD_MUTEX_INITIALIZER;
static atomic_bool running;
static atomic_int synthetic_steps;
static int synthetic_x,synthetic_y;
static char snapshot_path[96];
static atomic_int snapshot_request;
static int snapshot_rc;
static int fb=-1,input=-1,tab,phase=-1,styled_phase=-1,last_rc,volume=80,gain=75;
static uint8_t *pixels;
static struct fb_planeinfo_s plane;
static unsigned touches,flushes;
static int last_touch_x,last_touch_y;
static char role='?',program[32];
static double now_real,call_since,pending_until;
static int pending;
static uint64_t tx,rx,gaps,ctrl_tx,ctrl_rx,sync_samples;
static double rssi,snr,rx_age,phase_rms,best_rtt,offset;
static bool signal_valid;
static lv_obj_t *body,*content,*header_role,*uptime_label,*phase_label,*tip_label,*action,*action_label;
static lv_obj_t *nav[7],*values[4],*details,*footer,*page_title;
static char events[16][128];static unsigned event_count;

static double seconds(clockid_t id)
{struct timespec t;clock_gettime(id,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static uint32_t ticks(void){return (uint32_t)(seconds(CLOCK_MONOTONIC)*1000);}
/* Inputs are flat, small JSON objects atomically renamed by our own native
 * services. Missing/malformed data yields unavailable, never synthetic metrics.
 */
static size_t read_state(const char *name,char *dst,size_t size)
{
 char path[128];snprintf(path,sizeof(path),"/tmp/k7_pair/%s.json",name);
 FILE *f=fopen(path,"r");size_t n=f?fread(dst,1,size-1,f):0;if(f)fclose(f);dst[n]=0;
 if(n<2||dst[0]!='{'||!strchr(dst,'}')){dst[0]=0;return 0;}return n;
}
static const char *field(const char *s,const char *key)
{
 char pattern[80];snprintf(pattern,sizeof(pattern),"\"%s\":",key);
 const char *v=strstr(s,pattern);if(!v)return NULL;v+=strlen(pattern);
 while(*v==' '||*v=='\t')v++;return v;
}
static double number(const char *s,const char *key,double fallback)
{const char *v=field(s,key);if(!v)return fallback;char *end;double d=strtod(v,&end);return end==v?fallback:d;}
static bool truth(const char *s,const char *key)
{const char *v=field(s,key);return v&&!strncmp(v,"true",4);}
static void string(const char *s,const char *key,char *out,size_t size)
{
 const char *v=field(s,key);out[0]=0;if(!v||*v++!='"')return;
 const char *end=strchr(v,'"');if(!end||(size_t)(end-v)>=size)return;
 memcpy(out,v,end-v);out[end-v]=0;
}
static bool fresh(const char *s)
{double t=number(s,"epoch",-1);return t>=0&&now_real-t>=-1&&now_real-t<5;}
static void event(const char *text)
{
 unsigned sec=(unsigned)seconds(CLOCK_MONOTONIC);
 snprintf(events[event_count%16],sizeof(events[0]),"%02u:%02u:%02u  %s",sec/3600,sec/60%60,sec%60,text);event_count++;
}
static void read_telemetry(void)
{
 char cur[512],access[1536],call[2048],r[8],st[32],ast[32];
 read_state("current",cur,sizeof(cur));read_state("access",access,sizeof(access));read_state("call",call,sizeof(call));
 now_real=seconds(CLOCK_REALTIME);string(cur,"role",r,sizeof(r));role=r[0]?r[0]:'?';
 string(cur,"program",program,sizeof(program));string(call,"state",st,sizeof(st));string(access,"state",ast,sizeof(ast));
 bool alive=number(cur,"pid",-1)>0&&kill((pid_t)number(cur,"pid",-1),0)==0;
 bool voice=alive&&!strcmp(program,"k7_voice"),af=fresh(access),cf=fresh(call);
 int next=UNAVAILABLE;
 tx=cf&&voice?(uint64_t)number(call,"tx_packets",0):0;
 rx=cf&&voice?(uint64_t)number(call,"rx_packets",0):0;
 gaps=cf&&voice?(uint64_t)number(call,"seq_gaps",0):0;
 rx_age=number(call,"rx_age_ms",1e9);
 rssi=number(call,"rssi_last_dbm",-999);snr=number(call,"snr_last_db",-999);
 signal_valid=voice&&cf&&rx>0&&rx_age<3000&&rssi!=-999&&snr!=-999;
 ctrl_tx=(uint64_t)number(access,"control_tx",0);ctrl_rx=(uint64_t)number(access,"control_rx",0);
 sync_samples=(uint64_t)number(access,"sync_samples",0);best_rtt=number(access,"best_rtt_us",0);
 offset=number(voice&&cf?call:access,"master_minus_local_us",0);
 phase_rms=number(call,"software_phase_rms_us",0);
 if(voice){next=cf&&!strcmp(st,"ending")?ENDING:rx?INCALL:CONNECTING;}
 else if(alive&&!strcmp(program,"k7_access")){
  next=af?(truth(access,"registered")?READY:JOINING):JOINING;
  if(af&&(truth(access,"call_requested")||!strcmp(ast,"call_negotiating")||!strcmp(ast,"call_committed")))next=CONNECTING;
 }
 if(now_real<pending_until&&alive){
  if(pending==1&&next==READY)next=CONNECTING;
  if(pending==2&&(next==INCALL||next==CONNECTING))next=ENDING;
 }
 if(next!=phase){phase=next;event(phase_text[phase]);if(phase==INCALL)call_since=seconds(CLOCK_MONOTONIC);else if(phase!=ENDING)call_since=0;}
}
#include "ui/console_views.inc"
static void flush(lv_display_t *d,const lv_area_t *a,uint8_t *src)
{
 /* LVGL XRGB8888 is B,G,R,X in little endian; VOP uses R,G,B bytes. */
 for(int y=a->y1;y<=a->y2;y++)for(int x=a->x1;x<=a->x2;x++){
  uint8_t *dst=pixels+(1279-x)*plane.stride+y*3;
  dst[0]=src[2];dst[1]=src[1];dst[2]=src[0];src+=4;
 }
 struct fb_area_s area={.x=a->y1,.y=1279-a->x2,.w=a->y2-a->y1+1,.h=a->x2-a->x1+1};
 ioctl(fb,FBIO_UPDATE,(unsigned long)&area);flushes++;lv_display_flush_ready(d);
}
static void touch_read(lv_indev_t *indev,lv_indev_data_t *data)
{
 static bool down;static int x,y;
 int step=atomic_load(&synthetic_steps);
 if(step>0){x=synthetic_x;y=synthetic_y;down=step>1;atomic_fetch_sub(&synthetic_steps,1);}
 else if(input>=0){
  struct touch_sample_s sample;
  if(read(input,&sample,sizeof(sample))==(ssize_t)sizeof(sample)&&sample.npoints>0){
   struct touch_point_s *p=&sample.point[0];
   if(p->flags&TOUCH_POS_VALID){x=1279-p->y;y=p->x;if(x<0)x=0;if(x>=W)x=W-1;if(y<0)y=0;if(y>=H)y=H-1;}
   if(p->flags&TOUCH_UP)down=false;
   else if(p->flags&(TOUCH_DOWN|TOUCH_MOVE))down=true;
   touches++;last_touch_x=x;last_touch_y=y;
  }
 }
 data->point.x=x;data->point.y=y;data->state=down?LV_INDEV_STATE_PRESSED:LV_INDEV_STATE_RELEASED;
}
/* RGB565 RLE export of the actual framebuffer, only on explicit diagnostics.
 * No second frame copy in the voice realtime path. The UI thread owns writes.
 */
static int snapshot(const char *path)
{
 FILE *f=fopen(path,"wb");if(!f)return -errno;
 uint8_t h[]={ 'J','X','F','B',0,5,0xd0,2 };fwrite(h,1,sizeof(h),f);
 uint16_t prev=0,count=0;
 for(int y=0;y<H;y++)for(int x=0;x<W;x++){
  uint8_t *p=pixels+(1279-x)*plane.stride+y*3;
  uint16_t color=((p[0]&248)<<8)|((p[1]&252)<<3)|(p[2]>>3);
  if(count&&(prev!=color||count==65535)){
   uint8_t r[]={count&255,count>>8,prev&255,prev>>8};if(fwrite(r,1,4,f)!=4){fclose(f);return -EIO;}count=0;
  }prev=color;count++;
 }
 if(count){uint8_t r[]={count&255,count>>8,prev&255,prev>>8};fwrite(r,1,4,f);}
 return fclose(f)?-errno:0;
}
int k7_ui_main(int argc,char **argv)
{
 if(argc>1){
  if(!atomic_load(&running)){fprintf(stderr,"k7_ui is not running\n");return 1;}
  if(argc==4&&!strcmp(argv[1],"tap")){
   char *a,*b;long x=strtol(argv[2],&a,10),y=strtol(argv[3],&b,10);
   if(*a||*b||x<0||x>=W||y<0||y>=H)return 1;
   pthread_mutex_lock(&command_lock);
   if(atomic_load(&synthetic_steps)){pthread_mutex_unlock(&command_lock);return 1;}
   synthetic_x=x;synthetic_y=y;atomic_store(&synthetic_steps,4);pthread_mutex_unlock(&command_lock);return 0;
  }
  if(argc==3&&!strcmp(argv[1],"snapshot")&&(!strncmp(argv[2],"/tmp/",5))&&!strstr(argv[2],"..")&&strlen(argv[2])<sizeof(snapshot_path)){
   pthread_mutex_lock(&command_lock);if(atomic_load(&snapshot_request)){pthread_mutex_unlock(&command_lock);return 1;}
   strcpy(snapshot_path,argv[2]);atomic_store(&snapshot_request,1);
   for(int i=0;i<1000&&atomic_load(&snapshot_request);i++)usleep(10000);
   int rc=atomic_load(&snapshot_request)?-ETIMEDOUT:snapshot_rc;
   pthread_mutex_unlock(&command_lock);printf("UI snapshot rc=%d\n",rc);return rc?1:0;
  }
  puts("k7_ui [tap x y | snapshot /tmp/frame.rle]");return 1;
 }
 if(atomic_exchange(&running,true)){puts("k7_ui already running");return 1;}
 fb=open("/dev/fb0",O_RDWR);struct fb_videoinfo_s vi;
 memset(&plane,0,sizeof(plane));
 if(fb<0||ioctl(fb,FBIOGET_VIDEOINFO,(unsigned long)&vi)<0||ioctl(fb,FBIOGET_PLANEINFO,(unsigned long)&plane)<0||vi.xres!=720||vi.yres!=1280||plane.bpp!=24||!plane.fbmem){perror("k7_ui framebuffer");if(fb>=0)close(fb);atomic_store(&running,false);return 1;}
 pixels=plane.fbmem;input=open("/dev/input0",O_RDONLY|O_NONBLOCK);
 cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(2,&cpus);sched_setaffinity(0,sizeof(cpus),&cpus);
 lv_init();lv_tick_set_cb(ticks);lv_display_t *display=lv_display_create(W,H);
 void *draw=malloc(W*32*4);if(!display||!draw){fprintf(stderr,"UI allocation failed\n");atomic_store(&running,false);return 1;}
 lv_display_set_color_format(display,LV_COLOR_FORMAT_XRGB8888);
 lv_display_set_buffers(display,draw,NULL,W*32*4,LV_DISPLAY_RENDER_MODE_PARTIAL);lv_display_set_flush_cb(display,flush);
 lv_indev_t *pointer=lv_indev_create();lv_indev_set_type(pointer,LV_INDEV_TYPE_POINTER);lv_indev_set_read_cb(pointer,touch_read);
 build_shell();printf("UI ready: original console, 1280x720, original brand assets, CPU2\n");
 uint32_t last=ticks();
 for(;;){
  uint32_t delay=lv_timer_handler();
  if(ticks()-last>=1000){refresh();last=ticks();}
  if(atomic_load(&snapshot_request)){snapshot_rc=snapshot(snapshot_path);atomic_store(&snapshot_request,0);}
  if(delay>25)delay=25;if(delay<5)delay=5;usleep(delay*1000);
 }
}
