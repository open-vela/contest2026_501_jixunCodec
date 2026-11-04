/* Offline access and call admission over the existing 52-byte JR envelope.
 * Address/cookie/CRC are routing and corruption checks, not authentication.
 * All clock values use CLOCK_MONOTONIC microseconds. Never set system time.
 */
#include <inttypes.h>
#include <sys/random.h>
#include <sys/stat.h>
static int pair_registered,pair_response_seen,pair_pending,pair_committed,pair_want_call;
static uint32_t pair_cookie;
static uint64_t pair_last_request,pair_anchor;
static int64_t pair_offset_us;
static double pair_next_join,pair_last_peer,pair_call_deadline,pair_next_call,pair_last_control;
static double pair_best_rtt=1e9;
static unsigned pair_sync_samples,pair_tx_control,pair_rx_control;
static const uint64_t pair_cycle_us=96000,pair_slave_slot_us=48000;
static const char *pair_dir;
static uint64_t pair_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000+t.tv_nsec/1000;}
static void pair_put64(uint8_t*p,uint64_t v){for(int i=0;i<8;i++)p[i]=(uint8_t)(v>>(8*i));}
static uint64_t pair_get64(const uint8_t*p){uint64_t v=0;for(int i=0;i<8;i++)v|=(uint64_t)p[i]<<(8*i);return v;}
static int pair_send(sx1262_t*d,int type,uint64_t a,uint64_t b,uint64_t anchor){
 uint8_t p[52]={0};p[0]='J';p[1]='R';p[2]=1;p[3]=(uint8_t)type;
 wr32(p+4,0x10000001u);wr32(p+8,0x20000002u);wr32(p+12,pair_cookie);
 pair_put64(p+16,a);pair_put64(p+24,b);pair_put64(p+32,pair_us());pair_put64(p+40,anchor);
 /* Two-byte compact access plan: version[1:0], sender[2],
  * cycle/4ms[7:3], B-slot/4ms[11:8], voice-profile[15:12].
  * Profile1 means A261/3K/SF7/BW500k and protected JX0x45 frames. */
 uint16_t plan=2u|((unsigned)pair_slave<<2)|((pair_cycle_us/4000u)<<3)|((pair_slave_slot_us/4000u)<<8)|(1u<<12);
 p[48]=plan;p[49]=plan>>8;
 uint16_t c=crc16(p,50);p[50]=c;p[51]=c>>8;
 int rc=send_frame(d,p,52);pair_tx_control++;
 fprintf(stderr,"[access tx role=%c type=%d cookie=%08x anchor_us=%"PRIu64" rc=%d]\n",pair_slave?'B':'A',type,pair_cookie,anchor,rc);
 return rc;
}
static void pair_file(const char *name,char *path,size_t n){snprintf(path,n,"%s/%s",pair_dir,name);}
static int pair_command(void){
 char path[512],text[32];pair_file("request",path,sizeof(path));
 FILE*f=fopen(path,"r");if(!f)return 0;int ok=fgets(text,sizeof(text),f)!=NULL;fclose(f);if(!ok)return 0;
 if(!strncmp(text,"start",5)){unlink(path);return 1;}
 if(!strncmp(text,"toggle",6)){unlink(path);return 3;}
 if(!strncmp(text,"stop",4)){unlink(path);return 2;}return 0;
}
static void pair_status(void){
 char tmp[544],path[512];pair_file("access.json",path,sizeof(path));snprintf(tmp,sizeof(tmp),"%s.tmp",path);
 FILE*f=fopen(tmp,"w");if(!f)return;
 fprintf(f,"{\"role\":\"%c\",\"state\":\"%s\",\"registered\":%s,\"call_requested\":%s,\"cookie\":%u,\"clock_source\":\"LoRa JR v2 monotonic\",\"master_minus_local_us\":%"PRId64",\"sync_samples\":%u,\"best_rtt_us\":%.0f,\"call_anchor_master_us\":%"PRIu64",\"cycle_us\":%"PRIu64",\"B_slot_us\":%"PRIu64",\"control_tx\":%u,\"control_rx\":%u,\"epoch\":%.6f}\n",
 pair_slave?'B':'A',pair_committed?"call_committed":pair_pending?"call_negotiating":pair_registered?"registered":"joining",pair_registered?"true":"false",pair_want_call?"true":"false",pair_cookie,pair_offset_us,pair_sync_samples,pair_best_rtt<1e9?pair_best_rtt*1e6:0,pair_anchor,pair_cycle_us,pair_slave_slot_us,pair_tx_control,pair_rx_control,wall_s());
 if(fclose(f)==0)rename(tmp,path);
}
static void pair_save_call(void){
 char tmp[544],path[512];pair_file("call_session.json",path,sizeof(path));snprintf(tmp,sizeof(tmp),"%s.tmp",path);
 FILE*f=fopen(tmp,"w");if(!f)return;
 fprintf(f,"{\"role\":\"%c\",\"cookie\":%u,\"master_anchor_us\":%"PRIu64",\"master_minus_local_us\":%"PRId64",\"cycle_us\":%"PRIu64",\"B_slot_us\":%"PRIu64",\"sf\":7,\"bw\":6}\n",pair_slave?'B':'A',pair_cookie,pair_anchor,pair_offset_us,pair_cycle_us,pair_slave_slot_us);
 if(fclose(f)==0)rename(tmp,path);
}
static void pair_begin_prepare(sx1262_t*d){
 if(pair_slave||!pair_registered||pair_pending)return;
 uint64_t now=pair_us();pair_anchor=((now+3000000+pair_cycle_us-1)/pair_cycle_us)*pair_cycle_us;
 pair_pending=1;pair_call_deadline=now_s()+2.0;pair_next_call=now_s()+.3;
 pair_send(d,11,0,0,pair_anchor);
}
static int pair_receive(sx1262_t*d,const uint8_t*p,size_t n){
 if(n!=52)return 0;
 uint16_t plan=(uint16_t)(p[48]|p[49]<<8);
 if(p[0]!='J'||p[1]!='R'||p[2]!=1||(plan&3)!=2||((plan>>2)&1)==(unsigned)pair_slave||
    ((plan>>3)&31)*4000u!=pair_cycle_us||((plan>>8)&15)*4000u!=pair_slave_slot_us||(plan>>12)!=1||
    rd32(p+4)!=0x10000001u||rd32(p+8)!=0x20000002u||crc16(p,50)!=(uint16_t)(p[50]|p[51]<<8))return 0;
 uint64_t observed=pair_us();int type=p[3];uint32_t cookie=rd32(p+12);if(!cookie)return 0;
 if(!pair_slave&&type==1){
  if(pair_cookie!=cookie){pair_cookie=cookie;pair_registered=0;pair_pending=pair_committed=0;}
  pair_last_peer=now_s();pair_send(d,2,pair_get64(p+16),observed,0);
 }else if(cookie!=pair_cookie)return 0;
 else if(pair_slave&&type==2&&pair_get64(p+16)==pair_last_request){
  int64_t t1=(int64_t)pair_last_request,t2=(int64_t)pair_get64(p+24),t3=(int64_t)pair_get64(p+32),t4=(int64_t)observed;
  int64_t rtt=(t4-t1)-(t3-t2),offset=((t2-t1)+(t3-t4))/2;
  if(rtt<0||rtt>500000){fprintf(stderr,"[sync reject rtt_us=%"PRId64"]\n",rtt);return 0;}
  pair_sync_samples++;if(pair_sync_samples%5==1)pair_best_rtt=1e9;if((double)rtt/1e6<pair_best_rtt){pair_best_rtt=(double)rtt/1e6;pair_offset_us=offset;}
  pair_response_seen=1;pair_last_peer=now_s();pair_send(d,3,0,0,0);
  fprintf(stderr,"[sync sample=%u master_minus_local_us=%"PRId64" rtt_us=%"PRId64" selected_offset_us=%"PRId64"]\n",pair_sync_samples,offset,rtt,pair_offset_us);
 }else if(!pair_slave&&type==3){
  pair_registered=1;pair_last_peer=now_s();pair_send(d,4,0,0,0);
  fprintf(stderr,"[access registered role=A peer=K7_slave cookie=%08x]\n",cookie);
 }else if(pair_slave&&type==4&&pair_response_seen){
  pair_registered=1;pair_last_peer=now_s();pair_next_join=now_s()+(pair_sync_samples<5?1.0:5.0);
  fprintf(stderr,"[access registered role=B peer=K7_master cookie=%08x offset_us=%"PRId64"]\n",cookie,pair_offset_us);
 }else if(type==14){pair_want_call=pair_pending=pair_committed=0;pair_status();}
 else if(!pair_slave&&type==10&&pair_registered){pair_last_peer=now_s();pair_begin_prepare(d);}
 else if(pair_slave&&type==11&&pair_registered){
  uint64_t anchor=pair_get64(p+40);int64_t ahead=(int64_t)anchor-(int64_t)pair_us()-pair_offset_us;
  if(ahead<300000||ahead>5000000)return 0;
  pair_anchor=anchor;pair_pending=1;pair_call_deadline=now_s()+4;pair_next_call=now_s()+.25;pair_send(d,12,0,0,anchor);
 }else if(!pair_slave&&type==12&&pair_pending&&pair_get64(p+40)==pair_anchor){
  for(int i=0;i<3;i++){pair_send(d,13,0,0,pair_anchor);usleep(20000);}
  pair_committed=1;pair_save_call();fprintf(stderr,"[call COMMITTED role=A master_anchor_us=%"PRIu64"]\n",pair_anchor);
 }else if(pair_slave&&type==13&&pair_pending&&pair_get64(p+40)==pair_anchor){
  pair_committed=1;pair_save_call();fprintf(stderr,"[call COMMITTED role=B master_anchor_us=%"PRIu64" offset_us=%"PRId64"]\n",pair_anchor,pair_offset_us);
 }else return 0;
 pair_rx_control++;pair_status();return 1;
}
static void pair_init(void){
 pair_registered=pair_response_seen=pair_pending=pair_committed=pair_want_call=0;
 pair_cookie=0;pair_last_request=pair_anchor=0;pair_offset_us=0;
 pair_next_join=pair_last_peer=pair_call_deadline=pair_next_call=pair_last_control=0;
 pair_best_rtt=1e9;pair_sync_samples=pair_tx_control=pair_rx_control=0;

 const char*r=getenv("PAIR_ROLE");pair_slave=r&&(*r=='b'||*r=='B');pair_dir=getenv("PAIR_STATE_DIR");if(!pair_dir)pair_dir="/tmp/k7_pair";
 mkdir(pair_dir,0755);pair_registered=0;pair_offset_us=0;pair_next_join=0;
 if(pair_slave){getrandom(&pair_cookie,sizeof(pair_cookie),0);if(!pair_cookie)pair_cookie=1;}
 pair_status();
}
static int pair_tick(sx1262_t*d,int receiving_voice){
 double t=now_s();int cmd=pair_command();
 if(cmd==1)pair_want_call=1;
 if(cmd==3)pair_want_call=!pair_want_call;
 if(cmd==2||(cmd==3&&!pair_want_call)){
  if(pair_pending&&pair_registered)pair_send(d,14,0,0,pair_anchor);
  pair_want_call=pair_pending=pair_committed=0;pair_status();
 }
 if(pair_want_call&&pair_registered&&!receiving_voice&&!pair_pending){
  if(pair_slave){pair_send(d,10,0,0,0);pair_next_call=t+.5;pair_call_deadline=t+4;pair_pending=1;}
  else pair_begin_prepare(d);
 }
 if(!receiving_voice&&!pair_pending&&pair_slave&&t>=pair_next_join){
  pair_last_request=pair_us();pair_send(d,1,pair_last_request,0,0);pair_next_join=t+1.5;
 }
 if(pair_pending&&!pair_committed&&t>pair_call_deadline){pair_pending=0;fprintf(stderr,"[call admission_timeout]\n");pair_status();}
 if(pair_pending&&!pair_committed&&t>=pair_next_call){
  if(pair_slave)pair_send(d,pair_anchor?12:10,0,0,pair_anchor);else pair_send(d,11,0,0,pair_anchor);
  pair_next_call=t+.3;
 }
 if(pair_registered&&!pair_pending&&t-pair_last_peer>20){
  pair_registered=0;pair_response_seen=0;pair_best_rtt=1e9;pair_sync_samples=0;pair_status();
 }
 if(pair_committed){
  int64_t ahead=(int64_t)pair_anchor-(int64_t)pair_us()-pair_offset_us;
  if(ahead<=800000){pair_status();return 42;}
 }
 if(t-pair_last_control>1){pair_last_control=t;pair_status();}return 0;
}
