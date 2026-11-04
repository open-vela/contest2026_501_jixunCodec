/* Fixed A-3K call: aligned offline TDD, no IP or alternate codec negotiation. */
#include <sched.h>
#include <inttypes.h>
#include <unistd.h>
static int call_role;
static uint64_t call_anchor_us;
static atomic_llong call_offset_us;
static atomic_int call_end;
static atomic_ullong call_rx_count,call_tx_count,call_gap_count,call_sync_count;
static pthread_mutex_t call_stats_mu=PTHREAD_MUTEX_INITIALIZER;
static int call_snr_min=999,call_snr_max=-999;
static int call_snr_last,call_rssi_last;
static int64_t call_phase_min=INT64_MAX,call_phase_max=INT64_MIN;
static double call_phase_sq,call_phase_sum;
static int64_t call_clock_step_max;
static _Thread_local uint32_t call_tx_stamp;
static double call_end_deadline,call_last_status,call_last_receive;
static const char *call_state_dir;
static uint64_t call_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000+t.tv_nsec/1000;}
static uint32_t call_rd32(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static void call_wr32(uint8_t*p,uint32_t v){for(int i=0;i<4;i++)p[i]=v>>(8*i);}
static uint16_t call_crc(const uint8_t*p){uint8_t v[50];memcpy(v,p,14);memcpy(v+14,p+16,36);return crc16(v,50);}
static void call_pin(int requested){
 const int little[4]={2,3,1,0},big[4]={4,5,1,0};
 const int *physical=sysconf(_SC_NPROCESSORS_ONLN)>=8?big:little;const int priority[4]={150,148,178,176};
 if(requested<4||requested>7)return;int cpu=physical[requested-4];
 if(sysconf(_SC_NPROCESSORS_ONLN)<=cpu)return;
 cpu_set_t set;CPU_ZERO(&set);CPU_SET(cpu,&set);
 int rc=pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
 struct sched_param p={.sched_priority=priority[requested-4]};
 int sr=pthread_setschedparam(pthread_self(),SCHED_RR,&p);
 fprintf(stderr,"[native worker lane=%d cpu=%d affinity_rc=%d rr_priority=%d sched_rc=%d]\n",requested,cpu,rc,p.sched_priority,sr);
}

static void mode_init(int role){
 atomic_store(&call_end,0);atomic_store(&call_rx_count,0);atomic_store(&call_tx_count,0);
 atomic_store(&call_gap_count,0);atomic_store(&call_sync_count,0);
 call_snr_min=999;call_snr_max=-999;call_snr_last=call_rssi_last=0;
 call_phase_min=INT64_MAX;call_phase_max=INT64_MIN;call_phase_sq=call_phase_sum=0;
 call_clock_step_max=0;call_end_deadline=call_last_status=0;

 call_role=role;call_state_dir=getenv("PAIR_STATE_DIR");if(!call_state_dir)call_state_dir="/tmp/k7_pair";
 const char*a=getenv("PAIR_ANCHOR_US"),*o=getenv("PAIR_CLOCK_OFFSET_US");
 call_anchor_us=a?strtoull(a,NULL,10):call_us()+1000000;
 atomic_store(&call_offset_us,o?strtoll(o,NULL,10):0);call_last_receive=now_s();
 fprintf(stderr,"[call aligned role=%c master_anchor_us=%"PRIu64" offset_us=%lld period_us=96000 B_slot_us=48000 sf=7 bw=500k]\n",role?'B':'A',call_anchor_us,atomic_load(&call_offset_us));
}
static void mode_fill(uint8_t*p){
 p[3]=(uint8_t)(call_role|(atomic_load(&call_end)?2:0));call_wr32(p+8,call_tx_stamp);
}
static void mode_transmitted(uint32_t seq){(void)seq;atomic_fetch_add(&call_tx_count,1);}
static void mode_received(const uint8_t*p,uint32_t seq,int rssi,int snr){
 (void)seq;atomic_fetch_add(&call_rx_count,1);
 pthread_mutex_lock(&call_stats_mu);call_last_receive=now_s();
 call_snr_last=snr;call_rssi_last=rssi;
 if(snr<call_snr_min)call_snr_min=snr;if(snr>call_snr_max)call_snr_max=snr;
 if(p[3]&2)atomic_store(&call_end,1);
 if(call_role){
  /* SF7/BW500k, CR4/5, preamble8, explicit52B+CRC: 25664us.
   * The RX observation follows IRQ polling/SPI. Report residuals as software
   * phase, not as a calibrated physical clock-error measurement. */
  int64_t current=atomic_load(&call_offset_us);
  uint64_t estimated_master=call_us()+current;
  int32_t phase=(int32_t)((uint32_t)estimated_master-(call_rd32(p+8)+25664u));
  if(phase>-20000&&phase<20000){
   int64_t correction=-(int64_t)phase/16;atomic_store(&call_offset_us,current+correction);
   if(llabs(correction)>call_clock_step_max)call_clock_step_max=llabs(correction);
   atomic_fetch_add(&call_sync_count,1);call_phase_sum+=phase;call_phase_sq+=(double)phase*phase;
   if(phase<call_phase_min)call_phase_min=phase;if(phase>call_phase_max)call_phase_max=phase;
  }else fprintf(stderr,"[call sync_reject phase_us=%d]\n",phase);
 }
 pthread_mutex_unlock(&call_stats_mu);
}
static int mode_tick(void){
 char path[512],tmp[544],request[32]={0};snprintf(path,sizeof(path),"%s/request",call_state_dir);
 FILE*f=fopen(path,"r");if(f){
  int complete=fgets(request,sizeof(request),f)&&strchr(request,'\n');
  if(complete&&(!strncmp(request,"stop",4)||!strncmp(request,"toggle",6)))atomic_store(&call_end,1);
  fclose(f);if(complete)unlink(path);
 }
 double now=now_s();pthread_mutex_lock(&call_stats_mu);double last_rx=call_last_receive;pthread_mutex_unlock(&call_stats_mu);
 if(now-last_rx>8){fprintf(stderr,"[call peer_timeout offline_rejoin_required]\n");atomic_store(&call_end,1);}
 if(atomic_load(&call_end)&&!call_end_deadline)call_end_deadline=now+.7;
 if(now-call_last_status>=1||atomic_load(&call_end)){
  call_last_status=now;snprintf(path,sizeof(path),"%s/call.json",call_state_dir);snprintf(tmp,sizeof(tmp),"%s.tmp",path);f=fopen(tmp,"w");
  if(f){pthread_mutex_lock(&call_stats_mu);uint64_t count=atomic_load(&call_sync_count);
   char last_snr[32]="null",last_rssi[32]="null";
   if(atomic_load(&call_rx_count)){snprintf(last_snr,sizeof(last_snr),"%.2f",call_snr_last/4.0);snprintf(last_rssi,sizeof(last_rssi),"%d",call_rssi_last);}
   fprintf(f,"{\"role\":\"%c\",\"state\":\"%s\",\"tx_packets\":%llu,\"rx_packets\":%llu,\"seq_gaps\":%llu,\"master_minus_local_us\":%lld,\"sync_frames\":%"PRIu64",\"software_phase_mean_us\":%.3f,\"software_phase_rms_us\":%.3f,\"software_phase_min_us\":%"PRId64",\"software_phase_max_us\":%"PRId64",\"max_clock_step_us\":%"PRId64",\"snr_min_db\":%.2f,\"snr_max_db\":%.2f,\"runtime_network\":\"LoRa_only\",\"tx_parameter_dbm\":%d,\"pa_config\":\"%s\",\"rssi_last_dbm\":%s,\"snr_last_db\":%s,\"rx_age_ms\":%.3f,\"epoch\":%.9f}\n",call_role?'B':'A',atomic_load(&call_end)?"ending":"in_call",atomic_load(&call_tx_count),atomic_load(&call_rx_count),atomic_load(&call_gap_count),atomic_load(&call_offset_us),count,count?call_phase_sum/count:0,count?sqrt(call_phase_sq/count):0,count?call_phase_min:0,count?call_phase_max:0,call_clock_step_max,call_snr_min/4.0,call_snr_max/4.0,call_tx_parameter_dbm,call_pa_config,last_rssi,last_snr,(now-call_last_receive)*1000,epoch_s());
   pthread_mutex_unlock(&call_stats_mu);if(!fclose(f))rename(tmp,path);
  }
 }
 return call_end_deadline&&now>=call_end_deadline?3:0;
}
