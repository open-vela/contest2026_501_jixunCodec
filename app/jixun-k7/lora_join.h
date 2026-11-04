/* Single-master/single-slave LoRa test admission. CRC detects corruption;
 * node identifiers are routing identifiers, not authentication credentials. */
#include <unistd.h>
#include <sys/random.h>

static uint32_t join_cookie;
static void join_put32(uint8_t *p, uint32_t x) {
    for (int i=0;i<4;i++) p[i]=(uint8_t)(x>>(8*i));
}
static uint32_t join_get32(const uint8_t *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static int join_arm(sx1262_t *d) {
    if(sx1262_set_standby_xosc(d))return -1;
    sx1262_sw_rx(d);
    if(sx1262_clear_irq(d,0xFFFF))return -1;
    return sx1262_start_rx(d,500);
}
static int join_send(sx1262_t *d,int type,uint32_t cookie) {
    uint8_t p[PKT_BYTES]={0};
    p[0]='J';p[1]='R';p[2]=1;p[3]=(uint8_t)type;
    join_put32(p+4,0x10000001u); /* master */
    join_put32(p+8,0x20000002u); /* slave */
    join_put32(p+12,cookie);
    uint16_t c=crc16(p,PKT_BYTES-2);
    p[PKT_BYTES-2]=(uint8_t)c;p[PKT_BYTES-1]=(uint8_t)(c>>8);
    if(sx1262_set_standby_xosc(d))return -1;
    sx1262_sw_tx(d);
    if(sx1262_set_lora_payload_len(d,PKT_BYTES)||
       sx1262_write_buffer(d,0,p,PKT_BYTES)||sx1262_clear_irq(d,0xFFFF))return -1;
    double start=now_s();
    if(sx1262_start_tx(d,500))return -1;
    int done=0;
    while(now_s()-start<0.8) {
        uint16_t irq=0;
        if(sx1262_get_irq(d,&irq)||irq==0xFFFF)return -1;
        if(irq&SX_IRQ_TIMEOUT)break;
        if(irq&SX_IRQ_TX_DONE) {done=now_s()-start>=0.020;break;}
        struct timespec pause={0,1000000};nanosleep(&pause,NULL);
    }
    if(join_arm(d))return -1;
    fprintf(stderr,"[join tx type=%d cookie=%08x done=%d]\n",type,cookie,done);
    return done?0:-1;
}
/* 1: valid frame, 0: no frame, -1: hardware I/O error. */
static int join_receive(sx1262_t *d,int *type,uint32_t *cookie) {
    uint16_t irq=0;
    if(sx1262_get_irq(d,&irq)||irq==0xFFFF)return -1;
    if(!irq)return 0;
    if(irq&SX_IRQ_RX_DONE) {
        uint8_t info[2]={0},p[PKT_BYTES];
        if(sx1262_cmd_read_public(d,0x13,NULL,0,1,info,2))return -1;
        int valid=!(irq&(SX_IRQ_CRC_ERR|SX_IRQ_HEADER_ERR))&&info[0]==PKT_BYTES;
        if(valid&&sx1262_read_rx_payload(d,info[1],p,PKT_BYTES))return -1;
        if(join_arm(d))return -1;
        if(!valid)return 0;
        uint16_t c=(uint16_t)(p[PKT_BYTES-2]|((uint16_t)p[PKT_BYTES-1]<<8));
        if(p[0]!='J'||p[1]!='R'||p[2]!=1||c!=crc16(p,PKT_BYTES-2)||
           join_get32(p+4)!=0x10000001u||join_get32(p+8)!=0x20000002u)return 0;
        *type=p[3];*cookie=join_get32(p+12);
        return *cookie&&*type>=1&&*type<=4?1:0;
    }
    if(irq&(SX_IRQ_TIMEOUT|SX_IRQ_CRC_ERR|SX_IRQ_HEADER_ERR))return join_arm(d)?-1:0;
    return 0;
}
static int lora_join(sx1262_t *d,int slave,double seconds) {
    uint32_t cookie=0;
    if(slave) {
        if(getrandom(&cookie,sizeof(cookie),0)!=(ssize_t)sizeof(cookie))
            cookie=(uint32_t)(now_s()*1000000.0)^(uint32_t)getpid();
        if(!cookie)cookie=1;
    }
    if(join_arm(d))return 2;
    double start=now_s(),next=0;
    int response_seen=0,requests=0,acks=0;
    fprintf(stderr,"[join begin role=%c master=10000001 slave=20000002 timeout=%.0fs]\n",slave?'B':'A',seconds);
    while(!g_cancelled&&now_s()-start<seconds) {
        if(slave&&now_s()>=next) {
            int type=response_seen?3:1;
            if(join_send(d,type,cookie))return 3;
            if(type==1)requests++;else acks++;
            next=now_s()+0.7;
        }
        int type=0;uint32_t received=0;
        int rc=join_receive(d,&type,&received);
        if(rc<0)return 4;
        if(rc>0) {
            fprintf(stderr,"[join rx type=%d cookie=%08x]\n",type,received);
            struct timespec settle={0,20000000};nanosleep(&settle,NULL);
            if(!slave&&type==1) {
                cookie=received;
                if(join_send(d,2,cookie))return 5;
            } else if(!slave&&type==3&&cookie&&received==cookie) {
                for(int i=0;i<3;i++) {
                    if(join_send(d,4,cookie))return 6;
                    struct timespec gap={0,20000000};nanosleep(&gap,NULL);
                }
                join_cookie=cookie;
                fprintf(stderr,"[join registered role=A peer=20000002 cookie=%08x elapsed=%.3fs]\n",cookie,now_s()-start);
                return 0;
            } else if(slave&&received==cookie&&type==2) {
                response_seen=1;next=0;
            } else if(slave&&received==cookie&&type==4&&response_seen&&acks>0) {
                /* Master finishes its bounded READY retransmissions before either audio clock starts. */
                struct timespec barrier={0,300000000};nanosleep(&barrier,NULL);
                join_cookie=cookie;
                fprintf(stderr,"[join registered role=B peer=10000001 cookie=%08x elapsed=%.3fs req=%d ack=%d]\n",cookie,now_s()-start,requests,acks);
                return 0;
            }
        }
        struct timespec pause={0,1000000};nanosleep(&pause,NULL);
    }
    fprintf(stderr,"[join timeout role=%c voice_blocked=1]\n",slave?'B':'A');
    return 7;
}
