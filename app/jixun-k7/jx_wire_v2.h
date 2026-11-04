#ifndef JX_WIRE_V2_H
#define JX_WIRE_V2_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
/* Fixed52byte voice frame. Type0x22 is distinct from legacy0x21.
 * 0..1 JX,2type,3 EOS/halo/model,4..7session,8..9seq9bits+valid4bits,
 * 10..13 original PCM sample count on EOS only,14..15 CRC16 over header
 * 0..13 and payload16..51. Payload is still16*18bit tokens. */
#define JXW_BYTES 52
#define JXW_ROWS 512u
#define JXW_MODEL_249_3K 0u
#define JXW_MODEL_LIGHT_A_3K 1u
typedef struct {
 uint32_t session,seq,samples;
 unsigned valid,halo,model,eos;
 int64_t tokens[16];
} JxWireFrame;
static inline uint32_t jxw_rd32(const uint8_t*p){return (uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;}
static inline void jxw_wr32(uint8_t*p,uint32_t v){p[0]=v;p[1]=v>>8;p[2]=v>>16;p[3]=v>>24;}
static inline uint16_t jxw_crc_step(uint16_t c,uint8_t v){c^=(uint16_t)v<<8;for(int b=0;b<8;b++)c=(uint16_t)((c<<1)^((c&0x8000u)?0x1021u:0));return c;}
static inline uint16_t jxw_crc(const uint8_t*p){uint16_t c=0xffffu;for(int i=0;i<14;i++)c=jxw_crc_step(c,p[i]);for(int i=16;i<52;i++)c=jxw_crc_step(c,p[i]);return c;}
static inline int jxw_pack(uint8_t*p,const JxWireFrame*f){
 if(!p||!f||!f->session||f->seq>=JXW_ROWS||!f->valid||f->valid>16||f->halo>5||f->model>2||f->eos>1||((f->samples!=0)!=!!f->eos))return -1;
 for(unsigned i=0;i<f->valid;i++)if(f->tokens[i]<0||f->tokens[i]>0x3ffff)return -1;
 memset(p,0,JXW_BYTES);p[0]='J';p[1]='X';p[2]=0x22;p[3]=f->eos|(f->halo<<1)|(f->model<<4);
 jxw_wr32(p+4,f->session);uint16_t v=(uint16_t)(f->seq|((f->valid-1)<<9));p[8]=v;p[9]=v>>8;jxw_wr32(p+10,f->samples);
 for(unsigned i=0;i<f->valid;i++)for(unsigned b=0;b<18;b++)if(((uint32_t)f->tokens[i]>>b)&1)p[16+((i*18+b)>>3)]|=(uint8_t)(1u<<((i*18+b)&7));
 uint16_t crc=jxw_crc(p);p[14]=crc;p[15]=crc>>8;return 0;
}
static inline int jxw_unpack(JxWireFrame*f,const uint8_t*p,size_t n){
 if(!f||!p||n!=JXW_BYTES||p[0]!='J'||p[1]!='X'||p[2]!=0x22||(p[3]&0xc0)||((uint16_t)p[14]|(uint16_t)p[15]<<8)!=jxw_crc(p))return -1;
 uint16_t v=(uint16_t)p[8]|(uint16_t)p[9]<<8;if(v&0xe000u)return -1;
 memset(f,0,sizeof(*f));f->session=jxw_rd32(p+4);f->seq=v&0x1ffu;f->valid=((v>>9)&15u)+1;f->eos=p[3]&1;f->halo=(p[3]>>1)&7;f->model=(p[3]>>4)&3;f->samples=jxw_rd32(p+10);
 if(!f->session||f->halo>5||f->model>2||((f->samples!=0)!=!!f->eos))return -1;
 for(unsigned i=0;i<16;i++)for(unsigned b=0;b<18;b++)if((p[16+((i*18+b)>>3)]>>((i*18+b)&7))&1)f->tokens[i]|=(int64_t)1<<b;
 for(unsigned i=f->valid;i<16;i++)if(f->tokens[i])return -1;
 return 0;
}
#endif
