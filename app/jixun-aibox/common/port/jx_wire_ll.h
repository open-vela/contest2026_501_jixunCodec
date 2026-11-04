#ifndef JX_WIRE_LL_H
#define JX_WIRE_LL_H
#include "jx_wire_v2.h"
/* Type0x23 retains the legacy token payload, sequence, EOS and halo fields.
 * Header bytes8..9 are token position;10 is planned take seconds,11 is tag.
 * CRC covers header and payload. Type0x21 remains accepted without hints. */
#define JX_LL_TYPE 0x23
static inline int jxll_tag(uint8_t *p,unsigned seconds) {
 if(!p||seconds<1||seconds>8)return -1;
 p[2]=JX_LL_TYPE;p[10]=(uint8_t)seconds;p[11]=0xa5;
 uint16_t crc=jxw_crc(p);p[14]=(uint8_t)crc;p[15]=(uint8_t)(crc>>8);return 0;
}
static inline int jxll_seconds(const uint8_t *p,size_t n) {
 if(!p||n!=52||p[0]!='J'||p[1]!='X'||p[2]!=JX_LL_TYPE||
    p[11]!=0xa5||p[10]<1||p[10]>8||p[12]!=16||p[13]!=0||
    ((uint16_t)p[14]|(uint16_t)p[15]<<8)!=jxw_crc(p))return 0;
 return p[10];
}
#endif
