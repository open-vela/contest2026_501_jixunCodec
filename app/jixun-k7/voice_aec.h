/* SpeexDSP stable C ABI; runtime library is installed on both boards.
 * Reference samples are indexed by their estimated physical playback time.
 * Only the capture thread owns the adaptive filter. */
#ifndef VOICE_AEC_H
#define VOICE_AEC_H
#include <stdint.h>
#include <pthread.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
typedef struct SpeexEchoState_ SpeexEchoState;
extern SpeexEchoState *speex_echo_state_init(int,int);
extern void speex_echo_state_destroy(SpeexEchoState *);
extern int speex_echo_ctl(SpeexEchoState *,int,void *);
extern void speex_echo_cancellation(SpeexEchoState *,const int16_t *,const int16_t *,int16_t *);
typedef struct SpeexPreprocessState_ SpeexPreprocessState;
extern SpeexPreprocessState *speex_preprocess_state_init(int,int);
extern void speex_preprocess_state_destroy(SpeexPreprocessState *);
extern int speex_preprocess_ctl(SpeexPreprocessState *,int,void *);
extern int speex_preprocess_run(SpeexPreprocessState *,int16_t *);
#define AEC_FRAME 128
#define AEC_RING 131072
typedef struct {
    SpeexEchoState *echo;
    SpeexPreprocessState *noise;
    pthread_mutex_t mu;
    int16_t ref[AEC_RING];
    int64_t stamp[AEC_RING];
    unsigned long frames, reference_frames;
    double raw_energy, clean_energy;
} VoiceAec;
static VoiceAec voice_aec;
static int aec_init(void) {
    memset(&voice_aec,0,sizeof(voice_aec));
    for(int i=0;i<AEC_RING;i++)voice_aec.stamp[i]=-1;
    pthread_mutex_init(&voice_aec.mu,NULL);
    voice_aec.echo=speex_echo_state_init(AEC_FRAME,4096);
    int rate=16000;
    if(!voice_aec.echo)return -1;
    if(speex_echo_ctl(voice_aec.echo,24,&rate))return -1;
    voice_aec.noise=speex_preprocess_state_init(AEC_FRAME,rate);
    if(!voice_aec.noise)return -1;
    int on=1,off=0,noise_db=-20,echo_db=-35,active_db=-12;
    if(speex_preprocess_ctl(voice_aec.noise,0,&on)||
       speex_preprocess_ctl(voice_aec.noise,2,&off)||
       speex_preprocess_ctl(voice_aec.noise,18,&noise_db)||
       speex_preprocess_ctl(voice_aec.noise,20,&echo_db)||
       speex_preprocess_ctl(voice_aec.noise,22,&active_db)||
       speex_preprocess_ctl(voice_aec.noise,24,voice_aec.echo))return -1;
    fprintf(stderr,"[aec enabled engine=SpeexDSP rate=16000 frame=128 tail_ms=256 reference=alsa_playback_time]\n");
    fprintf(stderr,"[noise enabled suppress_db=-20 agc=off residual_echo=on]\n");
    return 0;
}
static void aec_reference(const int16_t *x,int n,double start) {
    int64_t base=llround(start*16000.0);
    pthread_mutex_lock(&voice_aec.mu);
    for(int i=0;i<n;i++) {
        int64_t tick=base+i;unsigned k=(uint64_t)tick%AEC_RING;
        voice_aec.ref[k]=x[i];voice_aec.stamp[k]=tick;
    }
    pthread_mutex_unlock(&voice_aec.mu);
}
static void aec_capture(int16_t *x,int n,double start) {
    int16_t ref[AEC_FRAME],clean[AEC_FRAME];
    for(int off=0;off<n;off+=AEC_FRAME) {
        int64_t base=llround(start*16000.0)+off;int nonzero=0;
        pthread_mutex_lock(&voice_aec.mu);
        for(int i=0;i<AEC_FRAME;i++) {
            int64_t tick=base+i;unsigned k=(uint64_t)tick%AEC_RING;
            ref[i]=voice_aec.stamp[k]==tick?voice_aec.ref[k]:0;
            nonzero|=ref[i]!=0;
        }
        pthread_mutex_unlock(&voice_aec.mu);
        speex_echo_cancellation(voice_aec.echo,x+off,ref,clean);
        for(int i=0;i<AEC_FRAME;i++) {
            voice_aec.raw_energy+=(double)x[off+i]*x[off+i];
            voice_aec.clean_energy+=(double)clean[i]*clean[i];
        }
        memcpy(x+off,clean,sizeof(clean));voice_aec.frames++;
        voice_aec.reference_frames+=nonzero;
    }
    if(voice_aec.frames%375==0)
        fprintf(stderr,"[aec active frames=%lu reference_frames=%lu power_change_db=%.2f]\n",
            voice_aec.frames,voice_aec.reference_frames,
            10*log10((voice_aec.raw_energy+1)/(voice_aec.clean_energy+1)));
}
static void aec_destroy(void) {
    speex_preprocess_state_destroy(voice_aec.noise);voice_aec.noise=NULL;
    speex_echo_state_destroy(voice_aec.echo);voice_aec.echo=NULL;
    pthread_mutex_destroy(&voice_aec.mu);
}
static void aec_denoise(int16_t *x,int n) {
    for(int off=0;off<n;off+=AEC_FRAME)speex_preprocess_run(voice_aec.noise,x+off);
}
#endif
