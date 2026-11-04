include $(APPDIR)/Make.defs
PROGNAME = k7_voice k7_access k7_pair
PRIORITY = 100 100 100
STACKSIZE = 32768 32768 8192
MAINSRC = k7_voice_main.c k7_access_main.c k7_pair_main.c
MODULE = $(CONFIG_EXAMPLES_JIXUN_K7)
CSRCS = native_runtime.c rf/k7_gpio.c rf/k7_native_spi.c rf/sx1262_acceptance.c
CSRCS += $(addprefix A_fixed/src/,nn_ops.c quantizer.c fsq_codec.c model_config.c stream_ops.c stream_trans.c stream_codec.c)
CSRCS += A_fixed/src/jx_neon.c
CSRCS += A_fixed/src/jx_parallel.c
CSRCS += $(addprefix speex/libspeexdsp/,mdf.c preprocess.c filterbank.c fftwrap.c smallft.c buffer.c scal.c)
CFLAGS += -O3 -DSQM_RATE_3K -DFLOATING_POINT -DUSE_SMALLFT -DEXPORT= -DVAR_ARRAYS
CFLAGS += -DJX_NATIVE_NEON
CFLAGS += -I$(CURDIR) -I$(CURDIR)/A_fixed/header -I$(CURDIR)/A_fixed/weights
CFLAGS += -I$(CURDIR)/speex/include -I$(CURDIR)/speex/libspeexdsp
CFLAGS += -I$(TOPDIR)/../vendor/rockchip/chips/rk3576
CFLAGS += -Wno-error=misleading-indentation -Wno-error=unused-variable -Wno-error=unused-function -Wno-error=shadow
ifeq ($(CONFIG_EXAMPLES_JIXUN_K7_UI),y)
PROGNAME += k7_ui
PRIORITY += 80
STACKSIZE += 65536
MAINSRC += k7_ui_main.c
CSRCS += ui/jx_original_assets.c
CSRCS += $(addprefix ui/jx_font_,16.c 17.c 23.c 27.c 30.c 35.c)
CFLAGS += -I$(APPDIR)/graphics/lvgl
endif
include $(APPDIR)/Application.mk
