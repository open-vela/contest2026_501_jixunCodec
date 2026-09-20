############################################################################
# apps/jixun-codec/Makefile
#
# 极讯 AI Codec 上板验证 app
#
#   make JX_RATE=3k      # 或 1k / 6k，选择编译进固件的码率（默认 3k）
#
# 音频外设层随应用携带，避免依赖官方分支中不存在的 plant-companion。
# 底层 hal_i2s 实现由 board/esp32s3-box 的 NuttX 补丁提供。
############################################################################

include $(APPDIR)/Make.defs

# ---- 码率选择（构建期）----------------------------------------------------
JX_RATE ?= 3k
JX_RATE_UC := $(shell echo $(JX_RATE) | tr a-z A-Z)

# ---- 随应用携带的音频层 ---------------------------------------------------
AUDIO = $(APPDIR)/jixun-codec/deps/audio
CFLAGS += -I$(AUDIO)/hal
CFLAGS += -I$(AUDIO)/components/voice_agent
CFLAGS += -I$(AUDIO)/ai_module/ai_voice
CFLAGS += -DPLANT_NO_NS=1

# hal_i2s.h / hal_gpio.h 包的是 arch 的芯片头（与 plant-companion 同处理）
ifneq ($(wildcard $(TOPDIR)/arch/xtensa/src/esp32s3),)
CFLAGS += -I$(TOPDIR)/arch/xtensa/src/esp32s3
endif

# ---- 我们自己的 codec -----------------------------------------------------
CFLAGS += -Iinclude
CFLAGS += -Iport
CFLAGS += -Iweights/$(JX_RATE)
CFLAGS += -DSQM_RATE_$(JX_RATE_UC)
CFLAGS += -O3
CFLAGS += -DJX_NO_PROFILE
CFLAGS += -DJX_SAFE_GETENV=1 -include $(APPDIR)/jixun-codec/include/jx_env.h
# r81d 实测反例：-ffp-contract=off 使 3980 -> 4230 ms，已回退。
# r84 实测反例：-funroll-loops -frename-registers 使 3980 -> 4100 ms，已回退。
# r89：寄存器压力相关的调度/分配选项。反汇编证据（xtensa 交叉编译器，只编
# int8_kernels.c 后统计）：jx_linear_i8_rows 的 FP 栈溢出条数
#     基线 246 -> -fira-loop-pressure 233 -> -fweb 228 -> 两者合用 217
#     （再加 -fira-region=all 为 206，但它同时把指令数从 3587 抬到 3591）
# 这两项只影响寄存器分配与 web 分解，不改变任何表达式的求值次序 -> 数值不变。
CFLAGS += -fweb -fira-loop-pressure

# ESP32-S3 的 PIE 128-bit 向量指令（int8 点积内核用）
ifeq ($(CONFIG_ARCH_CHIP_ESP32S3),y)
CFLAGS += -DJX_PIE=1
endif

# codec 核心（与 PC 端 sqcodec-light-c-v1/src 同源）
CSRCS += csrc/blocks.c
CSRCS += csrc/decoder.c
CSRCS += csrc/encoder.c
CSRCS += csrc/fsq_codec.c
CSRCS += csrc/nn_ops.c
CSRCS += csrc/quantizer.c
CSRCS += csrc/transformer.c
CSRCS += csrc/model_config.c
CSRCS += csrc/int8_kernels.c
CSRCS += csrc/jx_amax.c
CSRCS += csrc/jx_env.c

# 板端封装
CSRCS += port/jx_codec.c
CSRCS += port/jx_net.c
CSRCS += port/jx_serial.c
CSRCS += port/jx_streamtx.c
CSRCS += port/jx_ui.c

# 音频外设层（ES7210 + ES8311 + AI voice）
CSRCS += deps/audio/components/voice_agent/es7210.c
CSRCS += deps/audio/components/voice_agent/es8311.c
CSRCS += deps/audio/components/voice_agent/voice_agent.c
CSRCS += deps/audio/ai_module/ai_voice/ai_voice.c

# 第四十轮：P0 可行性探针（独立编译单元，避免扰动 jx_main.c 的寄存器分配）
CSRCS += main/jx_p0probe.c

# ---- 程序注册 -------------------------------------------------------------
MAINSRC   = main/jx_main.c
PROGNAME  = jixun
PRIORITY  = 100
STACKSIZE = 32768

include $(APPDIR)/Application.mk
