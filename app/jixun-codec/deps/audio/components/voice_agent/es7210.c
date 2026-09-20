/****************************************************************************
 * apps/plant-companion/components/voice_agent/es7210.c
 *
 * ES7210 ADC driver — 完全按 IDF espressif__esp_codec_dev 的寄存器配置。
 * 参考: managed_components/espressif__esp_codec_dev/device/es7210/es7210.c
 *
 * 关键配置：Slave 模式、I2S Standard、16-bit、24kHz @ MCLK=6.144MHz
 * （2 麦标准立体声：MIC1=左、MIC2=右；2026-09-02 起 REG08=0x00 对齐
 *  2 槽 RX，详见阶段 3 注释）
 ****************************************************************************/

#include <nuttx/config.h>
#include <stdio.h>
#include <errno.h>

#include <nuttx/i2c/i2c_master.h>
#include "../../hal/hal_i2c.h"
#include "es7210.h"

#define I2C_ADDR  0x40
#define I2C_FREQ  100000

/****************************************************************************
 * 时钟系数（IDF coeff_div 表）
 *
 * MCLK=6.144MHz, LRCK=24kHz: (xiaozhi 实测值)
 *   adc_div=1, doubler=0, dll=1, osr=0x20, lrck_h=0x02, lrck_l=0x00
 *   MCLK/LRCK = 512 = 256 × 2（标准配置）
 ****************************************************************************/

/* 与 IDF es7210_config_sample() 对应的寄存器写入 */

static int es7210_config_clock_24k(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  /* REG02: adc_div=1, doubler=1, dll=1
   * bit[7]=dll(1), bit[6]=doubler(1), bit[3:0]=adc_div(1)
   * ⚠️ 根因修复：slave 模式必须 doubler=1（官方 es7210_open 写 0xc1）。
   *    之前写 0x81（doubler=0）是 master 模式的 coeff 值——
   *    内部 ADC 时钟减半 → 调制器采样不完整 → 数据稀疏。
   *    官方 es7210_config_sample 在 slave 模式直接跳过，REG02 保持 0xc1。
   * 注：官方 slave 模式不写 REG04/05（LRCK 分频由外部 LRCK 决定），
   *    也不写 REG07 以外的时钟寄存器——这里只按官方 open 写 REG02+REG07。 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MAINCLK, 0xc1, I2C_FREQ);

  /* REG07: OSR = 0x20（官方 open 同值） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_OSR, 0x20, I2C_FREQ);

  return ret;
}

/****************************************************************************
 * 公共 API：初始化（IDF es7210_open + es7210_start 完整流程）
 *
 * 流程：
 *   1. 复位（0xff → 0x41）
 *   2. 时钟关闭 + 时间控制 + HPF
 *   3. Slave 模式
 *   4. 模拟电源 + MIC 偏置
 *   5. I2S 格式（16-bit Standard）
 *   6. 时钟系数（24kHz @ 12.288MHz MCLK）
 *   7. MIC 增益 + ADC 使能
 *   8. 最终复位序列启动
 ****************************************************************************/

int es7210_init(FAR struct i2c_master_s *i2c,
                uint32_t sample_rate, uint32_t mclk_hz)
{
  int ret = 0;
  uint8_t gain;

  (void)sample_rate;
  (void)mclk_hz;

  /* ==== 阶段 1：复位（IDF es7210_open 第一步） ==== */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_RESET, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_RESET, 0x41, I2C_FREQ);

  /* ==== 阶段 2：时钟关闭 + 时间控制 + HPF ==== */

  /* REG01: 关闭所有时钟（初始化前的干净状态） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_CLK_OFF, 0x3F, I2C_FREQ);

  /* REG09/0A: 芯片状态周期 + 上电状态周期 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_TIME_CTRL0, 0x30, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_TIME_CTRL1, 0x30, I2C_FREQ);

  /* REG22/23: ADC1+2 高通滤波器快速设置 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ADC12_HPF2, 0x2A, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ADC12_HPF1, 0x0A, I2C_FREQ);

  /* REG20/21: ADC3+4 高通滤波器（虽然不用，但 IDF 也配了） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ADC34_HPF2, 0x0A, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ADC34_HPF1, 0x2A, I2C_FREQ);

  /* ==== 阶段 3：Slave 模式 + LRCK_RATE_MODE ==== */

  /* REG08: bit0(MS_MODE)=0 → Slave（BCK/WS 由 ESP32-S3 I2S master 提供）
   * bits[7:4]=LRCK_RATE_MODE=1（0x10）。
   *
   * ⚠️ 2026-09-02 回退定案（勿再改！）——对齐 2026-08-25 已验证基线：
   *   · VOICE_WORKLOG 2026-08-25 ✅里程碑：官方驱动 2 槽 RX + ES7210
   *     2 麦标准模式（REG12=0x00、REG01=0x34、MIC3/4 关）+ REG08=0x10
   *     → 说话峰值 7065-14053，用户"有环境声和人声了"（已实机验证）。
   *   · 改 0x00（LRCK_RATE_MODE=0）→ 实测 1/8 稀疏（raw 每 16 采样一对
   *     小值，94% 零，diag"未拾音"）——旧文档 08-20 同记录。
   *   · 改 4 槽总线（total_slot=4）→ TX 仍 2 槽时 WS 变 48kHz，ES7210
   *     锁不上 MCLK/LRCK 比 → RX 0 数据（本次实测 RX FIFO 超时）。
   *   → 就保持官方驱动 2 槽/32bit 帧 + REG08=0x10，这是唯一实测可用的组合。 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MODE_CFG, 0x10, I2C_FREQ);

  /* ==== 阶段 4：模拟电源 + MIC 偏置 ==== */

  /* REG40: LDO + REF + VMID 使能，VDDA=3.3V */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ANALOG_SYS, 0x43, I2C_FREQ);

  /* REG41/42: MIC 偏置电压 2.87V（IDF 默认值 0x70） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_BIAS, 0x70, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_BIAS, 0x70, I2C_FREQ);

  /* ==== 阶段 5：I2S 数据格式（16-bit Standard） ==== */

  /* REG11: [7:5]=bit_width(0x60=16-bit), [1:0]=fmt(00=I2S Standard)
   * IDF es7210_set_bits(16): adc_iface |= 0x60
   * IDF es7210_config_fmt(NORMAL): adc_iface |= 0x00
   * 合并: 0x60 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_SDP_INTERFACE1, 0x60, I2C_FREQ);

  /* REG12: 标准 I2S 2 通道（2026-08-25：迁移官方 esp32s3_i2s 驱动！
   * 官方驱动只支持 1/2 通道（total_slot=2、32bit 帧），ES7210 配 2 麦
   * 标准模式：REG12=0x00（I2S 立体声），LRCK 低=左(MIC1)、高=右(MIC2)，
   * RX 2 槽取左声道 = MIC1 24k。xiaozhi es7210_mic_select：麦数 <3 时
   * 同样写 REG12=0x00（非 TDM）。 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_SDP_INTERFACE2, 0x00, I2C_FREQ);

  /* ==== 阶段 6：时钟系数（24kHz @ MCLK=6.144MHz） ==== */

  ret |= es7210_config_clock_24k(i2c);

  /* ==== 阶段 7：MIC 增益 + 电源（2 麦模式：只 MIC1/2，MIC3/4 关闭） ==== */

  /* REG47-4A: MIC1-4 PGA 电源使能（2 麦：MIC1/2 开，MIC3/4 关=0xFF） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_POWER, 0xFF, I2C_FREQ);

  /* REG4B/4C: MIC1-4 电源控制（先全关再开，IDF mic_select 流程；
   * 2 麦只操作 MIC12(REG4B)，MIC34(REG4C) 保持关） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_POWER, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_POWER, 0xFF, I2C_FREQ);

  /* REG43-46: MIC1-4 PGA 增益。
   * 格式（IDF es7210_reg.h）：bit4=0x10 PGA 使能，bits[3:0]=增益值。
   * ⚠️ 2026-09-02 降 32dB→27dB（值 11→9）：diag/raw 实测说话时 24k 原始
   * 流削顶（raw 峰值 32671≈满幅 32767）→ 削顶失真 → "录音全是噪声"。
   * 历史参考：37.5dB(14) 实测削顶；30dB(10) = 2026-08-25 已验证基线
   * （说话峰值 7065-14053，用户"有环境声和人声了"）；27dB(9) 干净但偏弱。
   * ⚠️ 2026-09-02 回退 30dB——27dB 是降噪链路（服务器 noisereduce）的
   * 折中；当前设备端直放回放场景下 27dB 偏弱易被底噪掩盖。 */

  gain = (1 << 4) | 10;   /* ~30dB（2026-08-25 验证基线） */
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_GAIN, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_GAIN, 0x00, I2C_FREQ);

  /* ==== 阶段 8：解除静音 + 时钟 + 最终启动 ==== */

  /* ⚠️ 根因修复：解除 ES7210 静音（官方 _es7210_set_mute(false)）
   * REG14/15 bits[1:0] = 00。默认/上电可能是静音(11)，
   * 之前从不写这两个寄存器 → ADC 大部分时间输出零（只有大声时才
   * 短暂解除）→ 数据 84% 零值。 */

  {
    uint8_t mv;

    ret |= hal_i2c_read_reg(i2c, I2C_ADDR, 0x14, &mv, I2C_FREQ);
    mv &= ~0x03;
    ret |= hal_i2c_write_reg(i2c, I2C_ADDR, 0x14, mv, I2C_FREQ);
    ret |= hal_i2c_read_reg(i2c, I2C_ADDR, 0x15, &mv, I2C_FREQ);
    mv &= ~0x03;
    ret |= hal_i2c_write_reg(i2c, I2C_ADDR, 0x15, mv, I2C_FREQ);
  }

  /* REG40: 再次确认模拟电源 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ANALOG_SYS, 0x43, I2C_FREQ);

  /* REG01: 打开 ADC 时钟（2 麦：只 MIC1/2）。
   * 0x3F 清掉 MIC1/2 时钟(0x0b) → 0x34；MIC3/4 时钟(0x15) 保持关。
   * （xiaozhi 2 麦序列：只 update_reg_bit(REG01, 0x0b, 0x00)） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_CLK_OFF, 0x34, I2C_FREQ);

  /* REG06: 解除 power down */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_POWER_DOWN, 0x00, I2C_FREQ);

  /* REG00: 最终复位序列（IDF es7210_start 最后两步） */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_RESET, 0x71, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_RESET, 0x41, I2C_FREQ);

  /* ⚠️ 根因修复：RESET 之后必须重配模拟电源！
   * 实测读回 REG40=0x42（bit0 LDO_EN 丢失 = 0）→ VMID 参考未建立
   * → ADC 输入共模错误 → 输入饱和 → 确定性极限环（麦克风无声）。
   * 官方顺序 RESET 在最后，但本芯片 RESET 会把模拟配置复位，
   * 所以 RESET 后重新写全部模拟寄存器（不只 MIC1！）：
   * REG40(模拟电源+VMID)、REG41/42(MICBIAS 2.87V)、
   * REG43-46(MIC1-4 增益)、REG47-4A(MIC1-4 PGA 电源)、
   * REG4B/4C(MIC12/34 电源控制)。 */

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ANALOG_SYS, 0x43, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_POWER, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_BIAS, 0x70, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_BIAS, 0x70, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_GAIN, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_GAIN, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_POWER, 0xFF, I2C_FREQ);

  /* 读回验证关键模拟寄存器 */

  {
    uint8_t rv40, rv4b, rv4c;

    hal_i2c_read_reg(i2c, I2C_ADDR, 0x40, &rv40, I2C_FREQ);
    hal_i2c_read_reg(i2c, I2C_ADDR, 0x4b, &rv4b, I2C_FREQ);
    hal_i2c_read_reg(i2c, I2C_ADDR, 0x4c, &rv4c, I2C_FREQ);
    printf("[ES7210] 模拟电源验证: REG40=0x%02x(期望0x43) "
           "REG4B=0x%02x(期望0x00,PDN全开) REG4C=0x%02x\n",
           rv40, rv4b, rv4c);
  }

  if (ret < 0)
    {
      printf("[ES7210] init failed (I2C error)\n");
      return ret;
    }

  printf("[ES7210] init OK: Slave, I2S Std, 16-bit, 24kHz, "
         "MCLK=6.154MHz (TDM)\n");
  return 0;
}

/****************************************************************************
 * 公共 API：恢复模拟配置（MCLK 启动后必须调用！）
 *
 * ⚠️ 2026-08-21 实测根因：es7210_init 时 GPIO2 尚未配置为 I2S0_MCLK
 * （MCLK 由 hal_i2s_init 稍后启动）——ES7210 在【无 MCLK】状态下完成
 * 初始化，之后 MCLK 突然出现会使 REG40 bit0(LDO_EN) 被清
 * （读回 0x43 → 0x42）→ VMID 参考未建立 → ADC 输入共模错误 →
 * 输入饱和 → 确定性极限环噪声（麦克风无声，只有时钟嘶声）。
 *
 * 因此 voice_agent_init 必须在 hal_i2s_start_tx_clock()【之后】调用本
 * 函数：完整重写模拟配置（REG40/41/42/43-46/47-4A/4B/4C）并读回验证。
 * 若不调用，录音只会采到极限环噪声（实测峰值 16657/32768 确定性图案）。
 ****************************************************************************/

int es7210_restore_analog(FAR struct i2c_master_s *i2c)
{
  int ret = 0;
  uint8_t gain = (1 << 4) | 9;   /* ~27dB（2026-09-02 定：raw 铁证 32dB 削顶）*/
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ANALOG_SYS, 0x43, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_POWER, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC12_BIAS, 0x70, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC34_BIAS, 0x70, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_GAIN, gain, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_GAIN, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_GAIN, 0x00, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC1_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC2_POWER, 0x08, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC3_POWER, 0xFF, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_MIC4_POWER, 0xFF, I2C_FREQ);

  {
    uint8_t rv40;

    hal_i2c_read_reg(i2c, I2C_ADDR, 0x40, &rv40, I2C_FREQ);
    printf("[ES7210] restore_analog: REG40=0x%02x%s\n",
           rv40, rv40 == 0x43 ? " (VMID/LDO OK)" :
                " ← 仍异常！");
  }

  return ret;
}

/****************************************************************************
 * 公共 API：ES7210 断电/恢复（rxprobe 变量分离实验用）
 *
 * power_down：REG06=0x07（全 power down）+ REG40=0xC0（模拟关）
 * power_up：REG06=0x00 + 完整恢复模拟配置
 *
 * ⚠️ 用法：plant voice rxprobe —— 阶段2 断电后若 RX 仍收到同样的大值
 * 噪声（>40% 采样 >2000）→ 问题在 RX 通路/DIN 引脚/解算（与 ES7210
 * 无关）；若 RX 变全 0/干净 → 噪声由 ES7210 输出产生（模拟/时钟）。
 ****************************************************************************/

int es7210_power_down(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_POWER_DOWN, 0x07, I2C_FREQ);
  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_ANALOG_SYS, 0xC0, I2C_FREQ);
  return ret;
}

int es7210_power_up(FAR struct i2c_master_s *i2c)
{
  int ret = 0;

  ret |= hal_i2c_write_reg(i2c, I2C_ADDR, ES7210_POWER_DOWN, 0x00, I2C_FREQ);
  ret |= es7210_restore_analog(i2c);
  return ret;
}
