/* 240x240 product view. All drawing uses the existing RGB565 band buffer.
 * No animation, timers, waveform synthesis or additional framebuffer. */
#include "jx_simple_logo.h"

enum simple_phase
{
  SIMPLE_IDLE, SIMPLE_PREP, SIMPLE_RECORD, SIMPLE_SEND, SIMPLE_SENT,
  SIMPLE_WAIT, SIMPLE_RX, SIMPLE_DECODE, SIMPLE_PLAY, SIMPLE_DONE,
  SIMPLE_ERROR, SIMPLE_STOP, SIMPLE_PROCESS
};

typedef struct
{
  int rx_role;
  const char *raw;
  int rx_pkt;
  int rx_rssi;
  int tx_pkt;
} simple_state_t;

#define S_BG       RGB(0x10,0x1a,0x18)
#define S_SURFACE  RGB(0x19,0x2b,0x25)
#define S_TEXT     RGB(0xed,0xf6,0xf0)
#define S_MUTED    RGB(0xa2,0xb8,0xac)
#define S_LINE     RGB(0x2d,0x43,0x38)
#define S_MINT     RGB(0x9c,0xe8,0xc4)
#define S_BLUE     RGB(0x9f,0xc7,0xe8)
#define S_AMBER    RGB(0xf3,0xc9,0x80)
#define S_RED      RGB(0xff,0xa1,0x9c)

static enum simple_phase simple_phase_for(const simple_state_t *s)
{
  const char *r = s->raw != NULL ? s->raw : "";
  if (strstr(r, "FAIL") || strstr(r, "ERROR") || strstr(r, "PEER NOT") ||
      strstr(r, "TIMEOUT")) { return SIMPLE_ERROR; }
  if (strstr(r, "OFF") || strstr(r, "STOPPED")) { return SIMPLE_STOP; }
  if (strstr(r, "PLAY")) { return SIMPLE_PLAY; }
  if (strstr(r, "RX DONE")) { return SIMPLE_DONE; }
  if (strstr(r, "TX DONE") || strcmp(r, "SENT") == 0) { return SIMPLE_SENT; }
  if (strstr(r, "PREP") || strstr(r, "WAIT RADIO")) { return SIMPLE_PREP; }
  if (strstr(r, "STOP REQ")) { return SIMPLE_SEND; }
  if (strstr(r, "REC")) { return SIMPLE_RECORD; }
  if (strstr(r, "DECODE")) { return SIMPLE_DECODE; }
  if (strstr(r, "ENC") || strstr(r, "SENDING") ||
      strstr(r, "TX BUSY") || strncmp(r, "TX ", 3) == 0)
    {
      return strstr(r, "READY") ? SIMPLE_IDLE : SIMPLE_SEND;
    }
  if (strstr(r, "RX WAIT") || strstr(r, "RX READY")) { return SIMPLE_WAIT; }
  if (strncmp(r, "RX ", 3) == 0 || strstr(r, "DUAL RX")) { return SIMPLE_RX; }
  if (r[0] == '\0' || strstr(r, "READY") || strncmp(r, "K0=", 3) == 0 ||
      strcmp(r, "等待中") == 0 || strcmp(r, "PTT IDLE") == 0)
    {
      return s->rx_role ? SIMPLE_WAIT : SIMPLE_IDLE;
    }
  return SIMPLE_PROCESS;
}

static const char *simple_title(enum simple_phase p)
{
  switch (p)
    {
      case SIMPLE_IDLE: return "准备录音";
      case SIMPLE_PREP: return "准备开始";
      case SIMPLE_RECORD: return "正在录音";
      case SIMPLE_SEND: return "正在发送";
      case SIMPLE_SENT: return "发送完成";
      case SIMPLE_WAIT: return "等待语音";
      case SIMPLE_RX: return "正在接收";
      case SIMPLE_DECODE: return "正在解码";
      case SIMPLE_PLAY: return "正在播放";
      case SIMPLE_DONE: return "播放完成";
      case SIMPLE_ERROR: return "操作失败";
      case SIMPLE_STOP: return "已停止";
      default: return "正在处理";
    }
}

static const char *simple_hint(enum simple_phase p)
{
  switch (p)
    {
      case SIMPLE_IDLE: return "按 BOOT 键开始";
      case SIMPLE_PREP: return "两声提示音后说话";
      case SIMPLE_RECORD: return "请说话，结束时有提示音";
      case SIMPLE_SEND: return "语音处理中，请稍候";
      case SIMPLE_SENT: return "本机发送完成";
      case SIMPLE_WAIT: return "收到语音后自动播放";
      case SIMPLE_RX: return "正在接收语音数据";
      case SIMPLE_DECODE: return "语音还原中，请稍候";
      case SIMPLE_PLAY: return "请听扬声器";
      case SIMPLE_DONE: return "本段语音已播放";
      case SIMPLE_ERROR: return "请检查设备状态";
      case SIMPLE_STOP: return "请检查运行状态";
      default: return "请稍候";
    }
}

static void simple_logo(void)
{
  for (int y = 0; y < 20; y += 4)
    {
      memcpy(B->scr, g_simple_logo + y * 20, 20 * 4 * sizeof(uint16_t));
      flush(12, 10 + y, 20, 4);
    }
}

static void simple_header(const simple_state_t *s, int info)
{
  uint16_t accent = s->rx_role ? S_BLUE : S_MINT;
  fill_rect(0, 0, 240, 38, S_BG);
  simple_logo();
  text1(40, 12, "极讯 AI", S_TEXT, S_BG);
  text1(154, 12, s->rx_role ? "接收" : "发送", accent, S_BG);
  /* Non-touch panel: no on-screen control or information hit target. */
  (void)info;
  fill_rect(12, 37, 216, 1, S_LINE);
}

static void simple_symbol(enum simple_phase p, uint16_t color)
{
  /* Static voice glyph: not an audio level meter or live waveform. */
  static const uint8_t heights[] = {4, 10, 18, 12, 26, 16, 8, 14, 4};
  if (p == SIMPLE_SENT || p == SIMPLE_DONE)
    {
      for (int i = 0; i < 8; i++) { fill_rect(107 + i, 87 + i, 3, 3, color); }
      for (int i = 0; i < 17; i++) { fill_rect(114 + i, 94 - i, 3, 3, color); }
    }
  else if (p == SIMPLE_ERROR)
    {
      fill_rect(118, 76, 4, 17, color);
      fill_rect(118, 98, 4, 4, color);
    }
  else
    {
      for (int i = 0; i < 9; i++)
        {
          int h = heights[i];
          fill_rect(94 + i * 6, 89 - h / 2, 3, h, color);
        }
    }
}

static void simple_main(const simple_state_t *s)
{
  enum simple_phase p = simple_phase_for(s);
  uint16_t accent = s->rx_role ? S_BLUE : S_MINT;
  uint16_t state_color = p == SIMPLE_ERROR ? S_RED :
                         p == SIMPLE_RECORD ? S_AMBER : accent;
  const char *title = simple_title(p);
  const char *action;
  int width = str_chars(title) * 32;
  int can_send = !s->rx_role &&
                 (p == SIMPLE_IDLE || p == SIMPLE_SENT || p == SIMPLE_ERROR);

  fill_rect(0, 38, 240, 202, S_BG);
  text1(16, 48, "3 kbps", S_MUTED, S_BG);
  text1(160, 48, "16 kHz", S_MUTED, S_BG);
  simple_symbol(p, state_color);
  text2((240 - width) / 2, 112, title, S_TEXT, S_BG);
  text_c(120, 154, simple_hint(p), S_MUTED, S_BG);

  action = can_send ? (p == SIMPLE_ERROR ? "BOOT 重试录音" : "BOOT 开始录音") :
           p == SIMPLE_RECORD ? "录音中" :
           p == SIMPLE_PLAY ? "播放中" :
           s->rx_role && (p == SIMPLE_WAIT || p == SIMPLE_DONE) ? "BOOT 查看信息" :
           p == SIMPLE_ERROR ? "检查接收状态" :
           p == SIMPLE_STOP ? "接收已停止" : "请稍候";
  fill_rect(16, 188, 208, 38, can_send ? accent : S_SURFACE);
  text_c(120, 199, action, can_send ? S_BG : state_color,
         can_send ? accent : S_SURFACE);
}

static void simple_info_row(int y, const char *label, const char *value)
{
  text1(16, y, label, S_MUTED, S_BG);
  text1(100, y, value, S_TEXT, S_BG);
}

static void simple_info(const simple_state_t *s)
{
  char value[24];
  fill_rect(0, 38, 240, 202, S_BG);
  simple_info_row(50, "设备", s->rx_role ? "接收端" : "发送端");
  simple_info_row(77, "语音", "3 kbps / 16 kHz");
  simple_info_row(104, "状态", simple_title(simple_phase_for(s)));
  snprintf(value, sizeof(value), "%d", s->rx_role ? s->rx_pkt : s->tx_pkt);
  simple_info_row(131, s->rx_role ? "收包" : "发包", value);
  if (s->rx_role && s->rx_pkt > 0)
    {
      snprintf(value, sizeof(value), "%d dBm", s->rx_rssi);
    }
  else { snprintf(value, sizeof(value), "--"); }
  simple_info_row(158, "RSSI", value);
  fill_rect(16, 188, 208, 38, S_SURFACE);
  text_c(120, 199, s->rx_role ? "BOOT 返回语音" : "设备信息", S_TEXT, S_SURFACE);
}

static void simple_render(const simple_state_t *s, int info)
{
  simple_header(s, info);
  if (info) { simple_info(s); }
  else { simple_main(s); }
}
