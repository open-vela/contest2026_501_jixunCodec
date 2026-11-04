/* Native rendering of the ORIGINAL deployed K7 console's comfort layout.
 * Geometry, colors, SVG icons and branding follow ui/index.html and its
 * captured 1280x720 screens. Business state still comes from the same node.
 */
#include "jx_original_assets.h"
LV_FONT_DECLARE(jx_font_16);LV_FONT_DECLARE(jx_font_17);
LV_FONT_DECLARE(jx_font_23);LV_FONT_DECLARE(jx_font_27);
LV_FONT_DECLARE(jx_font_30);LV_FONT_DECLARE(jx_font_35);

static lv_obj_t *canvas,*page_heading,*badge_label,*peer_label,*channel_label;
static lv_obj_t *stat_labels[8],*wave_bars[17],*nav_icons[7],*nav_selected,*value_units[4];
static lv_obj_t *clock_small,*source_label,*timer_label,*toast_label,*dimmer,*connection_dot;
static lv_obj_t *modal;
static bool comfort=true;
static int brightness=100;
static int badge_phase=-1;
static char rendered_role;
static double toast_until;
static const lv_image_dsc_t *muted_icons[]={&jx_icon_overview_muted,&jx_icon_nodes_muted,&jx_icon_channels_muted,&jx_icon_call_muted,&jx_icon_monitor_muted,&jx_icon_events_muted,&jx_icon_settings_muted};
static const lv_image_dsc_t *green_icons[]={&jx_icon_overview_green,&jx_icon_nodes_green,&jx_icon_channels_green,&jx_icon_call_green,&jx_icon_monitor_green,&jx_icon_events_green,&jx_icon_settings_green};
static const char *local_name(void){return role=='A'?"K7主板":role=='B'?"K7从板":"K7节点";}
static const char *peer_name(void){return role=='B'?"K7主板":"K7从板";}
static const char *local_id(void){return role=='B'?"20000002":"10000001";}
static const char *pair_label(void)
{
 static const char *labels[]={"业务尚未就绪","等待对端入网","已入网 · 等待通话","正在接通","双向通话中","正在结束"};
 return labels[phase<0?0:phase];
}
static const lv_font_t *font(int size)
{
 switch(size){case 16:return &jx_font_16;case 17:return comfort?&jx_font_17:&jx_font_16;
 case 23:return &jx_font_23;case 27:return &jx_font_27;case 30:return &jx_font_30;default:return &jx_font_35;}
}
static lv_obj_t *box(lv_obj_t *parent,int x,int y,int w,int h,uint32_t color)
{
 lv_obj_t *o=lv_obj_create(parent);lv_obj_remove_style_all(o);lv_obj_set_pos(o,x,y);lv_obj_set_size(o,w,h);
 lv_obj_set_style_bg_color(o,lv_color_hex(color),0);lv_obj_set_style_bg_opa(o,LV_OPA_COVER,0);
 lv_obj_set_style_radius(o,16,0);lv_obj_remove_flag(o,LV_OBJ_FLAG_SCROLLABLE);return o;
}
static lv_obj_t *panel(lv_obj_t *parent,int x,int y,int w,int h)
{
 lv_obj_t *o=box(parent,x,y,w,h,CARD);lv_obj_set_style_border_color(o,lv_color_hex(0x29363e),0);lv_obj_set_style_border_width(o,1,0);return o;
}
static lv_obj_t *text(lv_obj_t *parent,int x,int y,int w,const char *value,int size,uint32_t color)
{
 lv_obj_t *o=lv_label_create(parent);lv_obj_set_pos(o,x,y);lv_obj_set_width(o,w);
 lv_obj_set_style_text_font(o,font(size),0);lv_obj_set_style_text_color(o,lv_color_hex(color),0);
 lv_label_set_long_mode(o,LV_LABEL_LONG_WRAP);lv_label_set_text(o,value);return o;
}
static lv_obj_t *picture(lv_obj_t *parent,int x,int y,const lv_image_dsc_t *asset)
{lv_obj_t *o=lv_image_create(parent);lv_image_set_src(o,asset);lv_obj_set_pos(o,x,y);return o;}
static void set_text(lv_obj_t *o,const char *value)
{if(o&&strcmp(lv_label_get_text(o),value))lv_label_set_text(o,value);}
static void toast(const char *value)
{set_text(toast_label,value);lv_obj_remove_flag(lv_obj_get_parent(toast_label),LV_OBJ_FLAG_HIDDEN);toast_until=seconds(CLOCK_MONOTONIC)+4;}
static void render_page(void);
static void refresh(void);
static void navigation(lv_event_t *e)
{tab=(int)(intptr_t)lv_event_get_user_data(e);render_page();refresh();}
static void call_click(lv_event_t *e)
{
 bool stop=phase==INCALL||phase==CONNECTING||phase==ENDING;
 if(phase==ENDING||(!stop&&phase!=READY))return;
 char *args[]={"k7_pair",stop?"stop":"start",NULL};last_rc=k7_pair_main(2,args);
 if(!last_rc){pending=stop?2:1;pending_until=seconds(CLOCK_REALTIME)+2;event(stop?"本板请求结束通话":"本板请求开始通话");}
 else {event("操作未成功，请稍后重试");toast("操作未成功，请稍后重试");}
}
static lv_obj_t *button(lv_obj_t *parent,int x,int y,int w,int h,const char *value,const lv_image_dsc_t *asset,lv_event_cb_t cb,void *data)
{
 lv_obj_t *o=box(parent,x,y,w,h,0x2a3941);lv_obj_set_style_radius(o,11,0);
 lv_obj_set_style_border_color(o,lv_color_hex(0x3d505b),0);lv_obj_set_style_border_width(o,1,0);
 lv_obj_add_flag(o,LV_OBJ_FLAG_CLICKABLE);lv_obj_set_style_bg_color(o,lv_color_hex(0x3b505a),LV_STATE_PRESSED);
 lv_obj_set_style_opa(o,LV_OPA_40,LV_STATE_DISABLED);
 if(cb)lv_obj_add_event_cb(o,cb,LV_EVENT_CLICKED,data);
 lv_obj_t *t=text(o,0,0,w,value,comfort?23:17,WHITE);lv_obj_set_style_text_align(t,LV_TEXT_ALIGN_CENTER,0);lv_obj_center(t);
 if(asset){int tw=(int)strlen(value)*8;if(tw>w-60)tw=w-60;picture(o,(w-tw)/2-32,(h-27)/2,asset);}
 return o;
}
static lv_obj_t *badge(lv_obj_t *parent,int x,int y,int w,const char *value,bool green)
{
 lv_obj_t *o=box(parent,x,y,w,38,green?0x244136:0x26394e);lv_obj_set_style_radius(o,7,0);
 lv_obj_t *t=text(o,0,0,w,value,16,green?GREEN:0x8fb9e9);lv_obj_set_style_text_align(t,LV_TEXT_ALIGN_CENTER,0);lv_obj_center(t);return t;
}
static void make_call_button(lv_obj_t *parent,int x,int y,int w,int h)
{
 action=button(parent,x,y,w,h,"开始通话",NULL,call_click,NULL);action_label=lv_obj_get_child(action,0);
 picture(action,w/2-90,(h-27)/2,&jx_icon_call_ink);
}
static lv_obj_t *row(lv_obj_t *parent,int y,int width,const char *name,const lv_image_dsc_t *asset)
{
 if(asset)picture(parent,22,y+12,asset);
 text(parent,asset?60:23,y+12,width/2-40,name,17,MUTED);
 lv_obj_t *v=text(parent,width/2,y+12,width/2-23,"—",17,WHITE);lv_obj_set_style_text_align(v,LV_TEXT_ALIGN_RIGHT,0);
 lv_obj_t *line=box(parent,22,y+51,width-44,1,0x29383e);lv_obj_set_style_radius(line,0,0);return v;
}
static void make_stats(lv_obj_t *parent,int width,int y,bool short_rows)
{
 const char *names[]={"LoRa 入网","时间对齐","接收 / 发送","序号缺口 / 丢包率","RSSI / 当前SNR","软件相位 RMS"};
 const lv_image_dsc_t *icons[]={&jx_icon_link_muted,&jx_icon_clock_muted,&jx_icon_arrow_muted,&jx_icon_monitor_muted,&jx_icon_signal_muted,&jx_icon_clock_muted};
 for(int i=0;i<6;i++)stat_labels[i]=row(parent,y+i*(short_rows?46:52),width,names[i],icons[i]);
}
static void volume_click(lv_event_t *e)
{
 int v=(int)(intptr_t)lv_event_get_user_data(e);char b[8];snprintf(b,sizeof(b),"%d",v);
 char *args[]={"audioctl","volume",b,NULL};last_rc=audioctl_main(3,args);
 if(!last_rc){volume=v;event("已设置扬声器音量");toast("扬声器音量已设置");}else toast("音量设置失败");
}
static void export_click(lv_event_t *e)
{
 FILE *f=fopen("/tmp/k7_ui_diagnostics.json.tmp","w");if(!f){toast("诊断记录保存失败");return;}
 fprintf(f,"{\"role\":\"%c\",\"phase\":\"%s\",\"tx_packets\":%llu,\"rx_packets\":%llu,\"seq_gaps\":%llu,\"signal_valid\":%s,\"rssi_dbm\":%.0f,\"snr_db\":%.2f,\"events\":[",role,phases[phase],(unsigned long long)tx,(unsigned long long)rx,(unsigned long long)gaps,signal_valid?"true":"false",signal_valid?rssi:0,signal_valid?snr:0);
 unsigned n=event_count<16?event_count:16;for(unsigned i=0;i<n;i++)fprintf(f,"%s\"%s\"",i?",":"",events[(event_count-n+i)%16]);fputs("]}\n",f);
 int rc=fclose(f);if(!rc)rc=rename("/tmp/k7_ui_diagnostics.json.tmp","/tmp/k7_ui_diagnostics.json");toast(rc?"诊断记录保存失败":"诊断记录已保存");
}
static void comfort_click(lv_event_t *e)
{comfort=!comfort;render_page();refresh();toast(comfort?"已开启大字触控":"已切换研发密度");}
static void fullscreen_click(lv_event_t *e){toast("当前已为全屏触控");}
static void brightness_change(lv_event_t *e)
{brightness=lv_slider_get_value(lv_event_get_target(e));lv_obj_set_style_bg_opa(dimmer,(100-brightness)*255/100,0);}
static void close_modal(lv_event_t *e){if(modal){lv_obj_delete(modal);modal=NULL;}}
static void node_detail(lv_event_t *e)
{
 if(modal)lv_obj_delete(modal);modal=panel(lv_layer_top(),220,130,840,460);
 bool local=(intptr_t)lv_event_get_user_data(e)==0;
 text(modal,28,22,700,local?local_name():peer_name(),27,WHITE);
 button(modal,710,20,95,60,"关闭",NULL,close_modal,NULL);
 char b[800];snprintf(b,sizeof(b),"节点 ID：%s\n角色：%s\n业务信道：CH-01\n入网状态：%s\n时间来源：LoRa 接入包 / 语音帧\nREQ   →   RESP   →   ACK   →   READY",local?local_id():role=='A'?"20000002":"10000001",local?(role=='A'?"主机":"从机"):(role=='A'?"从机":"主机"),local?"本板运行中":phase>=READY?"已接入":"等待对端");
 lv_obj_t *t=text(modal,28,110,780,b,23,WHITE);lv_obj_set_style_text_line_space(t,14,0);
}
static void render_page(void)
{
 rendered_role=role;
 if(content)lv_obj_delete(content);
 content=box(body,110,94,1170,588,BG);lv_obj_set_style_radius(content,0,0);
 lv_obj_add_flag(content,LV_OBJ_FLAG_SCROLLABLE);lv_obj_set_scroll_dir(content,LV_DIR_VER);lv_obj_set_scrollbar_mode(content,LV_SCROLLBAR_MODE_AUTO);
 lv_obj_set_style_width(content,5,LV_PART_SCROLLBAR);lv_obj_set_style_bg_color(content,lv_color_hex(0x455962),LV_PART_SCROLLBAR);
 canvas=box(content,25,20,1120,548,BG);lv_obj_set_style_radius(canvas,0,0);
 details=NULL;phase_label=NULL;tip_label=NULL;action=NULL;action_label=NULL;badge_label=NULL;peer_label=NULL;channel_label=NULL;timer_label=NULL;styled_phase=-1;badge_phase=-1;
 memset(values,0,sizeof(values));memset(stat_labels,0,sizeof(stat_labels));memset(wave_bars,0,sizeof(wave_bars));
 for(int i=0;i<7;i++){
  lv_obj_set_style_bg_color(nav[i],lv_color_hex(i==tab?0x21362e:0x0d1418),0);
  lv_image_set_src(nav_icons[i],i==tab?green_icons[i]:muted_icons[i]);
  lv_obj_t *caption=lv_obj_get_child(nav[i],1);lv_obj_set_style_text_color(caption,lv_color_hex(i==tab?GREEN:MUTED),0);
 }
 lv_obj_set_y(nav_selected,tab==6?642:100+79*tab);set_text(page_heading,tab==6?"控制台设置":tabs[tab]);
 if(tab==0){
  lv_obj_set_height(canvas,1160);
  const char *names[]={"已入网节点","当前语音模型","本板接收","接收序号缺口"};
  const lv_image_dsc_t *icons[]={&jx_icon_nodes_muted,&jx_icon_chip_muted,&jx_icon_arrow_muted,&jx_icon_monitor_muted};
  for(int i=0;i<4;i++){
   lv_obj_t *c=panel(canvas,i*283,0,270,123);text(c,17,17,216,names[i],17,MUTED);picture(c,229,17,icons[i]);
   values[i]=text(c,17,46,165,"—",35,WHITE);lv_obj_set_width(values[i],LV_SIZE_CONTENT);
   value_units[i]=text(c,i==1?209:i==0?45:64,70,70,i==0?"个":i==1?"Light A":"包",16,i==1?GREEN:MUTED);
  }
  lv_obj_t *c=panel(canvas,0,141,1120,396);text(c,23,24,750,"主从实时语音",23,WHITE);
  badge_label=badge(c,935,16,162,pair_label(),phase==INCALL);
  text(c,23,62,760,role=='B'?"CH-01 · K7从板":"CH-01 · K7主板",16,GREEN);
  text(c,23,81,800,"K7主板 ↔ K7从板",27,WHITE);
  channel_label=text(c,23,120,800,"433.175 MHz / SF8 / 500 kHz",17,MUTED);
  badge(c,951,81,146,"3K · 96 ms 分包",true);
  lv_obj_t *left=panel(c,23,154,386,80);lv_obj_set_style_bg_color(left,lv_color_hex(0x101c1f),0);lv_obj_set_style_border_color(left,lv_color_hex(0x344a43),0);lv_obj_set_style_radius(left,12,0);
  lv_obj_t *ni=box(left,13,16,44,46,0x244136);lv_obj_set_style_radius(ni,8,0);picture(ni,8,9,role=='B'?&jx_icon_radio_green:&jx_icon_antenna_green);
  text(left,65,13,290,local_name(),23,WHITE);char b[80];snprintf(b,sizeof(b),"本板 · %s",local_id());text(left,65,44,290,b,16,MUTED);
  lv_obj_t *right=panel(c,710,154,386,80);lv_obj_set_style_bg_color(right,lv_color_hex(0x101c1f),0);lv_obj_set_style_border_color(right,lv_color_hex(0x344a43),0);lv_obj_set_style_radius(right,12,0);
  ni=box(right,13,16,44,46,0x244136);lv_obj_set_style_radius(ni,8,0);picture(ni,8,9,role=='B'?&jx_icon_antenna_green:&jx_icon_radio_green);
  text(right,65,13,290,peer_name(),23,WHITE);peer_label=text(right,65,44,290,"等待入网",16,MUTED);
  for(int i=0;i<17;i++){wave_bars[i]=box(c,510+i*6,178,3,4,0x498e76);lv_obj_set_style_radius(wave_bars[i],1,0);}
  text(c,512,201,180,"LoRa 离线链路",16,GREEN);
  make_call_button(c,23,246,530,84);
  button(c,567,246,530,84,"通话详情",&jx_icon_monitor_text,navigation,(void*)3);
  text(c,23,358,1075,"点一次开始即可持续双向通话，无需按住；任一板结束，两边恢复收听。",17,MUTED);
  c=panel(canvas,0,555,1120,144);text(c,23,18,1000,"离线运行",23,WHITE);
  text(c,23,64,1070,"入网、时间对齐、呼叫和语音均通过 LoRa。屏幕可直接控制本板。",17,MUTED);
  c=panel(canvas,0,717,1120,409);text(c,23,18,1000,"本轮链路状态",23,WHITE);badge(c,930,16,167,"实时遥测",false);make_stats(c,1120,65,false);
  text(c,23,382,1070,"相位值为软件接收观测；主板为时间基准。空闲时语音统计显示为“—”。",16,MUTED);
 }else if(tab==1){
  lv_obj_set_height(canvas,830);text(canvas,0,0,850,"LoRa 离线入网",27,WHITE);text(canvas,0,45,1010,"REQ → RESP → ACK → READY · 接入包对齐单调时钟",17,MUTED);
  badge_label=badge(canvas,945,0,175,pair_label(),false);
  lv_obj_t *c=panel(canvas,0,94,1120,271);const char *heads[]={"节点 / ID","角色","状态","信道","详情"};int xs[]={24,430,570,790,960};
  for(int i=0;i<5;i++)text(c,xs[i],16,210,heads[i],17,MUTED);
  for(int i=0;i<2;i++){
   int y=76+i*92;text(c,24,y,360,i?peer_name():local_name(),23,WHITE);text(c,24,y+35,360,i?(role=='A'?"20000002":"10000001"):local_id(),16,MUTED);
   text(c,430,y+10,120,i?(role=='A'?"从机":"主机"):(role=='A'?"主机":"从机"),17,WHITE);
   lv_obj_t *t=badge(c,555,y,170,i?(phase>=READY?"已接入":"等待对端"):"本板",i==0||phase>=READY);if(i)peer_label=t;
   text(c,790,y+10,160,"CH-01",17,WHITE);button(c,960,y-3,135,66,"查看",NULL,node_detail,(void*)(intptr_t)i);
  }
  c=panel(canvas,0,383,1120,400);text(c,23,18,1000,"时间与时隙",23,WHITE);
  details=text(c,23,72,1070,"",17,WHITE);lv_obj_set_style_text_line_space(details,18,0);
 }else if(tab==2){
  lv_obj_set_height(canvas,590);text(canvas,0,0,950,"主从实时语音信道",27,WHITE);text(canvas,0,45,1050,"固定单主单从配置 · 双方经 LoRa 协商进入语音模式",17,MUTED);badge(canvas,995,0,125,"CH-01",false);
  lv_obj_t *c=panel(canvas,0,94,1120,470);text(c,23,20,1080,"433.175 MHz / 500 kHz / CR 4/5",23,WHITE);
  details=text(c,23,80,1070,"",17,WHITE);lv_obj_set_style_text_line_space(details,24,0);make_call_button(c,23,355,1074,84);
 }else if(tab==3){
  lv_obj_t *c=panel(canvas,0,0,614,548);char b[80];snprintf(b,sizeof(b),"%s · 实时通话",local_name());text(c,20,17,395,b,23,WHITE);badge_label=badge(c,407,16,187,pair_label(),phase==INCALL);
  lv_obj_t *orb=box(c,269,74,76,76,0x244136);lv_obj_set_style_radius(orb,38,0);picture(orb,25,24,&jx_icon_call_green);
  snprintf(b,sizeof(b),"%s ↔ %s",local_name(),peer_name());lv_obj_t *t=text(c,20,166,574,b,27,WHITE);lv_obj_set_style_text_align(t,LV_TEXT_ALIGN_CENTER,0);
  t=text(c,20,212,574,"板载麦克风 / 板载扬声器 · 连续双向语音",16,MUTED);lv_obj_set_style_text_align(t,LV_TEXT_ALIGN_CENTER,0);
  timer_label=text(c,20,253,574,"00:00",35,WHITE);lv_obj_set_style_text_align(timer_label,LV_TEXT_ALIGN_CENTER,0);
  badge(c,126,319,161,"Light A · 3K",true);badge(c,305,319,191,phase>=READY?"LoRa 时间已对齐":"等待 LoRa 对齐",false);
  make_call_button(c,23,396,568,84);text(c,23,498,568,"点一次开始，接通后直接说话；点一次结束。",16,MUTED);
  c=panel(canvas,632,0,488,365);text(c,20,17,445,"本轮收发监控",23,WHITE);make_stats(c,488,64,true);
  c=panel(canvas,632,383,488,165);text(c,20,17,445,"通话说明",23,WHITE);text(c,20,65,445,"可在任意一块板发起或结束通话。\n对端操作后，本屏状态自动更新。",17,MUTED);
 }else if(tab==4){
  lv_obj_set_height(canvas,1260);text(canvas,0,0,960,"射频与处理链状态",27,WHITE);text(canvas,0,45,1040,"只显示实际接入的观测；未采集的指标显示“—”",17,MUTED);badge_label=badge(canvas,945,0,175,pair_label(),false);
  const char *names[]={"频率","带宽","扩频因子","编码率","TX 配置","分包周期"};const char *v[]={"433.175 MHz","500 kHz","SF8","4/5","22 dBm","96 ms"};
  for(int i=0;i<6;i++){lv_obj_t *c=panel(canvas,(i%3)*379,94+(i/3)*128,362,110);text(c,18,12,320,names[i],17,MUTED);lv_obj_t *t=text(c,18,45,320,v[i],30,WHITE);if(i==2)channel_label=t;}
  lv_obj_t *c=panel(canvas,0,366,1120,433);text(c,23,18,1070,"空口与同步",23,WHITE);make_stats(c,1120,65,false);
  stat_labels[6]=row(c,377,1120,"状态距今",&jx_icon_clock_muted);
  c=panel(canvas,0,817,1120,400);text(c,23,18,1070,"语音与系统",23,WHITE);
  details=text(c,23,72,1070,"语音模型：SQ Codec Light A · 3K\n采集 / 播放：16 kHz · 双通道 I2S\n回声消除 / 降噪：未提供遥测\n核心温度：未提供遥测\n嘴到耳实际延迟：未测量\nSNR 是射频信噪比；语音质量需要匹配原音与解码音后计算。",17,WHITE);lv_obj_set_style_text_line_space(details,18,0);
 }else if(tab==5){
  lv_obj_set_height(canvas,770);text(canvas,0,0,870,"每一次接入与切换，都有迹可循",27,WHITE);text(canvas,0,45,880,"本板状态变化及屏幕操作；导出包含当前业务遥测",17,MUTED);
  button(canvas,920,0,200,84,"导出记录",&jx_icon_download_text,export_click,NULL);
  lv_obj_t *c=panel(canvas,0,106,1120,644);details=text(c,24,18,1070,"",17,WHITE);lv_obj_set_style_text_line_space(details,35,0);
 }else{
  lv_obj_set_height(canvas,960);char b[120];snprintf(b,sizeof(b),"%s · 控制台设置",local_name());text(canvas,0,0,890,b,27,WHITE);text(canvas,0,45,1000,"本地资源 · 无需外网",17,MUTED);badge(canvas,900,0,220,"LoRa 离线业务",false);
  lv_obj_t *c=panel(canvas,0,94,1120,834);
  text(c,23,20,680,"通话控制",23,WHITE);text(c,23,64,680,"屏幕控制本板；呼叫、挂断和时间对齐通过 LoRa 完成。",17,MUTED);make_call_button(c,788,25,307,84);
  text(c,23,163,680,"大字触控",23,WHITE);text(c,23,205,680,"调整文字和触点大小。",17,MUTED);button(c,788,168,307,84,comfort?"切回研发密度":"开启大字触控",NULL,comfort_click,NULL);
  text(c,23,306,680,"界面明暗",23,WHITE);text(c,23,348,680,"调整界面亮度。",17,MUTED);
  lv_obj_t *slider=lv_slider_create(c);lv_obj_set_pos(slider,794,350);lv_obj_set_size(slider,295,22);lv_slider_set_range(slider,70,100);lv_slider_set_value(slider,brightness,LV_ANIM_OFF);lv_obj_set_style_bg_color(slider,lv_color_hex(GREEN),LV_PART_INDICATOR);lv_obj_add_event_cb(slider,brightness_change,LV_EVENT_VALUE_CHANGED,NULL);
  text(c,23,449,680,"全屏触控",23,WHITE);text(c,23,491,680,"1280 × 720 横屏。",17,MUTED);button(c,788,454,307,84,"进入全屏",&jx_icon_expand_text,fullscreen_click,NULL);
  text(c,23,592,680,"扬声器音量",23,WHITE);details=text(c,23,634,680,"",17,MUTED);
  button(c,23,706,320,84,"40%",NULL,volume_click,(void*)40);button(c,395,706,320,84,"60%",NULL,volume_click,(void*)60);button(c,767,706,328,84,"80%",NULL,volume_click,(void*)80);
 }
}
static void refresh(void)
{
 char b[2304];read_telemetry();
 if(rendered_role!=role&&role!='?')render_page();
 snprintf(b,sizeof(b),"%s · LoRa 实时业务",local_name());set_text(source_label,b);
 unsigned s=(unsigned)seconds(CLOCK_MONOTONIC);snprintf(b,sizeof(b),"%02u:%02u:%02u",s/3600,s/60%60,s%60);set_text(uptime_label,b);
 snprintf(b,sizeof(b),"%s · %s",local_name(),pair_label());set_text(footer,b);
 if(badge_label)set_text(badge_label,pair_label());
 if(badge_phase!=phase){
  if(badge_label){lv_obj_set_style_bg_color(lv_obj_get_parent(badge_label),lv_color_hex(phase==INCALL?0x244136:0x26394e),0);lv_obj_set_style_text_color(badge_label,lv_color_hex(phase==INCALL?GREEN:0x8fb9e9),0);}
  if(connection_dot)lv_obj_set_style_bg_color(connection_dot,lv_color_hex(phase>=READY?GREEN:0xf09999),0);
  badge_phase=phase;
 }
 if(action){
  bool stop=phase==CONNECTING||phase==INCALL||phase==ENDING;
  set_text(action_label,phase==ENDING?"正在结束…":phase==CONNECTING?"取消接通":stop?"结束通话":"开始通话");
  if(styled_phase!=phase){
   if(phase==READY||(stop&&phase!=ENDING))lv_obj_remove_state(action,LV_STATE_DISABLED);else lv_obj_add_state(action,LV_STATE_DISABLED);
   lv_obj_set_style_bg_color(action,lv_color_hex(stop?0x442e34:GREEN),0);lv_obj_set_style_border_color(action,lv_color_hex(stop?0x5a3941:GREEN),0);
   lv_obj_set_style_text_color(action_label,lv_color_hex(stop?0xffb5b5:0x14291f),0);styled_phase=phase;
  }
 }
 if(tab==0){
  snprintf(b,sizeof(b),"%d",phase>=READY?2:phase==UNAVAILABLE?0:1);set_text(values[0],b);set_text(values[1],"3K");
  if(phase==INCALL||phase==ENDING){snprintf(b,sizeof(b),"%llu",(unsigned long long)rx);set_text(values[2],b);snprintf(b,sizeof(b),"%llu",(unsigned long long)gaps);set_text(values[3],b);}else {set_text(values[2],"—");set_text(values[3],"—");}
  for(int i=0;i<4;i++)if(i!=1){lv_obj_update_layout(values[i]);lv_obj_set_x(value_units[i],17+lv_obj_get_width(values[i])+6);}
  set_text(peer_label,phase>=READY?"已入网":"等待入网");snprintf(b,sizeof(b),"433.175 MHz / SF%d / 500 kHz",!strcmp(program,"k7_voice")?7:8);set_text(channel_label,b);
  static const int wave_height[17]={8,17,31,29,14,8,19,32,27,12,9,21,32,25,10,10,23};
  for(int i=0;i<17;i++){int height=phase==INCALL?wave_height[i]:4;lv_obj_set_height(wave_bars[i],height);lv_obj_set_y(wave_bars[i],178+(26-height)/2);}
 }else if(tab==1){
  char rt[48]="— ms",pr[48]="— ms";
  if(role=='B'&&best_rtt>0)snprintf(rt,sizeof(rt),"%.3f ms",best_rtt/1000);
  if(role=='B'&&signal_valid)snprintf(pr,sizeof(pr),"%.3f ms",phase_rms/1000);
  snprintf(b,sizeof(b),"时间来源       LoRa 接入包 / 语音帧\n调度周期 / 从板时隙       96 / 48 ms\n本板同步样本       %llu\n接入往返时间       %s\n语音软件相位 RMS       %s\n主板提供时间基准，从板通过 LoRa 跟随；软件观测不代表物理同步精度。",(unsigned long long)sync_samples,rt,pr);set_text(details,b);
  if(peer_label)set_text(peer_label,phase>=READY?"已接入":"等待对端");
 }else if(tab==2){
  snprintf(b,sizeof(b),"入网和空闲        SF8 · LoRa 接入包\n双向通话        SF7 · 96 ms 周期，48 ms 交错时隙\n模型 / 帧长        Light A 3K / 52 Bytes · 16 token\n控制链路        双方均通过 LoRa 收发控制包\n当前状态        %s",pair_label());set_text(details,b);
 }else if(tab==3){unsigned d=call_since?(unsigned)(seconds(CLOCK_MONOTONIC)-call_since):0;snprintf(b,sizeof(b),"%02u:%02u",d/60,d%60);set_text(timer_label,b);
 }else if(tab==4){set_text(channel_label,!strcmp(program,"k7_voice")?"SF7":"SF8");
 }else if(tab==5){b[0]=0;unsigned n=event_count<8?event_count:8;for(unsigned i=0;i<n;i++){size_t k=strlen(b);snprintf(b+k,sizeof(b)-k,"%s%s",i?"\n":"",events[(event_count-n+i)%16]);}set_text(details,b);
 }else if(tab==6){snprintf(b,sizeof(b),"扬声器设置 %d%% · 麦克风增益设置 %d%%",volume,gain);set_text(details,b);}
 if(stat_labels[0]){
  set_text(stat_labels[0],phase>=READY?"已完成":"等待对端");set_text(stat_labels[1],phase>=READY?(role=='A'?"主板时间基准":"已跟随主板"):"等待接入包");
  if(phase==INCALL||phase==ENDING)snprintf(b,sizeof(b),"%llu / %llu 包",(unsigned long long)rx,(unsigned long long)tx);else strcpy(b,"— / — 包");set_text(stat_labels[2],b);
  if(phase==INCALL||phase==ENDING)snprintf(b,sizeof(b),"%llu / %.2f %%",(unsigned long long)gaps,rx+gaps?100.0*gaps/(rx+gaps):0);else strcpy(b,"— / — %");set_text(stat_labels[3],b);
  if(signal_valid)snprintf(b,sizeof(b),"%.0f dBm / %.2f dB",rssi,snr);else strcpy(b,"— dBm / — dB");set_text(stat_labels[4],b);
  if(role=='B'&&signal_valid)snprintf(b,sizeof(b),"%.3f ms",phase_rms/1000);else strcpy(b,"— ms");set_text(stat_labels[5],b);
  if(signal_valid)snprintf(b,sizeof(b),"%.2f s",rx_age/1000);else strcpy(b,"— s");set_text(stat_labels[6],b);
 }
 if(toast_label&&seconds(CLOCK_MONOTONIC)>toast_until)lv_obj_add_flag(lv_obj_get_parent(toast_label),LV_OBJ_FLAG_HIDDEN);
 lv_area_t rect={0};if(action)lv_obj_get_coords(action,&rect);
 FILE *f=fopen("/tmp/k7_ui.json.tmp","w");if(f){fprintf(f,"{\"role\":\"%c\",\"phase\":\"%s\",\"design\":\"original_console\",\"tab\":%d,\"width\":1280,\"height\":720,\"touches\":%u,\"last_touch_x\":%d,\"last_touch_y\":%d,\"flushes\":%u,\"last_rc\":%d,\"signal_valid\":%s,\"button\":[%ld,%ld,%ld,%ld]}\n",role,phases[phase],tab,touches,last_touch_x,last_touch_y,flushes,last_rc,signal_valid?"true":"false",(long)rect.x1,(long)rect.y1,(long)(rect.x2-rect.x1+1),(long)(rect.y2-rect.y1+1));if(!fclose(f))rename("/tmp/k7_ui.json.tmp","/tmp/k7_ui.json");}
}
static void build_shell(void)
{
 body=lv_screen_active();lv_obj_set_style_bg_color(body,lv_color_hex(BG),0);lv_obj_remove_flag(body,LV_OBJ_FLAG_SCROLLABLE);
 lv_obj_t *rail=box(body,0,0,110,720,0x0d1418);lv_obj_set_style_radius(rail,0,0);picture(rail,26,12,&jx_original_brand);
 for(int i=0;i<7;i++){
  int y=i==6?632:76+i*79;
  nav[i]=box(rail,8,y,93,76,0x0d1418);lv_obj_set_style_radius(nav[i],13,0);lv_obj_add_flag(nav[i],LV_OBJ_FLAG_CLICKABLE);lv_obj_add_event_cb(nav[i],navigation,LV_EVENT_CLICKED,(void*)(intptr_t)i);
  nav_icons[i]=picture(nav[i],33,11,muted_icons[i]);lv_obj_t *t=text(nav[i],0,43,93,i==6?"设置":tabs[i],17,MUTED);lv_obj_set_style_text_align(t,LV_TEXT_ALIGN_CENTER,0);
 }
 nav_selected=box(rail,0,100,3,24,GREEN);lv_obj_set_style_radius(nav_selected,1,0);
 lv_obj_t *top=box(body,110,0,1170,94,BG);lv_obj_set_style_radius(top,0,0);
 text(top,25,16,690,"极讯AI Codec  /  LoRa语音通信控制台",16,0xa7bdb5);
 page_heading=text(top,25,45,690,"通信总览",30,WHITE);
 lv_obj_t *source=box(top,742,28,215,38,0x3b3328);lv_obj_set_style_radius(source,7,0);source_label=text(source,0,0,215,"K7节点 · LoRa 实时业务",16,0xedbc78);lv_obj_set_style_text_align(source_label,LV_TEXT_ALIGN_CENTER,0);lv_obj_center(source_label);
 lv_obj_t *expand=box(top,983,20,64,56,BG);picture(expand,18,14,&jx_icon_expand_muted);lv_obj_add_flag(expand,LV_OBJ_FLAG_CLICKABLE);lv_obj_add_event_cb(expand,fullscreen_click,LV_EVENT_CLICKED,NULL);
 lv_obj_t *line=box(top,1031,22,1,48,0x2c3941);lv_obj_set_style_radius(line,0,0);
 uptime_label=text(top,1050,21,112,"00:00:00",23,WHITE);lv_obj_set_style_text_align(uptime_label,LV_TEXT_ALIGN_RIGHT,0);
 clock_small=text(top,1050,57,112,"运行时间",16,MUTED);lv_obj_set_style_text_align(clock_small,LV_TEXT_ALIGN_RIGHT,0);
 line=box(body,110,93,1170,1,0x2c3941);lv_obj_set_style_radius(line,0,0);
 lv_obj_t *bottom=box(body,110,682,1170,38,BG);lv_obj_set_style_radius(bottom,0,0);
 line=box(bottom,0,0,1170,1,0x2c3941);lv_obj_set_style_radius(line,0,0);connection_dot=box(bottom,28,16,7,7,GREEN);lv_obj_set_style_radius(connection_dot,4,0);
 footer=text(bottom,42,8,450,"等待本板状态",16,MUTED);text(bottom,475,8,510,"LoRa 离线业务 · 自动刷新 1 s · 5″ / 1280 × 720",16,MUTED);
 lv_obj_t *export=box(bottom,1050,1,120,36,BG);text(export,0,7,100,"导出诊断",16,MUTED);picture(export,90,5,&jx_icon_download_muted);lv_obj_add_flag(export,LV_OBJ_FLAG_CLICKABLE);lv_obj_add_event_cb(export,export_click,LV_EVENT_CLICKED,NULL);
 dimmer=box(lv_layer_top(),0,0,1280,720,0);lv_obj_set_style_radius(dimmer,0,0);lv_obj_set_style_bg_opa(dimmer,0,0);lv_obj_remove_flag(dimmer,LV_OBJ_FLAG_CLICKABLE);
 lv_obj_t *toast_box=panel(lv_layer_top(),350,580,580,75);toast_label=text(toast_box,12,17,556,"",23,WHITE);lv_obj_set_style_text_align(toast_label,LV_TEXT_ALIGN_CENTER,0);lv_obj_add_flag(toast_box,LV_OBJ_FLAG_HIDDEN);
 read_telemetry();render_page();refresh();
}
