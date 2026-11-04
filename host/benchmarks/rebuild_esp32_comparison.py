"""Summarize ESP32 measurements. Host processing only scores exported waveforms."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import csv,hashlib,json,re,statistics,sys,wave
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset',type=Path,required=True)
P=parser.parse_args().dataset.resolve()
if not (P/'evidence').is_dir() or not (P/'baseline_amr/results').is_dir():
 raise SystemExit('dataset must contain the original ESP32 evidence and AMR outputs')
sys.path.insert(0,str(P/'python_deps'))
import numpy as np
import scipy
from scipy.signal import resample_poly,correlate,correlation_lags
from pesq import pesq
from pystoi import stoi

H=P/'evidence/COM18_history';A=P/'baseline_amr/results'
Q=P/'quality';Q.mkdir(exist_ok=True)
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(values):return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
def csvwrite(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def s16(a):return np.clip(np.rint(a),-32768,32767).astype('<i2')
def down(a):return s16(resample_poly(np.asarray(a,dtype=float),1,2))
def savewav(path,data,sr=8000):
 with wave.open(str(path),'wb') as w:w.setparams((1,2,sr,0,'NONE','not compressed'));w.writeframes(np.asarray(data,dtype='<i2').tobytes())
def score(ref,deg,sr=8000):
 x=np.asarray(ref,dtype=float)/32768.;y=np.asarray(deg,dtype=float)/32768.
 assert len(x)==len(y) and len(x)>=sr//4
 x=x-x.mean();y=y-y.mean();energy=float(x@x);eps=1e-20
 error=float((x-y)@(x-y));target=(y@x/max(energy,eps))*x
 residual=y-target;frame=round(.03*sr);hop=round(.01*sr);seg=[]
 for i in range(0,len(x)-frame+1,hop):
  a=x[i:i+frame];b=y[i:i+frame];e=float(a@a)
  if e/frame>=1e-6:seg.append(float(np.clip(10*np.log10(e/max(float((a-b)@(a-b)),eps)),-10,35)))
 return {'snr_db':float(10*np.log10(energy/max(error,eps))),'segmental_snr_db':statistics.mean(seg),'si_sdr_db':float(10*np.log10(max(float(target@target),eps)/max(float(residual@residual),eps))),'pesq_nb':float(pesq(sr,x,y,'nb')),'stoi':float(stoi(x,y,sr,extended=False)),'samples':len(x),'sample_rate':sr}
def mdtable(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v).replace('|','/') for v in r)+' |' for r in rows])
def fmt(v,d=3):return '未测/未完成' if v is None else f'{v:.{d}f}'

cases=[('fixed2s_01','固定2秒首轮','固定/40ms/手动接收'),('fixed2s_gap80','固定2秒间隔对照','固定/80ms/手动接收'),('fixed2s_receive_then_decode','接收后解码诊断','诊断模式/40ms'),('fixed2s_gap40_repeat','固定2秒原配置复测','固定/40ms/手动接收'),('real_mic2s_01','真人2秒','真人/40ms/手动接收'),('after_other_tx_paused_01','暂停其他设备后第1轮','固定/40ms/自动接收'),('after_other_tx_paused_02','暂停其他设备后第2轮','固定/40ms/自动接收'),('after_reset_fixed2s_01','复位后固定2秒','固定/40ms/手动接收')]
rows=[];quality_rows=[];clean=np.fromfile(H/'fixed2s_gap40_repeat/COM18_out.pcm','<i2')
for name,label,group in cases:
 f=H/name;events=[json.loads(v) for v in (f/'events.jsonl').read_text(encoding='utf8').splitlines()]
 trigger=next(e['t'] for e in events if e['port']=='COM12_CMD' and e['line']=='jixun ui send')
 def find(port,text):return next((e for e in events if e['port']==port and text in e['line'] and e['t']>=trigger),None)
 def when(port,text):
  e=find(port,text);return e['t']-trigger if e else None
 summary=find('COM18','收包结束');m=re.search(r'：(\d+) 包，坏包 (\d+)，丢包 (\d+)',summary['line']) if summary else None
 pb=find('COM18','[SRX_PLAY] mode=');play=dict(re.findall(r'(\w+)=([^\s]+)',pb['line'])) if pb else {}
 complete=find('COM12','LoRa TX done rc=0') is not None
 first=when('COM18','开始播放');capture=when('COM12','recording done')
 row={'test':name,'label':label,'group':group,'TX_complete':complete,'trigger_to_first_PCM_s':first,'capture_end_to_first_PCM_s':first-capture if first is not None and capture is not None else None,'trigger_to_capture_end_s':capture,'trigger_to_last_RF_s':when('COM18',' LAST'),'trigger_to_decode_end_s':when('COM18','收包结束'),'trigger_to_play_end_s':when('COM18','[SRX_PLAY] mode='),'accepted_speech_packets':int(m[1]) if m and complete else None,'invalid_RX_session_count':int(m[2]) if m else None,'missing_speech_sequence_packets':int(m[3]) if m and complete else None,'played_samples':int(play.get('samples',0)),'play_wall_ms':int(play.get('wall_ms',0)),'starvation_count':int(play.get('starvation',0)),'source_PCM_available':(f/'COM12_raw.pcm').exists(),'output_PCM_available':(f/'COM18_out.pcm').exists(),'PCM_equal_to_clean_same_input':None,'max_abs_PCM_delta':None,'different_PCM_samples':None,'quality_8k':None,'notes':''}
 if name=='after_other_tx_paused_01':row['notes']='无效计数107覆盖暂停其他设备之前的空闲监听，不能当成暂停后的错误率。'
 if not complete:row['notes']='发送首包阶段停止进度，95秒超时，USB写超时；接收端零完整播放，部分RX统计不归属本次语音。'
 if name=='fixed2s_receive_then_decode':row['notes']='只用于收包负载诊断，仅播放8170采样；不列入正常播放验收。'
 if row['source_PCM_available'] and row['output_PCM_available']:
  raw=np.fromfile(f/'COM12_raw.pcm','<i2');out=np.fromfile(f/'COM18_out.pcm','<i2')
  row['source_sha256']=sha(f/'COM12_raw.pcm');row['output_sha256']=sha(f/'COM18_out.pcm')
  if name!='real_mic2s_01':
   delta=out.astype(np.int32)-clean.astype(np.int32);row['PCM_equal_to_clean_same_input']=bool(np.array_equal(out,clean));row['max_abs_PCM_delta']=int(np.max(np.abs(delta)));row['different_PCM_samples']=int(np.sum(delta!=0))
  valid=out[4118:4118+len(raw)];assert len(valid)==len(raw)
  ref8=down(raw);out8=down(valid);row['quality_8k']=score(ref8,out8)
  wb_ref=raw.astype(float)/32768.;wb_out=valid.astype(float)/32768.
  row['PESQ_WB_16k_supplement']=float(pesq(16000,wb_ref-wb_ref.mean(),wb_out-wb_out.mean(),'wb'))
  savewav(Q/(name+'_AI_8k.wav'),out8)
  quality_rows.append({'test':name,'implementation':'当前AI/含实际传输结果',**row['quality_8k']})
 row['full_playback']=complete and row['played_samples']==32000 and row['starvation_count']==0
 row['all_checks_passed_for_this_round']=row['full_playback'] and row['missing_speech_sequence_packets']==0 and (row['PCM_equal_to_clean_same_input'] is not False)
 rows.append(row)
csv_rows=[{k:v for k,v in r.items() if k!='quality_8k'} for r in rows]
csvwrite(P/'当前版本逐轮指标.csv',csv_rows)

raw_float=(P/'evidence/COM18_decoder/fixed3s_output_float32.bin').read_bytes()
fnv=1469598103934665603
for b in raw_float:fnv=((fnv^b)*1099511628211)&0xffffffffffffffff
assert f'{fnv:016x}'=='71267fed96703ab1'
local=np.frombuffer(raw_float,dtype='<f4');assert len(local)==52224
v=np.clip(local,-1,1);scaled=v*np.float32(32767)
local16=np.trunc(np.where(v>=0,scaled+np.float32(.5),scaled-np.float32(.5))).astype('<i2')
source3=np.fromfile(P/'inputs/fixed3s_s16le_16k_mono.pcm','<i2')
local8=down(local16[4118:4118+len(source3)])
savewav(Q/'fixed3s_AI_local_8k.wav',local8)
local_quality=score(down(source3),local8)
quality_rows.append({'test':'fixed3s_local_decoder_fixture','implementation':'当前AI/ESP32本地解码',**local_quality})

amr_runs=json.loads((A/'all_runs.json').read_text(encoding='utf8'))
assert len(amr_runs)==9
calref=np.fromfile(A/'fixed2s/reference_8k.pcm','<i2').astype(float)
calout=np.fromfile(A/'fixed2s/run01_out.pcm','<i2').astype(float)
c=correlate(calout-calout.mean(),calref-calref.mean(),method='fft');lags=correlation_lags(len(calout),len(calref));mask=(lags>=0)&(lags<=400)
amr_delay=int(lags[mask][np.argmax(c[mask])]);assert amr_delay==40
amr_stats={};paired=[]
for name in ['fixed2s','human2s','fixed3s']:
 rs=[r for r in amr_runs if r['case']==name];meta=json.loads((A/name/'input_metadata.json').read_text())
 x=np.fromfile(A/name/'reference_8k.pcm','<i2');qs=[]
 for r in rs:
  y=np.fromfile(A/name/('run%02d_out.pcm'%r['run']),'<i2')[amr_delay:amr_delay+len(x)]
  assert len(y)==len(x);m=score(x,y);qs.append(m)
  quality_rows.append({'test':name,'run':r['run'],'implementation':'AMR-NB/ESP32本地编解码',**m})
  if r['run']==1:savewav(Q/(name+'_AMR_aligned_8k.wav'),y)
 amr_stats[name]={'input':meta,'encoding_ms':stats([r['enc_ms'] for r in rs]),'decoding_ms':stats([r['dec_ms'] for r in rs]),'encoding_rtf_processed':stats([r['enc_rtf_processed'] for r in rs]),'decoding_rtf_processed':stats([r['dec_rtf_processed'] for r in rs]),'output_hash_stable':len({r['output_sha256'] for r in rs})==1,'output_sha256':rs[0]['output_sha256'],'quality_8k':qs[0],'container_bitrate_bps':rs[0]['container_bitrate_bps'],'application_preallocated_buffer_bytes_source_calculated':160800,'algorithm_state_peak_heap_bytes':None}
 ai_quality=next(r['quality_8k'] for r in rows if r['test']==('fixed2s_gap40_repeat' if name=='fixed2s' else 'real_mic2s_01')) if name!='fixed3s' else local_quality
 scope='AI：已收齐包且PCM与基线一致的固定输入轮；AMR：本地串口编解码' if name=='fixed2s' else ('AI：冻结真人录音的既有无线输出；AMR：相同录音经8k重采样本地编解码' if name=='human2s' else 'AI：同源3秒语音的固定token本地解码；AMR：同源语音本地编解码')
 row={'case':name,'source_seconds':meta['reference_seconds'],'scope':scope}
 for key in ['pesq_nb','stoi','snr_db','segmental_snr_db','si_sdr_db']:
  row['AI_'+key]=ai_quality[key];row['AMR_'+key]=qs[0][key];row['AI_minus_AMR_'+key]=ai_quality[key]-qs[0][key]
 paired.append(row)
csvwrite(P/'AMR基线逐次指标.csv',amr_runs);csvwrite(P/'音质逐次指标.csv',quality_rows);csvwrite(P/'同输入音质对比.csv',paired)
core=json.loads((P/'evidence/COM18_decoder/decoder_metrics.json').read_text())
speed=core['mean_rtf']/amr_stats['fixed3s']['decoding_rtf_processed']['mean']
manual=[r for r in rows if r['group']=='固定/40ms/手动接收'];auto=[r for r in rows if r['group']=='固定/40ms/自动接收']
groups={}
for label,group in [('固定40ms手动接收',manual),('固定40ms自动接收',auto)]:
 groups[label]={'attempts':len(group),'TX_completed':sum(r['TX_complete'] for r in group),'full_playback':sum(r['full_playback'] for r in group),'same_input_PCM_comparisons':sum(r['PCM_equal_to_clean_same_input'] is not None for r in group),'same_input_PCM_exact':sum(r['PCM_equal_to_clean_same_input'] is True for r in group),'first_PCM_s_completed_only':stats([r['trigger_to_first_PCM_s'] for r in group if r['TX_complete'] and r['trigger_to_first_PCM_s'] is not None]),'note':'现场历史记录，含外设干扰与复位等不同上下文；不是随机化严格A/B或长期故障率。'}
protocol={'device':'ESP32-S3 COM18 / 30:ed:a0:be:3f:68','CPU_MHz':240,'AI_decoder':'NuttX, parallel requested, step16, window128, depth1','AMR':'User-supplied ESP-IDF firmware, MR475, DTX off, single task','all_codec_computation':'ESP32 only','host_tasks':'Frozen input preparation/resampling, USB/log collection, waveform metrics; no host codec simulation used as measured result','quality_sample_rate':8000,'AI_declared_delay_samples_at_16k':4118,'AMR_fixed_alignment_samples_at_8k':amr_delay,'AMR_alignment_calibration':'Once on fixed2s run1, lag constrained to 0..400 at8k; applied unchanged to every clip/run. This 5ms waveform offset is NOT full algorithm or first-audio latency.','quality_preprocessing':'Both waveforms DC removed; no output gain fit or per-clip external lag search. PESQ and STOI retain their standard internal processing.','segmental_SNR':'30ms window,10ms hop,-60dBFS reference threshold,clipped[-10,35]dB','versions':{'numpy':np.__version__,'scipy':scipy.__version__},'radio_comparison':'AMR baseline has no LoRa transport or speaker path. No AMR RF first-audio/loss/continuous-playback claim.','memory_warning':'AI decoder arena peak, AMR source-calculated preallocated buffers, and free heap are different scopes; no memory reduction ratio is computed.'}
report={'protocol':protocol,'current_decoder':core,'current_pipeline_rounds':rows,'current_pipeline_group_summary':groups,'AMR_cases':amr_stats,'AMR_runs':amr_runs,'paired_quality':paired,'AI_local_fixed3s_quality_8k':local_quality,'normalized_decode_RTF_ratio_AI_over_AMR_fixed3s':speed,'quality_test_distinct_clips':3,'AMR_repeats_per_clip':3,'current_pipeline_stability_passed':False,'AMR_RF_transport_tested':False,'AI_encoder_isolated_compute_measured':False,'RAM_peak_comparable_measurement_available':False}
dump(P/'量化指标汇总.json',report);dump(P/'测评口径.json',protocol)

commonrows=[['实际板子','COM18 / ESP32-S3','同一块COM18 / ESP32-S3'],['CPU频率','240MHz','240MHz'],['运行环境','现有NuttX；解码请求双核','随附ESP-IDF基线固件；单任务'],['固件','RX f031b0fe…；TX 9dc1431b…','AMR b71e0006…'],['采样率','16kHz','8kHz'],['标称语音码率','3.0kbps','4.75kbps'],['AMR帧头计入后的文件/流码率','不适用AMR帧格式','5.2kbps（不含LoRa包头）'],['同源3秒语音：实际解码生成时长','3.264秒','3.280秒（对齐20ms帧）'],['纯解码重复次数','3','3（fixed3s）'],['纯解码平均耗时(ms)',fmt(core['work_ms']['mean']),fmt(amr_stats['fixed3s']['decoding_ms']['mean'])],['纯解码耗时范围(ms)',f"{core['work_ms']['min']:.1f}～{core['work_ms']['max']:.1f}",f"{amr_stats['fixed3s']['decoding_ms']['min']:.3f}～{amr_stats['fixed3s']['decoding_ms']['max']:.3f}"],['解码RTF（按实际生成时长归一化）',fmt(core['mean_rtf'],4),fmt(amr_stats['fixed3s']['decoding_rtf_processed']['mean'],4)],['相同固定输入的重复输出','纯解码3次哈希一致','每段3次一致，共9次'],['完整无线首音','当前真人6.893秒；未达到5秒','未测，无LoRa/扬声器链路'],['无线连续运行','未通过：丢包、一次发送卡住、PCM不一致','未测，不能用串口重复测试代替'],['解码arena峰值','1877760字节','随附固件未暴露同口径数值'],['应用预分配缓冲','不与arena直接混比','160800字节，按源码计算，非峰值'],['纯编码耗时','当前流式版本未独立拆分测量',fmt(amr_stats['fixed3s']['encoding_ms']['mean'])+'ms（3秒素材+补尾）']]
qt=[]
for r in paired:qt.append([r['case'],fmt(r['AI_pesq_nb'],4),fmt(r['AMR_pesq_nb'],4),fmt(r['AI_stoi'],4),fmt(r['AMR_stoi'],4),fmt(r['AI_si_sdr_db']),fmt(r['AMR_si_sdr_db'])])
qmeans={codec:{key:statistics.mean(r[codec+'_'+key] for r in paired) for key in ['pesq_nb','stoi']} for codec in ['AI','AMR']}
doc=f'''# 当前AI版本与AMR-NB：ESP32量化对比

日期：2026-11-04。输入来自已冻结的2秒固定语音、2秒真人语音和3秒固定语音；每一对质量对照使用同源输入。两套编解码均实际运行于ESP32-S3，电脑只收集、重采样和评分。本轮未修改两套原工程。

## 结论与适用范围

同源3秒语音、各自完成补尾后，当前AI纯解码平均 **{core['work_ms']['mean']:.1f}ms**，AMR-NB平均 **{amr_stats['fixed3s']['decoding_ms']['mean']:.3f}ms**；按生成音频时长归一化，AI的解码RTF约为AMR的 **{speed:.2f}倍**。这是同一块ESP32上两套现有实现的结果，运行框架、线程配置与码率不同，不是仅更换单个算子的严格实验。

质量统一按8kHz窄带比较。三段非加权平均PESQ-NB：AI **{qmeans['AI']['pesq_nb']:.4f}**、AMR **{qmeans['AMR']['pesq_nb']:.4f}**；STOI：AI **{qmeans['AI']['stoi']:.4f}**、AMR **{qmeans['AMR']['stoi']:.4f}**。只有3段短语音，其中固定2秒与3秒素材相关，不代表大规模独立语料结论。AMR标称码率比AI高58.33%，同时是窄带输出；不能按相同码率解释差异。

当前AI的无线链路仍有已记录的稳定性问题。AMR本次仅跑单板编解码和USB导出，**没有AMR的LoRa首音、空口丢包率或扬声器连续性数据**。不能把AMR的153ms左右解码时间直接与AI的6.89秒无线首音比较。

## 总览

{mdtable(['项目','当前AI版本','AMR-NB MR475基线'],commonrows)}

RTF=板上计算耗时/实际生成的音频时长，低于1表示该计算阶段快于实时。源语音均为3秒，但模型补尾和AMR帧对齐不同，故同时披露实际生成时长。首音计时为触发到I2S提交去掉模型延迟后的PCM，不是声学首字检测。

## 同输入窄带质量

{mdtable(['语音','AI PESQ-NB','AMR PESQ-NB','AI STOI','AMR STOI','AI SI-SDR(dB)','AMR SI-SDR(dB)'],qt)}

fixed2s配对使用AI已经验证PCM一致的成功轮；human2s使用冻结真人录音及当时无线接收输出，用户确认连续、首尾完整；fixed3s使用同源固定token的ESP32本地解码输出。其他失败轮没有删除，完整列于[当前版本指标明细.md](当前版本指标明细.md)及CSV。AMR每段重复3次，输出相同；重复次数不是新增语料数。

## 质量对齐与公平性

AI按模型声明的4118个16kHz采样裁掉重建延迟，再用固定抗混叠重采样降至8kHz；参考信号使用同一重采样规则。AMR在fixed2s第一轮一次校准得到40个8kHz采样偏移，所有其他样本与轮次固定使用40，不逐条搜索最佳延迟，不拟合输出增益。该5ms导出波形偏移不是算法总延迟或首音时延。

两边在计算SNR、分段SNR、SI-SDR前均去直流。PESQ-NB和STOI按标准实现计算，数值越高越好。AI的16kHz宽带能力与原PESQ-WB属于补充信息，不能与AMR的PESQ-NB直接混比。

## 复现与原始数据

- [当前版本指标明细.md](当前版本指标明细.md)：每轮成功、失败和诊断结果。
- [AMR基线指标明细.md](AMR基线指标明细.md)：9次板上耗时、帧数、码流大小与输出哈希。
- [测评口径.md](测评口径.md)：计时、补尾、内存和稳定性定义。
- `当前版本逐轮指标.csv`、`AMR基线逐次指标.csv`、`同输入音质对比.csv`、`音质逐次指标.csv`：可直接用于表格分析。
- `量化指标汇总.json`：全部原始精度的结果；`quality/`含两套对齐后的8kHz WAV。
- `evidence/COM18_decoder/`：AI三次纯解码串口记录及ESP32 float32导出；本机没有运行神经模型代替上板。
- `baseline_amr/`：用户给定基线的源码快照、原始固件、9次导出和恢复资料。使用随附固件运行，没有声称重新编译；固件哈希及来源见`baseline_amr/identity.json`。

AMR固件的可选ASCII结束行未返回，采集脚本改用实际提供的AMRO二进制头与完整PCM校验帧数、长度和耗时，未修改算法。这一采集适配不计作codec故障。内存峰值、CPU任务占用和功耗没有两套同口径实测，不能据现有数字计算改善比例。
'''
(P/'二者量化对比.md').write_text(doc,encoding='utf8')

def packet_count(v):return '—' if v is None else str(v)
table_rows=[[r['label'],r['group'],fmt(r['trigger_to_first_PCM_s']),fmt(r['capture_end_to_first_PCM_s']),packet_count(r['accepted_speech_packets'])+'/'+packet_count(r['invalid_RX_session_count'])+'/'+packet_count(r['missing_speech_sequence_packets']),r['played_samples'],'是' if r['PCM_equal_to_clean_same_input'] is True else ('否' if r['PCM_equal_to_clean_same_input'] is False else '未比/不适用'),'通过该轮' if r['all_checks_passed_for_this_round'] else '失败/诊断'] for r in rows]
current_quality_table=mdtable(['轮次','PESQ-NB','STOI','SNR(dB)','SI-SDR(dB)','补充PESQ-WB'],[[r['label'],fmt(r['quality_8k']['pesq_nb'],4),fmt(r['quality_8k']['stoi'],4),fmt(r['quality_8k']['snr_db']),fmt(r['quality_8k']['si_sdr_db']),fmt(r.get('PESQ_WB_16k_supplement'),4)] for r in rows if r['quality_8k']])
current=f'''# 当前AI版本：量化指标

RX固件f031b0fe…、TX固件9dc1431b…，模型为sqcodec-light-A-c-i8，解码首块88通道，未裁剪。正式历史链路为COM12→COM18；纯解码新增测量也在COM18。

## 纯解码三次新测量

{mdtable(['次数','计算(ms)','墙钟(ms)','RTF','最大一步(ms)','arena峰值(字节)','FNV64'],[[r['run'],r['work_ms'],r['wall_ms'],r['rtf'],r['max_step_ms'],r['arena_peak'],r['hash']] for r in core['runs']])}

544 token、52224个生成采样、3.264秒；mean/median/min/max计算时间为{core['work_ms']['mean']:.1f}/{core['work_ms']['median']:.1f}/{core['work_ms']['min']:.1f}/{core['work_ms']['max']:.1f}ms，平均RTF {core['mean_rtf']:.4f}。三次输出哈希一致。导出质量用的额外运行含串口输出，不纳入耗时平均。

## 全链路逐轮明细

{mdtable(['轮次','条件','首PCM(s)','录完等待(s)','有效/无效/丢','播放采样','PCM一致','该轮结论'],table_rows)}

首音不同输入、手动/自动接收、40/80ms、复位与外部发送条件均有区别，不能直接合并取一个最好值或平均值当成版本指标。仅固定40ms手动接收的3次，发送和完整播放3/3，导出PCM一致1/3；固定40ms自动接收2次中1次发送完成、1次发送卡住。是有限现场观测，不作为长期故障概率。

无效计数107包含暂停其他设备前的空闲监听；发送卡住轮的部分接收包也不能认定属于本轮。诊断模式只播放8170采样，不能算完整播放通过。收齐24包仍出现不同PCM的恢复轮，最大绝对差8973、17130个不同采样，来源未定位到发送、空口或解码。完整失败数据均保留。

真人2秒原轮首音6.893秒、录完等3.241秒，用户确认连续、首尾完整；不能用该确认覆盖其他失败轮。5秒首音目标未达到。算子细分日志因构建关闭profile显示0，不当作算子零耗时。

## 已导出音频的质量明细（含失败轮）

{current_quality_table}

失败轮质量照实列出。首轮含序列丢包，恢复轮虽然未统计到丢包却不满足PCM一致性；不能把质量分数接近当作逐字节一致性通过。WB为16kHz补充评分，AMR不填对应宽带分数。
'''
(P/'当前版本指标明细.md').write_text(current,encoding='utf8')
amr_table=[[r['case'],r['run'],r['frames'],r['amr_bytes'],fmt(r['encoded_audio_seconds']),fmt(r['enc_ms']),fmt(r['dec_ms']),fmt(r['enc_rtf_processed'],4),fmt(r['dec_rtf_processed'],4),r['output_sha256'][:12]] for r in amr_runs]
amrdoc=f'''# AMR-NB MR475基线：ESP32量化指标

源码来源：gl_prj/amr-nb（原项目根目录下）。使用该目录随附的ESP32-S3 amr_bench.bin，SHA-256 b71e0006df0dfc8f6952c153d0aa6d7c83d00c4a4bc33fb120d87b6b57b3fb28。原目录未修改，快照与测试结果在本新目录内。

实际运行于COM18、ESP32-S3、240MHz，MR475、8kHz、160采样/20ms帧、DTX关闭、BFI=0。输入经USB送入板子，编码、解码计时在板上完成；PC没有运行AMR编码器或解码器代替测试。

{mdtable(['输入','次','帧数','码流字节','处理音频(s)','编码(ms)','解码(ms)','编码RTF','解码RTF','输出SHA前12位'],amr_table)}

3段各3次输出一致。补尾保证质量对齐后保留完整源语音：2秒源处理2.320秒，3秒源处理3.280秒；表中RTF以实际处理时长为分母，JSON另存源时长。每帧13字节，含1字节AMR帧头，因此码流5200bps；标称4750bps只算有效语音编码位。不是LoRa空口码率。

本固件无无线与扬声器路径，没有提供RF首音、空口丢包、声学连续性；输出重复一致只证明本次9次串口编解码重复性。源码算得预分配缓存160800字节，算法状态峰值和系统峰值未提供，不能与AI解码arena峰值作内存下降比例。

三段质量、一次固定40采样对齐及原始输出见总表与quality目录。原接收程序会恢复，恢复证据位于baseline_amr/recovery。
'''
(P/'AMR基线指标明细.md').write_text(amrdoc,encoding='utf8')
method='''# 测评口径

1. 编解码与性能计时均在ESP32执行；电脑只固定输入、重采样、收日志与PCM、计算评分，不用PC模型或历史PC耗时替代板端数据。
2. 编解码核心耗时排除USB传输、录音、LoRa、提示音和播放；AI work_ms与AMR enc_us/dec_us是主指标。主表明确初始化是否被排除：两份循环计算计时均在状态初始化后开始，释放前结束。
3. 无线首PCM计时单独统计，起点为串口触发与BOOT相同的发送任务，终点为接收端开始提交已裁模型延迟的PCM；录完等待以采集结束为起点。不等同声学首字，保留自然静音。AMR单板没有该指标。
4. RTF同时保存实际生成/处理音频时长和源时长；当前主要纯解码输入是同源3秒固定语音，AI生成3.264秒，AMR按20ms帧处理3.280秒。报告按处理时长归一化，不能把这两个原始毫秒数当完全相同输出长度。
5. 窄带质量主表统一8kHz、PESQ-NB、STOI、SNR、分段SNR、SI-SDR。16kHz宽带分数可作为AI补充，但不与NB直接比。重采样固定使用scipy resample_poly、整数PCM统一舍入/限幅。
6. AI固定声明延迟4118@16k，AMR在fixed2s首轮一次标定40@8k后冻结所有样本。不逐条择优对齐、不预先拟合输出增益。40@8k只是导出波形偏移，不是算法总延迟。
7. 成功、丢包、发送超时、数据不一致及诊断模式全部列出；失败轮不偷偷从稳定性统计删除。不同输入、接收方式、间隔与外部发送条件分组，不用一次最优值代表版本。
8. 无效包计数不等于语音序列丢包。107的会话计数跨越外部发送暂停时刻，不计算暂停后的坏包率。发送未完成时，不把接收端的部分会话包简单归入当前语音。
9. 三次重复报告n、均值、中位数、最小最大；不据3次估计可靠的P95或长期故障概率。3段语音不是大规模独立语料集，固定2/3秒素材存在关联。AMR每段重复3次不是9段独立语音。
10. 标称码率、AMR帧头后的流码率、模型补尾与LoRa包头开销分开。AMR的5.2kbps不是空口码率，AI与AMR也不是同标称码率。
11. AI arena峰值、AMR源码静态缓存和系统heap不是同一口径。未测系统峰值、任务CPU占用、功耗时保留缺测，不用RTF冒充CPU占用。
12. 原源码、原模型、原AMR目录不改。新目录保存来源、固件哈希、输入与输出哈希、串口记录、CSV/JSON及恢复资料。当前AI异常未在本轮修改算法掩盖。
'''
(P/'测评口径.md').write_text(method,encoding='utf8')
flat=[{'metric':'decoder_mean_work_ms_fixed3s','AI':core['work_ms']['mean'],'AMR':amr_stats['fixed3s']['decoding_ms']['mean'],'unit':'ms','scope':'Source3s, actual output3.264s/3.280s, framework/configuration disclosed'},{'metric':'decoder_RTF_processed_fixed3s','AI':core['mean_rtf'],'AMR':amr_stats['fixed3s']['decoding_rtf_processed']['mean'],'unit':'time/audio time','scope':'Actual generated duration normalized'}]
for r in paired:
 for key in ['pesq_nb','stoi','snr_db','segmental_snr_db','si_sdr_db']:flat.append({'metric':r['case']+'_'+key,'AI':r['AI_'+key],'AMR':r['AMR_'+key],'unit':'score/dB','scope':r['scope']})
csvwrite(P/'对比总表.csv',flat)
print(json.dumps({'AI_decode_mean_ms':core['work_ms']['mean'],'AMR_fixed3_decode_mean_ms':amr_stats['fixed3s']['decoding_ms']['mean'],'normalized_RTF_ratio':speed,'quality_means':qmeans,'AMR_alignment_samples_8k':amr_delay,'all_AMR_output_stable':all(v['output_hash_stable'] for v in amr_stats.values()),'reports_generated':True},ensure_ascii=False,indent=2))
