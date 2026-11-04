"""Exercise real LVGL call buttons, LoRa voice, telemetry and peer rejoin.
 * Both endpoints are native; there is no SSH/IP control of the peer.
"""
import sys,argparse,json,time,re,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_uart import NativeUART,ROOT
p=argparse.ArgumentParser();p.add_argument('--board',choices=['master','slave'],required=True);p.add_argument('--samples',type=int,default=8);a=p.parse_args()
assert 4<=a.samples<=20
role='A' if a.board=='master' else 'B'
def read(u,name,optional=False):
 for attempt in range(3):
  raw=u.command('cat '+name,allow_error=True);m=re.search(r'\{[^\r\n]*\}',raw)
  if m:
   try:return json.loads(m[0])
   except ValueError:pass
 if optional and 'open failed: 2' in raw:return {}
 raise AssertionError(raw)
def point(u):
 s=read(u,'/tmp/k7_ui.json');assert s['design']=='original_console' and s['role']==role
 if s['tab']!=0:u.command('k7_ui tap 54 113');time.sleep(1.2);s=read(u,'/tmp/k7_ui.json')
 x,y,w,h=s['button'];assert w>200 and h>=60 and x>=0 and y+h<=682,s
 return s,x+w//2,y+h//2
folder=ROOT/'deployment'/time.strftime('native-ui-call-%Y%m%d-%H%M%S');folder.mkdir(parents=True)
r={'board':a.board,'role':role,'passed':False,'control':'real LVGL pointer events','network_used':False,'samples':[],'started_at':datetime.datetime.now().astimezone().isoformat()}
def save():(folder/'report.json').write_text(json.dumps(r,indent=2),encoding='utf8')
try:
 with NativeUART(a.board+'_original_ui_call') as u:
  before,x,y=point(u);r['before']=before;assert before['phase']=='ready','Pair must be registered before the UI call test'
  r['start']=u.command('k7_ui tap %d %d'%(x,y));at=time.monotonic()
  try:
   for i in range(a.samples):
    time.sleep(3);ui=read(u,'/tmp/k7_ui.json');current=read(u,'/tmp/k7_pair/current.json')
    # Admission, weight loading and audio startup precede the first status
    # file. Permit that window, but require real voice data from sample 3.
    call=read(u,'/tmp/k7_pair/call.json',optional=i<2) if current['program']=='k7_voice' else {}
    row={'elapsed_s':round(time.monotonic()-at,3),'ui':ui,'current':current,'call':call};r['samples'].append(row);save()
    print(json.dumps({'phase':ui['phase'],'tx':call.get('tx_packets'),'rx':call.get('rx_packets'),'gaps':call.get('seq_gaps'),'rssi':call.get('rssi_last_dbm'),'snr':call.get('snr_last_db')}),flush=True)
    if i>=2:
     assert current['program']=='k7_voice' and ui['phase']=='in_call' and ui['signal_valid'],row
     assert call['rx_packets']>=20 and call['seq_gaps']==0,row
  finally:
   # Always stop the actual business, even if a UI assertion fails.
   try:
    s,x,y=point(u);r['stop']=u.command('k7_ui tap %d %d'%(x,y))
   except Exception:r['emergency_stop']=u.command('k7_pair stop',allow_error=True)
  for i in range(30):
   time.sleep(.5);s=read(u,'/tmp/k7_ui.json')
   if s['phase']=='ready':break
  assert s['phase']=='ready','UI endpoints failed to stop and rejoin'
  r['after']=s;r['final_call']=read(u,'/tmp/k7_pair/previous_call.json')
  assert r['final_call']['tx_packets']>=30 and r['final_call']['rx_packets']>=30 and r['final_call']['seq_gaps']==0
  r['processes_after']=u.command('ps');assert '100.0%' not in r['processes_after']
  r.update(passed=True,uart_evidence=str(u.folder),bounded_stop_and_lora_rejoin_verified=True)
  print('ORIGINAL_UI_BUTTON_CALL_AND_REJOIN_PASS',flush=True)
finally:r['ended_at']=datetime.datetime.now().astimezone().isoformat();save();print('REPORT',folder,flush=True)
