"""Exercise LVGL pointer events against the real framebuffer and widgets."""
import sys,json,re,time,struct,argparse
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_uart import NativeUART,ROOT
p=argparse.ArgumentParser();p.add_argument('--board',choices=['master','slave'],default='master');a=p.parse_args()
role='A' if a.board=='master' else 'B'
def state(u):
 for attempt in range(3):
  raw=u.command('cat /tmp/k7_ui.json');m=re.search(r'\{[^\r\n]+\}',raw)
  if m:return json.loads(m[0])
 raise AssertionError(raw)
r={'passed':False,'board':a.board,'role':role,'pages':[]}
try:
 with NativeUART(a.board+'_ui_navigation') as u:
  r['before']=state(u);assert r['before']['role']==role
  for i in [1,2,3,4,5,6,0]:
   original=r['before'].get('design')=='original_console'
   u.command('k7_ui tap '+('54 '+str(670 if i==6 else 113+i*79) if original else '80 '+str(145+i*74)));time.sleep(1.2)
   s=state(u);assert s['tab']==i,s;r['pages'].append(s)
  u.command('k7_ui snapshot /tmp/ui.rle',20);data=u.pull('/tmp/ui.rle')
  assert data[:8]==b'JXFB\x00\x05\xd0\x02';pixels=bytearray()
  for count,color in struct.iter_unpack('<HH',data[8:]):
   pixels.extend(bytes((((color>>11)&31)*255//31,((color>>5)&63)*255//63,(color&31)*255//31))*count)
  assert len(pixels)==1280*720*3
  Image.frombytes('RGB',(1280,720),bytes(pixels)).save(u.folder/'overview.png')
  r['after']=state(u);r['devices']=u.command('ls /dev');r['boot_status']=u.command('nbootctl status')
  r['evidence']=str(u.folder);r['screenshot']=str(u.folder/'overview.png');r['passed']=True
  print(json.dumps(r,indent=2),flush=True)
finally:(ROOT/'deployment'/('ui_'+a.board+'_navigation.json')).write_text(json.dumps(r,indent=2),encoding='utf8')
