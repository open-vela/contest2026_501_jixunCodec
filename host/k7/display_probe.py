"""Fast necessary display-link check, correlated with the observed black glass.
 * A green result is NOT sufficient for physical screen/touch acceptance.
"""
import sys,re,struct,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from native_uart import NativeUART,ROOT
with NativeUART('display_link_probe') as u:
 raw=u.command('xd 0x27d80018 16');data=bytearray()
 for line in raw.splitlines():
  m=re.match(r'^\s*[0-9a-fA-F]+: ((?:[0-9a-fA-F]{2} ){15}[0-9a-fA-F]{2})',line)
  if m:data.extend(bytes.fromhex(m[1]))
 assert len(data)==16,raw
 ctrl,mode,core,manual=struct.unpack('<4I',data)
 r={'mode_ctrl':ctrl,'mode_status':mode,'core_status':core,'manual':manual,'video_mode':(mode&7)==3,'uart_evidence':str(u.folder)}
 (ROOT/'deployment/display_link_probe_latest.json').write_text(json.dumps(r,indent=2))
 print(json.dumps(r),flush=True)
 assert r['video_mode'],'RED: DSI is not in video mode; framebuffer pixels alone do not prove display output'
