"""Native NSH transport with byte-exact small-file transfers and SHA checks."""
import os,serial,serial.tools.list_ports,time,pathlib,json,datetime,re,hashlib
ROOT=pathlib.Path(__file__).resolve().parent
ANSI=re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

class NativeUART:
    def __init__(self,purpose,ready_timeout=5):
        assert purpose.replace('_','').isalnum()
        self.folder=ROOT/'artifacts/uart'/(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')+purpose)
        self.folder.mkdir(parents=True,exist_ok=False)
        (ROOT/'deployment').mkdir(exist_ok=True)
        self.raw=bytearray();self.rows=[]
        self.ready_timeout=ready_timeout
        self.serial=serial.Serial(port=None,baudrate=1500000,timeout=.02,write_timeout=3)
        self.serial.dtr=False;self.serial.rts=False;self.serial.port=os.environ.get('K7_UART_PORT','COM15')
    def receive(self,timeout=15):
        end=time.monotonic()+timeout;data=bytearray()
        while time.monotonic()<end:
            b=self.serial.read(min(max(self.serial.in_waiting,1),65536))
            self.raw.extend(b);data.extend(b)
            # Dumped firmware contains the literal PS1 string "nsh> ".
            # A real console prompt starts a line; an ASCII dump column does not.
            if re.search(rb'(?:^|[\r\n])nsh>(?: |$)|nsh> \x1b\[K',data):
                text=ANSI.sub('',data.decode('utf8',errors='replace'))
                diagnostics='\n'.join(line for line in text.splitlines()
                                      if not re.match(r'^[0-9a-fA-F]+: [0-9a-fA-F]{2} ',line))
                if 'Assertion failed' in diagnostics or 'dump_fatal_info' in diagnostics or 'DDR 2f85' in diagnostics:
                    raise RuntimeError('native panic/reboot during command')
                return text
        raise TimeoutError(data[-1200:].decode(errors='replace'))
    def __enter__(self):
        ports=[p for p in serial.tools.list_ports.comports() if p.device==self.serial.port]
        assert len(ports)==1 and ports[0].vid==0x1a86 and ports[0].pid==0x7523
        self.serial.open();self.serial.write(b'\r');self.receive(self.ready_timeout)
        assert 'NuttX' in self.command('uname -a')
        return self
    def command(self,cmd,timeout=15,allow_error=False):
        assert '\r' not in cmd and '\n' not in cmd
        start=len(self.raw);prompt_recovered=False
        self.serial.write(cmd.encode()+b'\r');self.serial.flush()
        try:output=self.receive(timeout)
        except TimeoutError:
            # The CH340 occasionally drops the end of the output prompt.
            # Ask for an empty fresh prompt; never replay a mutating command.
            self.serial.write(b'\r');self.serial.flush();self.receive(5)
            output=ANSI.sub('',self.raw[start:].decode('utf8',errors='replace'))
            prompt_recovered=True
        self.rows.append({'command':cmd,'output':output,'prompt_recovered':prompt_recovered})
        if not allow_error:
            assert not re.search(r'^nsh:',output,re.M),output[-1000:]
            assert not re.search(r'^nbootctl:.*(?:failed|cannot|mismatch)',output,re.M),output[-1000:]
        return output
    @staticmethod
    def path(path):
        assert re.fullmatch(r'/(?:tmp|data|sd|emmc)/[A-Za-z0-9_./-]+',path) and '..' not in path
        return path
    def digest(self,path):
        for attempt in range(3):
            out=self.command('nbootctl digest '+self.path(path),timeout=10)
            found=re.findall(r'^\s*(\d+) ([0-9a-f]{64})\s*$',out,re.M)
            if len(found)==1:return int(found[0][0]),found[0][1]
        raise AssertionError(out[-1000:])
    def push(self,path,data,words=24,progress=None,resume=False):
        self.path(path);assert data and len(data)%4==0 and 1<=words<=30
        start=0
        if resume:
            existing=self.command('nbootctl digest '+path,timeout=120,allow_error=True)
            match=re.search(r'^\s*(\d+) ([0-9a-f]{64})\s*$',existing,re.M)
            if match:
                start=int(match[1]);assert start<=len(data) and start%4==0
                assert hashlib.sha256(data[:start]).hexdigest()==match[2],'resume prefix SHA mismatch'
        if not start:self.command('truncate -s 0 '+path,timeout=2)
        for offset in range(start,len(data),words*4):
            values=["'\\x%08x'"%int.from_bytes(data[k:k+4],'little')
                    for k in range(offset,min(offset+words*4,len(data)),4)]
            self.command('printf '+' '.join(values)+' >> '+path,timeout=2)
            if progress and (offset//(words*4))%512==0:progress(offset,len(data))
        assert self.digest(path)==(len(data),hashlib.sha256(data).hexdigest()),'UART upload SHA mismatch'
    def pull(self,path):
        expected_size,expected_hash=self.digest(path)
        # NSH uses a short caption buffer: long paths truncate the block offset.
        # A short tmpfs copy preserves full offsets and the original file digest.
        readable=self.path(path)
        if len(readable)>16:
            self.command('cp '+readable+' /tmp/uget.bin',timeout=120)
            readable='/tmp/uget.bin'
            assert self.digest(readable)==(expected_size,expected_hash)
        # The CH340/console path can drop lines during a long output burst.
        # Keep each dump to one sector, rejecting malformed output and checking
        # the complete downloaded file against its target-side SHA afterwards.
        result=bytearray()
        for offset in range(0,expected_size,512):
            wanted=min(512,expected_size-offset)
            self.command('dd if='+readable+' of=/tmp/ux.bin bs=512 skip='+str(offset//512)+' count=1')
            for attempt in range(3):
                try:
                    block=self.parse_hexdump(self.command('hexdump /tmp/ux.bin'))
                    assert len(block)==wanted,'UART sector length mismatch'
                    break
                except AssertionError:
                    if attempt==2:raise
            result.extend(block)
        assert len(result)==expected_size and hashlib.sha256(result).hexdigest()==expected_hash,'UART download SHA mismatch'
        return bytes(result)
    @staticmethod
    def parse_hexdump(out):
        result=bytearray();base=0
        for line in out.splitlines():
            heading=re.search(r' at ([0-9a-fA-F]+):$',line)
            if heading:
                base=int(heading[1],16)
                assert base==len(result),'hexdump block offset discontinuity'
                continue
            match=re.match(r'^([0-9a-fA-F]+): (.*)$',line)
            if not match:continue
            assert base+int(match[1],16)==len(result),'hexdump offset discontinuity'
            values=match[2][:47].split();assert all(re.fullmatch('[0-9a-fA-F]{2}',v) for v in values)
            result.extend(int(v,16) for v in values)
        return bytes(result)
    def __exit__(self,*args):
        self.serial.close()
        (self.folder/'raw.bin').write_bytes(self.raw)
        (self.folder/'console.txt').write_text(self.raw.decode(errors='replace'),encoding='utf8')
        (self.folder/'commands.json').write_text(json.dumps(self.rows,indent=2),encoding='utf8')

if __name__=='__main__':
    payload=bytes(range(256))
    with NativeUART('file_transfer_check') as uart:
        uart.push('/tmp/uf.bin',payload,words=1)
        assert uart.pull('/tmp/uf.bin')==payload
        print(json.dumps({'uart_upload_and_download_all_byte_values_sha_verified':True,
                          'evidence':str(uart.folder),'emmc_written':False}),flush=True)
