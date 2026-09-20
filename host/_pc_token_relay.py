#!/usr/bin/env python3
"""PC token broker for the SAFE DEMO over two USB serial ports."""

import argparse
import json
import math
import queue
import struct
import subprocess
import sys
import threading
import time
import zlib
from collections import deque
from pathlib import Path

import serial


sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PREFIX = "@@JXTOK1:"
PCM_PREFIX = "@@JXPCM1:"
SERIAL_PAYLOAD_MAX = 192
HEADER = struct.Struct("<IHHHHIIIIIIIII")
ROOT = Path(__file__).resolve().parent
VERBOSE = False
UI_EVENT_FILE = ROOT / "_pc_voice_rx" / "ui_events.jsonl"
RATE_PROFILES = {
    "1k": {"id": 0, "bits": 17},
    "3k": {"id": 1, "bits": 18},
    "6k": {"id": 2, "bits": 18},
}


def publish_ui_line(line):
    try:
        UI_EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with UI_EVENT_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line.rstrip() + "\n")
    except OSError:
        pass

PCM_LAYOUT = {
    "raw": ("raw_record.wav", 2, 1),
    "pre": ("decoded_pregain.wav", 4, 3),
    "play": ("decoded_playback.wav", 2, 1),
}


def interesting(line):
    return ("[usb]" in line or "[Voice]" in line or "[ERR]" in line or
            "[WARN]" in line or "播放块" in line or "录音统计" in line or
            "录音结束" in line or "编码" in line or "解码" in line or
            "第一块" in line or "流式" in line)


def pcm_stats(data, width):
    count = len(data) // width
    if count <= 0:
        return {"samples": 0}

    if width == 2:
        values = struct.unpack(f"<{count}h", data[:count * 2])
        clipped = sum(1 for value in values
                      if value >= 32767 or value <= -32768)
        peak = max(abs(value) for value in values)
        full_scale = 32768.0
    else:
        values = struct.unpack(f"<{count}f", data[:count * 4])
        clipped = sum(1 for value in values if abs(value) >= 1.0)
        peak = max(abs(value) for value in values)
        full_scale = 1.0

    rms = math.sqrt(sum(float(value) * float(value)
                        for value in values) / count)
    return {
        "samples": count,
        "peak": peak,
        "peak_dbfs": (20.0 * math.log10(peak / full_scale)
                      if peak > 0.0 else -999.0),
        "rms": rms,
        "rms_dbfs": (20.0 * math.log10(rms / full_scale)
                     if rms > 0.0 else -999.0),
        "clipped_samples": clipped,
    }


def write_wav(path, data, rate, width, format_tag):
    data = data[:len(data) - (len(data) % width)]
    header = (
        b"RIFF" +
        struct.pack("<I", 36 + len(data)) +
        b"WAVEfmt " +
        struct.pack("<IHHIIHH", 16, format_tag, 1, rate,
                    rate * width, width, width * 8) +
        b"data" +
        struct.pack("<I", len(data))
    )
    path.write_bytes(header + data)


class PcmCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.message_buffers = {kind: bytearray() for kind in PCM_LAYOUT}
        self.run_buffers = {kind: bytearray() for kind in PCM_LAYOUT}
        self.expected = {kind: 0 for kind in PCM_LAYOUT}
        self.rates = {kind: 0 for kind in PCM_LAYOUT}
        self.message_chunks = {kind: 0 for kind in PCM_LAYOUT}
        self.blocks = {kind: 0 for kind in PCM_LAYOUT}
        self.gaps = {kind: 0 for kind in PCM_LAYOUT}
        self.summaries = {}

    def finalize(self, kind):
        payload_bytes = bytes(self.run_buffers[kind])
        if not payload_bytes:
            return

        filename, width, format_tag = PCM_LAYOUT[kind]
        output = self.directory / filename
        write_wav(output, payload_bytes, self.rates[kind], width, format_tag)
        stats = pcm_stats(payload_bytes, width)
        stats.update({
            "file": str(output),
            "rate": self.rates[kind],
            "width": width,
            "chunks": self.message_chunks[kind],
            "blocks": self.blocks[kind],
            "sequence_gaps": self.gaps[kind],
        })
        self.summaries[kind] = stats
        print(f"[PCM] {kind}: {output} "
              f"samples={stats['samples']} chunks={stats['chunks']} "
              f"blocks={stats['blocks']} gaps={stats['sequence_gaps']} "
              f"peak_dbfs={stats['peak_dbfs']:.1f} "
              f"rms_dbfs={stats['rms_dbfs']:.1f} "
              f"clipped={stats['clipped_samples']}")

        self.run_buffers[kind].clear()
        self.message_buffers[kind].clear()
        self.message_chunks[kind] = 0
        self.blocks[kind] = 0
        self.gaps[kind] = 0
        self.expected[kind] = 0

    def handle(self, line):
        marker = line.find(PCM_PREFIX)
        if marker < 0:
            return False

        payload = line[marker + len(PCM_PREFIX):].strip()
        parts = payload.split(":", 4)
        if len(parts) != 5:
            print(f"[PCM] malformed line: {payload[:80]}", file=sys.stderr)
            return True

        kind, seq_text, last_text, rate_text, hexdata = parts
        if kind not in PCM_LAYOUT:
            print(f"[PCM] unknown kind: {kind}", file=sys.stderr)
            return True

        try:
            seq = int(seq_text)
            last = int(last_text) != 0
            rate = int(rate_text)
            data = bytes.fromhex(hexdata)
        except ValueError:
            print(f"[PCM] invalid fields: {payload[:80]}", file=sys.stderr)
            return True

        if kind == "raw" and seq == 0:
            self.finalize("pre")
            self.finalize("play")
            self.gaps[kind] = 0

        if seq == 0:
            self.message_buffers[kind].clear()
            self.expected[kind] = 0

        if seq != self.expected[kind]:
            self.gaps[kind] += 1
            print(f"[PCM] {kind} sequence gap: got {seq}, "
                  f"expected {self.expected[kind]}", file=sys.stderr)

        self.message_buffers[kind].extend(data)
        self.expected[kind] = seq + 1
        self.message_chunks[kind] += 1
        self.rates[kind] = rate

        if last:
            self.run_buffers[kind].extend(self.message_buffers[kind])
            self.blocks[kind] += 1
            self.message_buffers[kind].clear()
            self.expected[kind] = 0
            if kind == "raw":
                self.finalize("raw")

        return True

    def finish(self, async_quality=False):
        self.finalize("raw")
        self.finalize("pre")
        self.finalize("play")
        if not self.summaries:
            print("[PCM] no complete audio captures")
            return

        if async_quality:
            quality_log = self.directory / "quality_async.log"
            flags = 0
            flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
            flags |= getattr(subprocess, "DETACHED_PROCESS", 0)
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            with quality_log.open("ab") as log_handle:
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "_pcm_quality.py"),
                     str(self.directory), "--write"],
                    cwd=str(ROOT),
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=flags,
                )
            print(f"[QUALITY] async pid={proc.pid} log={quality_log}")
            return

        try:
            from _pcm_quality import analyze_capture_dir

            quality = analyze_capture_dir(self.directory)
            if quality:
                self.summaries["quality"] = quality
                for name, metrics in quality["comparisons"].items():
                    raw_snr = metrics.get("snr_raw_db")
                    adjusted_snr = metrics.get("snr_gain_adjusted_db")
                    pesq = metrics.get("pesq_wb")
                    raw_text = f"{raw_snr:.2f}" if raw_snr is not None else "n/a"
                    adjusted_text = (f"{adjusted_snr:.2f}"
                                     if adjusted_snr is not None else "n/a")
                    pesq_text = f"{pesq:.3f}" if pesq is not None else "n/a"
                    print(f"[SNR] raw vs {name}: "
                          f"raw={raw_text} dB "
                          f"gain_adjusted={adjusted_text} dB "
                          f"lag={metrics['best_lag_ms']:.2f} ms "
                          f"corr={metrics['correlation']:.4f} "
                          f"PESQ={pesq_text}")
        except Exception as exc:
            print(f"[SNR] unavailable: {exc}", file=sys.stderr)

        summary_path = self.directory / "pcm_summary.json"
        summary_path.write_text(
            json.dumps(self.summaries, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[PCM] summary: {summary_path}")


class Board:
    def __init__(self, name, port, baud):
        self.name = name
        self.port = serial.Serial()
        self.port.port = port
        self.port.baudrate = baud
        self.port.timeout = 0.05
        self.port.write_timeout = 1.0
        self.port._dtr_state = False
        self.port._rts_state = False
        self.port.open()
        self.port.dtr = False
        self.port.rts = False
        self.port.reset_input_buffer()
        self.pending = ""
        self.lines = []
        self.lock = threading.Lock()
        self.lines_q = queue.Queue()
        self.stop_event = threading.Event()
        self.reader = None

    def write(self, text):
        with self.lock:
            self.port.write(text.encode("utf-8", "replace"))

    def command(self, text):
        self.write(text + "\r\n")

    def read_once(self):
        with self.lock:
            data = self.port.read(4096)
        if not data:
            return []
        self.pending += data.decode("utf-8", "replace")
        ready = []
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            line = line.rstrip("\r")
            ready.append(line)
        return ready

    def start_reader(self):
        if self.reader is not None:
            return

        def run():
            while not self.stop_event.is_set():
                try:
                    for line in self.read_once():
                        self.lines_q.put(line)
                except Exception:
                    break

        self.reader = threading.Thread(target=run, daemon=True)
        self.reader.start()

    def next_line(self):
        try:
            return self.lines_q.get_nowait()
        except queue.Empty:
            return None

    def close(self):
        self.stop_event.set()
        if self.reader is not None:
            self.reader.join(timeout=1.0)
        self.port.close()


def wait_prompt(board, timeout=30):
    board.start_reader()
    end = time.time() + timeout
    while time.time() < end:
        board.command("")
        end_read = time.time() + 0.25
        while time.time() < end_read:
            line = board.next_line()
            if line is None:
                time.sleep(0.005)
                continue
            if VERBOSE and line:
                print(f"[{board.name}] {line}")
            if "nsh>" in line:
                return True
        time.sleep(0.02)
    return False


def parse_frame(line):
    marker = line.find(PREFIX)
    if marker < 0:
        return None
    text = line[marker + len(PREFIX):].strip()
    try:
        data = bytes.fromhex(text)
    except ValueError:
        return None
    if len(data) < HEADER.size:
        return None
    values = HEADER.unpack_from(data, 0)
    names = (
        "magic", "ver", "rate_id", "bits", "reserved", "ntok", "tt",
        "orig_t", "seq", "ttl", "payload", "crc", "chunk", "last",
    )
    header = dict(zip(names, values))
    if header["magic"] != 0x3143584A or header["ver"] != 2:
        return None
    payload_len = int(header["payload"])
    payload = data[HEADER.size:HEADER.size + payload_len]
    if len(payload) != payload_len:
        return None
    return header, payload, data


def unpack_tokens(payload, ntok, bits):
    values = []
    bit = 0
    for _ in range(ntok):
        value = 0
        for _ in range(bits):
            value = (value << 1) | ((payload[bit >> 3] >> (7 - (bit & 7))) & 1)
            bit += 1
        values.append(value)
    return values


def format_header(header, payload):
    preview_count = min(int(header["ntok"]),
                        (len(payload) * 8) // int(header["bits"]))
    tokens = unpack_tokens(payload, preview_count, int(header["bits"]))
    preview = ",".join(str(v) for v in tokens[:8])
    if len(tokens) > 8:
        preview += ",..."
    ntok = int(header["ntok"])
    orig_samples = int(header["orig_t"])
    tokens_per_second = (ntok * 16000.0 / orig_samples
                         if orig_samples > 0 else 0.0)
    return (
        f"chunk={int(header['chunk']):02d} "
        f"tokens={ntok:02d} "
        f"bits={int(header['bits']):02d} "
        f"bytes={len(payload):02d} "
        f"frag={int(header['seq'])}/{int(header['ttl'])} "
        f"tokens_per_second={tokens_per_second:.1f} "
        f"orig_samples={orig_samples} "
        f"first=[{preview}]"
    )


def decode_tokens_to_wav(tokens, rate_id, orig_t, out_dir, tag="pc_decode"):
    """Decode one complete token snapshot on the PC."""
    out_dir = Path(out_dir).resolve()
    names = {0: "sq_1k_codec_main.exe", 1: "sq_3k_codec_main.exe",
             2: "sq_6k_codec_main.exe"}
    exe = (ROOT / "codec_code" / "sqcodec-light-c-v1" / "build" /
           names.get(int(rate_id), names[1]))
    if not exe.exists() or not tokens:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = out_dir / f"{tag}.idx"
    wav_path = out_dir / f"{tag}.wav"
    with idx_path.open("wb") as handle:
        handle.write(struct.pack("<iii", 1, len(tokens), int(orig_t)))
        for value in tokens:
            handle.write(struct.pack("<q", value))

    started = time.perf_counter()
    result = subprocess.run(
        [str(exe), "recv", str(idx_path), str(wav_path)],
        cwd=str(exe.parent), capture_output=True, text=True,
    )
    elapsed = time.perf_counter() - started
    if result.returncode != 0 or not wav_path.exists():
        print(f"[PC-DECODE] failed rc={result.returncode}: "
              f"{result.stdout.strip()} {result.stderr.strip()}",
              file=sys.stderr)
        return None

    return {"wav": wav_path, "seconds": elapsed,
            "tokens": len(tokens), "orig": int(orig_t)}


def decode_frames_to_wav(frames, rate_id, out_dir):
    """Decode a complete token stream on the PC as a latency/quality fallback."""
    groups = {}
    total_orig = 0

    for item in frames:
        header = item["header"]
        chunk = int(header["chunk"])
        seq = int(header["seq"])
        if chunk not in groups:
            groups[chunk] = {
                "header": header,
                "parts": {},
            }
            total_orig += int(header["orig_t"])
        groups[chunk]["parts"][seq] = bytes.fromhex(item["payload_hex"])

    if not groups:
        return None

    tokens = []
    for chunk in sorted(groups):
        group = groups[chunk]
        header = group["header"]
        payload = b"".join(group["parts"][seq]
                           for seq in sorted(group["parts"]))
        tokens.extend(unpack_tokens(payload, int(header["ntok"]),
                                    int(header["bits"])))

    return decode_tokens_to_wav(tokens, rate_id, total_orig, out_dir)


def play_wav_async(path):
    def worker():
        try:
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME)
        except Exception as exc:
            print(f"[PC-DECODE] playback failed: {exc}", file=sys.stderr)

    thread = threading.Thread(target=worker, daemon=False)
    thread.start()
    return thread


class PcStreamPlayer:
    """Continuously play 16 kHz mono PCM decoded from a PC token snapshot."""

    def __init__(self, out_wav, prebuffer_ms=900):
        self.out_wav = Path(out_wav)
        self.prebuffer_bytes = int(16000 * 2 * prebuffer_ms / 1000)
        self.buf = bytearray()
        self.all_pcm = bytearray()
        self.lock = threading.Lock()
        self.started = False
        self.finished = False
        self.underruns = 0
        self._underrun_reported = False
        self.first_play_at = None
        self.stream = None

    def feed(self, pcm_bytes):
        if not pcm_bytes:
            return
        with self.lock:
            self.buf.extend(pcm_bytes)
            self.all_pcm.extend(pcm_bytes)
            should_start = (not self.started and
                            len(self.buf) >= self.prebuffer_bytes)
        if should_start:
            self._start()

    def finish(self):
        with self.lock:
            self.finished = True
            should_start = not self.started and bool(self.all_pcm)
        if should_start:
            self._start()

    def _start(self):
        with self.lock:
            if self.started:
                return
            self.started = True
        try:
            import sounddevice as sd
        except Exception as exc:
            print(f"[PC-STREAM] sounddevice unavailable: {exc}",
                  file=sys.stderr)
            return

        def callback(outdata, frames, _time_info, _status):
            need = frames * 2
            with self.lock:
                take = min(need, len(self.buf))
                if take:
                    outdata[:take] = self.buf[:take]
                    del self.buf[:take]
                if take < need:
                    outdata[take:need] = b"\x00" * (need - take)
                    if not self.finished:
                        self.underruns += 1
                        if not self._underrun_reported:
                            self._underrun_reported = True
                            print("[PC-STREAM] playback underrun")

        self.stream = sd.RawOutputStream(
            samplerate=16000, channels=1, dtype="int16",
            blocksize=1024, callback=callback,
        )
        self.stream.start()
        self.first_play_at = time.time()
        print("[PC-STREAM] playback started")

    def close(self):
        deadline = time.time() + 30.0
        while time.time() < deadline:
            with self.lock:
                if not self.buf:
                    break
            time.sleep(0.05)
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def save(self):
        import wave
        self.out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.out_wav), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(bytes(self.all_pcm))


class PcStreamDecoder:
    """Decode cumulative token snapshots and feed only newly emitted PCM."""

    def __init__(self, rate_id, out_dir, step_ms=200, prebuffer_ms=900):
        self.rate_id = int(rate_id)
        self.out_dir = Path(out_dir).resolve()
        self.tokens = []
        self.last_decoded = 0
        self.finished = False
        self.first_chunk_at = None
        self.last_chunk_at = None
        self.condition = threading.Condition()
        self.player = PcStreamPlayer(
            self.out_dir / "pc_stream.wav", prebuffer_ms=prebuffer_ms)
        self.hop = {0: 90, 1: 96, 2: 48}.get(self.rate_id, 96)
        self.step_tokens = max(
            1, int(round(step_ms * 16000.0 / 1000.0 / self.hop)))
        self.worker = threading.Thread(target=self._run, daemon=False)
        self.worker.start()

    def add_chunk(self, values, last):
        with self.condition:
            now = time.time()
            if self.first_chunk_at is None:
                self.first_chunk_at = now
            if last:
                self.last_chunk_at = now
            self.tokens.extend(values)
            if last:
                self.finished = True
            self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                while (not self.finished and
                       len(self.tokens) - self.last_decoded < self.step_tokens):
                    self.condition.wait()
                target = len(self.tokens)
                is_last = self.finished
                snapshot = self.tokens[:target]

            if target <= self.last_decoded:
                if is_last:
                    break
                continue

            result = decode_tokens_to_wav(
                snapshot, self.rate_id, target * self.hop,
                self.out_dir, tag="pc_stream",
            )
            if result is None:
                with self.condition:
                    self.finished = True
                break

            audio_seconds = target * self.hop / 16000.0
            print(f"[PC-STREAM] decoded tokens={target} "
                  f"audio={audio_seconds:.3f}s "
                  f"decode={result['seconds']:.3f}s")

            import wave
            with wave.open(str(result["wav"]), "rb") as handle:
                pcm = handle.readframes(handle.getnframes())
            already = len(self.player.all_pcm)
            if len(pcm) > already:
                self.player.feed(pcm[already:])

            with self.condition:
                self.last_decoded = target
                more_ready = (len(self.tokens) - self.last_decoded >=
                              self.step_tokens)
                done = self.finished and self.last_decoded >= len(self.tokens)
            if done:
                self.player.finish()
                break
            if not more_ready and not self.finished:
                continue

    def close(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()
        self.worker.join(timeout=30.0)
        self.player.finish()
        self.player.save()
        self.player.close()
        if self.first_chunk_at is not None and self.player.first_play_at is not None:
            print(f"[PC-STREAM] first_play_ms="
                  f"{1000.0 * (self.player.first_play_at - self.first_chunk_at):.0f} "
                  f"underruns={self.player.underruns}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tx-port", default="COM4")
    parser.add_argument("--rx-port", default="COM23")
    parser.add_argument("--rate", choices=tuple(RATE_PROFILES), default="3k")
    parser.add_argument("--rx-pf", type=int, choices=(0, 1), default=None,
                        help="override JX_PF on the receiver for stability A/B")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=int, default=2)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--ack-timeout", type=float, default=4.0)
    parser.add_argument("--ack-retries", type=int, default=2)
    parser.add_argument("--save", default="")
    parser.add_argument("--pcm-dir", default="")
    parser.add_argument("--no-pcm-dump", action="store_true")
    parser.add_argument("--async-quality", action="store_true")
    parser.add_argument("--quality-pass", action="store_true")
    parser.add_argument("--play-gain", type=int, default=1)
    parser.add_argument("--mic-gain", type=int, default=14)
    parser.add_argument("--legacy-skip", type=int, default=0)
    parser.add_argument("--dual-half", action="store_true")
    parser.add_argument("--dual-pipeline", action="store_true")
    parser.add_argument("--live-tx", action="store_true")
    parser.add_argument("--rx-batch-chunks", type=int, default=1)
    parser.add_argument("--buffer-play", action="store_true")
    parser.add_argument("--stream-play", action="store_true")
    parser.add_argument("--pc-decode", action="store_true")
    parser.add_argument("--pc-stream", action="store_true")
    parser.add_argument("--pc-stream-step-ms", type=int, default=200)
    parser.add_argument("--pc-stream-prebuffer-ms", type=int, default=900)
    parser.add_argument("--inject-wav", default="")
    parser.add_argument("--no-configure", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--beep", action="store_true")
    args = parser.parse_args()
    if args.pc_stream:
        args.pc_decode = False
    global VERBOSE
    VERBOSE = args.verbose

    tx = Board("TX", args.tx_port, args.baud)
    rx = Board("RX", args.rx_port, args.baud)
    queue = deque()
    pending_groups = {}
    batch_pending = []
    in_flight = False
    in_flight_group = None
    in_flight_started = 0.0
    ack_retries = 0
    sent_groups = 0
    played = 0
    quality_play_dumps = 0
    frames = []
    injected = False
    pc_decoded = False
    pc_play_thread = None
    pc_stream = None
    pcm_dir = (Path(args.pcm_dir) if args.pcm_dir else
               (Path(args.save).parent if args.save else Path("_pc_voice_rx")))
    capture = PcmCapture(pcm_dir)

    def send_group(group, retry=False):
        nonlocal sent_groups, in_flight_group, in_flight_started, ack_retries

        if args.rx_batch_chunks <= 1:
            for header, data in group:
                rx.write(PREFIX + data.hex() + "\r\n")
        else:
            batch_pending.extend(group)
            chunks = {int(item[0]["chunk"]) for item in batch_pending}
            is_last = any(int(item[0]["last"]) for item in batch_pending)
            if len(chunks) < args.rx_batch_chunks and not is_last:
                return
            first = batch_pending[0][0]
            payload = b"".join(item[1] for item in batch_pending)
            merged = dict(first)
            merged["ntok"] = sum(int(item[0]["ntok"]) for item in batch_pending)
            merged["tt"] = sum(int(item[0]["tt"]) for item in batch_pending)
            merged["orig_t"] = sum(int(item[0]["orig_t"]) for item in batch_pending)
            merged["seq"] = 0
            merged["ttl"] = 1
            merged["payload"] = len(payload)
            merged["chunk"] = int(first["chunk"])
            merged["last"] = 1 if is_last else 0
            name_order = ("magic", "ver", "rate_id", "bits", "reserved",
                          "ntok", "tt", "orig_t", "seq", "ttl", "payload",
                          "crc", "chunk", "last")
            fragments = [payload[i:i + SERIAL_PAYLOAD_MAX]
                         for i in range(0, len(payload), SERIAL_PAYLOAD_MAX)]
            merged["ttl"] = len(fragments)
            for seq, fragment in enumerate(fragments):
                merged["seq"] = seq
                merged["payload"] = len(fragment)
                merged["crc"] = zlib.crc32(fragment) & 0xffffffff
                data = (HEADER.pack(*(merged[name] for name in name_order)) +
                        fragment)
                rx.write(PREFIX + data.hex() + "\r\n")
            batch_pending.clear()
        if not retry:
            sent_groups += 1
        first = group[0][0]
        prefix = "[RETRY]" if retry else "[RELAY]"
        relay_line = (f"{prefix} t={time.time() - run_t0:.3f}s "
                      f"sent chunk={int(first['chunk']):02d} "
                      f"fragments={len(group)} bytes={sum(len(item[1]) for item in group)}")
        print(relay_line)
        publish_ui_line(relay_line)
        in_flight_group = group
        in_flight_started = time.time()
        if not retry:
            ack_retries = 0

    run_t0 = time.time()
    try:
        UI_EVENT_FILE.parent.mkdir(parents=True, exist_ok=True)
        UI_EVENT_FILE.write_text(
            f"[RUN] tx={args.tx_port} rx={args.rx_port} "
            f"rate={args.rate} seconds={args.seconds} "
            f"chunk_ms={args.chunk_ms}\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    def phase(label):
        phase_line = f"[TIME] {label} t={time.time() - run_t0:.3f}s"
        print(phase_line, flush=True)
        publish_ui_line(phase_line)

    try:
        if not wait_prompt(tx) or not wait_prompt(rx):
            print("NSH prompt timeout", file=sys.stderr)
            return 2
        phase("nsh_ready")

        if not args.no_configure:
            for board in (tx, rx):
                board.command("set JX_USB 1")
                board.command(f"set JX_BEEP {1 if (args.beep and not args.quality_pass) else 0}")
                board.command(f"set JX_PCM_DUMP "
                              f"{0 if args.no_pcm_dump else 1}")
                board.command("set JX_I8AQ 2")
                time.sleep(0.08)
            rx.command(f"set JX_PLAY_GAIN {args.play_gain}")
            rx.command(f"set JX_BUFFER_PLAY "
                       f"{1 if args.buffer_play else 0}")
            rx.command(f"set JX_STREAM_PLAY "
                       f"{1 if args.stream_play else 0}")
            rx.command(f"set JX_PLAY_MUTE "
                       f"{1 if (args.pc_stream or args.pc_decode or args.quality_pass) else 0}")
            rx.command(f"set JX_LEGACY_SKIP {args.legacy_skip}")
            stability_rx_pf = args.rx_pf
            if stability_rx_pf is None and args.rate in {"1k", "6k"}:
                stability_rx_pf = 0
            if stability_rx_pf is not None:
                rx.command(f"set JX_PF {stability_rx_pf}")
            tx.command(f"set JX_MICGAIN {args.mic_gain}")
            tx.command(f"set JX_DUAL_SELF {1 if args.dual_half else 0}")
            rx.command(f"set JX_DUAL_HALF {1 if args.dual_half else 0}")
            tx.command(f"set JX_DUAL_TX_REST "
                       f"{1 if args.dual_pipeline else 0}")
            tx.command(f"set JX_LIVE_TX "
                       f"{1 if args.live_tx else 0}")
            if args.rate == "6k":
                tx.command("set JX_FASTNET 0")
                tx.command("set JX_ENC_CACHE 0")
                tx.command("set JX_SKIP_DEC 1")
                tx.command("set JX_PF 0")
            rx.command(f"set JX_DUAL_RX_FIRST "
                       f"{1 if args.dual_pipeline else 0}")
            total_chunks = (
                (args.seconds * 1000 + args.chunk_ms - 1) // args.chunk_ms
                if args.chunk_ms > 0 else 1
            )
            rx.command(f"set JX_RX_CHUNKS {total_chunks}")
            time.sleep(0.08)
        phase("configured")

        rx.command("jixun net rx 45678 120")
        rx_ready = False
        start = time.time()
        while time.time() - start < 8:
            while True:
                line = rx.next_line()
                if line is None:
                    break
                if VERBOSE or interesting(line):
                    print(f"[RX] {line}")
                if "接收端" in line:
                    rx_ready = True
            if rx_ready:
                break
            time.sleep(0.02)
        phase("rx_ready")

        tx.command(
            f"jixun net tx 127.0.0.1 {args.seconds} 45678 {args.chunk_ms}"
        )
        phase("tx_started")

        print("=== AI TOKEN RELAY ===")
        print("PC forwards tokens only. No PCM is relayed.")
        print(f"PCM capture directory: {pcm_dir}")
        print(f"requested playback gain: {args.play_gain}x")
        print(f"requested microphone gain: {args.mic_gain}")
        print(f"buffered playback: {1 if args.buffer_play else 0}")
        print(f"stream playback: {1 if args.stream_play else 0}")
        print(f"PC stream playback: {1 if args.pc_stream else 0}")
        print(f"dual-board split decode: {1 if args.dual_half else 0}")
        print(f"dual-board pipeline decode: {1 if args.dual_pipeline else 0}")
        start = time.time()
        while time.time() - start < args.timeout:
            tx_lines = []
            rx_lines = []
            while True:
                line = tx.next_line()
                if line is None:
                    break
                tx_lines.append(line)
            while True:
                line = rx.next_line()
                if line is None:
                    break
                rx_lines.append(line)

            for line in tx_lines:
                if (args.inject_wav and not injected and
                        ("录音中" in line or "live tx:" in line)):
                    try:
                        import winsound

                        winsound.PlaySound(
                            str(Path(args.inject_wav).resolve()),
                            winsound.SND_FILENAME | winsound.SND_ASYNC,
                        )
                        injected = True
                        inject_line = (f"[INJECT] t={time.time() - run_t0:.3f}s "
                                       f"playing {args.inject_wav}")
                        print(inject_line)
                        publish_ui_line(inject_line)
                    except Exception as exc:
                        print(f"[INJECT] failed: {exc}", file=sys.stderr)
                if capture.handle(line):
                    continue
                frame = parse_frame(line)
                if frame is None:
                    if (VERBOSE or interesting(line)) and line.strip():
                        print(f"[TX] {line}")
                    continue
                header, payload, data = frame
                expected = RATE_PROFILES[args.rate]
                if (int(header["rate_id"]) != expected["id"] or
                        int(header["bits"]) != expected["bits"]):
                    warning = (f"[WARN] expected {args.rate} "
                               f"id={expected['id']} bits={expected['bits']}, got "
                               f"id={int(header['rate_id'])} bits={int(header['bits'])}")
                    print(warning, file=sys.stderr)
                    publish_ui_line(f"[WARN] {warning}")
                frames.append({
                    "header": header,
                    "payload_hex": payload.hex(),
                    "host_time": time.time(),
                })
                token_line = (f"[TOKEN] t={time.time() - run_t0:.3f}s "
                              f"{format_header(header, payload)}")
                print(token_line)
                publish_ui_line(token_line)

                seq = int(header["seq"])
                ttl = int(header["ttl"])
                chunk = int(header["chunk"])
                group = pending_groups.setdefault(chunk, {})
                group[seq] = (header, data)
                if len(group) == ttl:
                    ready = [group[i] for i in range(ttl)]
                    del pending_groups[chunk]
                    if args.pc_stream:
                        ready_header = ready[0][0]
                        ready_payload = b"".join(item[1] for item in ready)
                        ready_tokens = unpack_tokens(
                            ready_payload, int(ready_header["ntok"]),
                            int(ready_header["bits"]))
                        if pc_stream is None:
                            pc_stream = PcStreamDecoder(
                                ready_header["rate_id"],
                                pcm_dir / "pc_stream",
                                step_ms=args.pc_stream_step_ms,
                                prebuffer_ms=args.pc_stream_prebuffer_ms,
                            )
                        pc_stream.add_chunk(
                            ready_tokens, bool(ready_header["last"]))
                    if args.rx_batch_chunks > 1:
                        send_group(ready)
                    else:
                        queue.append(ready)
                        if not in_flight:
                            send_group(queue.popleft())
                            in_flight = True

            for line in rx_lines:
                if capture.handle(line):
                    continue
                if VERBOSE or interesting(line):
                    rx_line = f"[RX] t={time.time() - run_t0:.3f}s {line}"
                    print(rx_line)
                    publish_ui_line(rx_line)
                if args.quality_pass and "[usb] pcm dump kind=play" in line:
                    quality_play_dumps += 1
                if "[usb] received chunk=" in line:
                    if args.rx_batch_chunks <= 1:
                        in_flight_group = None
                        ack_retries = 0
                        if queue:
                            send_group(queue.popleft())
                        else:
                            in_flight = False
                if line.startswith("[Voice]") and "播放完成" in line and frames:
                    played += 1
                    last_chunk = max(int(item["header"]["chunk"])
                                     for item in frames)
                    expected_plays = 1 if args.buffer_play else last_chunk + 1
                    progress_line = (f"[PROGRESS] t={time.time() - run_t0:.3f}s "
                                     f"played={played}/{expected_plays} "
                                     f"relayed={sent_groups}")
                    print(progress_line)
                    publish_ui_line(progress_line)

            last_chunk = (max(int(item["header"]["chunk"]) for item in frames)
                          if frames else -1)
            expected_plays = 1 if args.buffer_play else last_chunk + 1
            if (args.pc_decode and not pc_decoded and frames and
                    frames[-1]["header"]["last"] and not pending_groups):
                result = decode_frames_to_wav(
                    frames, frames[-1]["header"]["rate_id"],
                    pcm_dir / "pc_decode",
                )
                if result is not None:
                    print(f"[PC-DECODE] tokens={result['tokens']} "
                          f"orig={result['orig']} "
                          f"seconds={result['seconds']:.3f} "
                          f"rtf={result['seconds'] / max(0.001, result['orig'] / 16000):.2f}")
                    pc_play_thread = play_wav_async(result["wav"])
                pc_decoded = True
            if frames and frames[-1]["header"]["last"] and not pending_groups \
                    and not queue and not in_flight:
                if not args.quality_pass or quality_play_dumps >= last_chunk + 1:
                    if args.quality_pass or played >= expected_plays:
                        break
            if in_flight and in_flight_group is not None:
                if time.time() - in_flight_started > args.ack_timeout:
                    if ack_retries < args.ack_retries:
                        ack_retries += 1
                        retry_line = (f"[WARN] chunk ack timeout; retry "
                                      f"{ack_retries}/{args.ack_retries}")
                        print(retry_line, file=sys.stderr)
                        publish_ui_line(retry_line)
                        send_group(in_flight_group, retry=True)
                    else:
                        print("[WARN] receiver did not acknowledge chunk after retries",
                              file=sys.stderr)
                        break
            time.sleep(0.01)
        else:
            print(f"[WARN] relay timeout after {args.timeout:.0f}s: "
                  f"fragments={len(frames)} relayed={sent_groups} played={played}",
                  file=sys.stderr)

        print(f"=== DONE elapsed={time.time() - run_t0:.3f}s "
              f"fragments={len(frames)} groups={sent_groups} "
              f"played={played} ===")
        if pc_stream is not None:
            pc_stream.close()
        if pc_play_thread is not None:
            pc_play_thread.join(timeout=30.0)
        capture.finish(async_quality=args.async_quality)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as handle:
                for frame in frames:
                    json.dump(frame, handle, ensure_ascii=True)
                    handle.write("\n")
            print(f"saved {args.save}")
        return 0 if frames else 3
    finally:
        tx.close()
        rx.close()


if __name__ == "__main__":
    raise SystemExit(main())
