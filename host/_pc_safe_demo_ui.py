#!/usr/bin/env python3
"""Desktop showcase for the dual-USB AI token voice relay."""

import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

try:
    import winsound
except ImportError:
    winsound = None


ROOT = Path(__file__).resolve().parent
RELAY = ROOT / "_pc_token_relay.py"
RESULT_DIR = ROOT / "_pc_voice_rx"
UI_EVENT_FILE = RESULT_DIR / "ui_events.jsonl"
DEFAULT_WAV = ROOT / "_demo_assets" / "ai_token_demo_cn.wav"
PACKAGED_WAV = ROOT / "assets" / "ai_token_demo_cn.wav"
FALLBACK_WAV = ROOT / "_tmp" / "pc_test_voice_cn3.wav"

COLORS = {
    "bg": "#141414",
    "panel": "#1f1f1f",
    "panel2": "#282828",
    "line": "#3a3a3a",
    "text": "#f3f3f3",
    "muted": "#9b9b9b",
    "teal": "#2dd4bf",
    "amber": "#f59e0b",
    "red": "#ef4444",
    "green": "#22c55e",
}


class SafeDemoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("极讯 AI Codec - SAFE DEMO")
        self.root.geometry("1180x760")
        self.root.minsize(980, 660)
        self.root.configure(bg=COLORS["bg"])
        self.record_layout = "--record-layout" in sys.argv
        if self.record_layout:
            self.root.geometry("1180x760+0+0")
            self.root.attributes("-topmost", True)
            self.root.lift()

        self.events = queue.Queue()
        self.proc = None
        self.owns_relay = False
        self.external_offset = 0
        self.reader = None
        self.running = False
        self.started_at = 0.0
        try:
            self.external_offset = UI_EVENT_FILE.stat().st_size
        except OSError:
            self.external_offset = 0
        self.token_fragments = 0
        self.token_total = 0
        self.token_sum = 0
        self.token_bytes = 0
        self.relayed_groups = 0
        self.played = 0
        self.payload_text = "--"
        self.first_sound_text = "--"
        self.first_sound_at = None
        if DEFAULT_WAV.exists():
            self.wav_path = DEFAULT_WAV
        elif PACKAGED_WAV.exists():
            self.wav_path = PACKAGED_WAV
        else:
            self.wav_path = FALLBACK_WAV

        self.tx_port = tk.StringVar(value="COM23")
        self.rx_port = tk.StringVar(value="COM4")
        self.mode = tk.StringVar(value="quality")
        self.source_mode = tk.StringVar(value="auto")
        self.rate = tk.StringVar(value="3k")
        self.status = tk.StringVar(value="READY")
        self.status_detail = tk.StringVar(value="等待开始")
        self.elapsed = tk.StringVar(value="0.0 s")
        self.stats = tk.StringVar(value="Token 分片 0  |  转发组 0  |  播放 0")
        self.token_preview = tk.StringVar(value="等待 token 数据...")
        self.metrics_primary = tk.StringVar(
            value="模型档位 3 kbps  |  Token 速率 -- tokens/s  |  累计 --  |  当前状态 READY"
        )
        self.metrics_secondary = tk.StringVar(
            value="首声 -- s  |  转发组 0  |  播放 --"
        )
        self.auto_restart = tk.BooleanVar(value=False)
        self.trigger_file = None
        self.last_trigger = ""
        self.current_rate = ""
        self.selected_rate = "3k"
        self.current_tokens_per_second = 0.0
        self.token_audio_seconds = 0.0
        self.flashing = False
        self.rate_buttons = []
        for i, arg in enumerate(sys.argv):
            if arg == "--trigger-file" and i + 1 < len(sys.argv):
                self.trigger_file = Path(sys.argv[i + 1])
            if arg == "--rate" and i + 1 < len(sys.argv) and sys.argv[i + 1] in {"1k", "3k", "6k"}:
                self.rate.set(sys.argv[i + 1])
                self.selected_rate = sys.argv[i + 1]
            if arg == "--source" and i + 1 < len(sys.argv) and sys.argv[i + 1] in {"auto", "live"}:
                self.source_mode.set(sys.argv[i + 1])
        if self.trigger_file is not None:
            self.root.attributes("-topmost", True)
            self.root.lift()
            self.root.after(1600, lambda: self.root.attributes("-topmost", False))

        self._configure_style()
        self._build_ui()
        self._draw_waveform()
        self.root.after(50, self._poll_events)
        self.root.after(100, self._tick)
        self.root.after(250, self._poll_external_events)
        if self.trigger_file is not None:
            self.root.after(300, self._poll_trigger)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
        style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Sub.TLabel", background=COLORS["bg"], foreground=COLORS["muted"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 13, "bold"))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"),
                        padding=(18, 10), background=COLORS["teal"], foreground="#0b1715")
        style.map("Accent.TButton",
                  background=[("active", "#5eead4"), ("disabled", "#35514d")],
                  foreground=[("disabled", "#8a9b98")])
        style.configure("Stop.TButton", font=("Microsoft YaHei UI", 11, "bold"),
                        padding=(18, 10), background=COLORS["red"], foreground="#ffffff")
        style.map("Stop.TButton",
                  background=[("active", "#f87171"), ("disabled", "#5b3333")])
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 8))
        style.configure("TRadiobutton", background=COLORS["panel"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("TCheckbutton", background=COLORS["panel"], foreground=COLORS["text"],
                        font=("Microsoft YaHei UI", 10))
        style.configure("TEntry", fieldbackground="#111111", foreground=COLORS["text"],
                        insertcolor=COLORS["text"], padding=6)
        style.configure("Horizontal.TProgressbar", background=COLORS["teal"],
                        troughcolor=COLORS["panel2"], bordercolor=COLORS["panel2"],
                        lightcolor=COLORS["teal"], darkcolor=COLORS["teal"])

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=22)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x")
        left = ttk.Frame(header)
        left.pack(side="left", fill="x", expand=True)
        ttk.Label(left, text="极讯 AI Codec", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="双 USB + PC 中转  |  只传 AI Token，不传音频 PCM",
                  style="Sub.TLabel").pack(anchor="w", pady=(4, 0))
        self.status_badge = tk.Label(header, textvariable=self.status, bg=COLORS["panel2"],
                                     fg=COLORS["teal"], font=("Consolas", 12, "bold"),
                                     padx=18, pady=10)
        self.status_badge.pack(side="right", padx=(20, 0))

        pipeline = ttk.Frame(outer, style="Panel.TFrame", padding=18)
        pipeline.pack(fill="x", pady=(20, 14))

        self.pipeline = tk.Canvas(pipeline, height=180, bg=COLORS["panel"],
                                  highlightthickness=0)
        self.pipeline.pack(fill="x", expand=True)
        self.pipeline.bind("<Configure>", lambda _event: self._draw_pipeline())

        controls = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        controls.pack(fill="x", pady=(0, 14))

        row = ttk.Frame(controls, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="发送端", style="Muted.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.tx_port, width=9).pack(side="left", padx=(8, 18))
        ttk.Label(row, text="接收端", style="Muted.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.rx_port, width=9).pack(side="left", padx=(8, 22))

        ttk.Label(row, text="链路模式", style="Muted.TLabel").pack(side="left")
        ttk.Radiobutton(row, text="质量优先", variable=self.mode,
                        value="quality").pack(side="left", padx=(8, 4))
        ttk.Radiobutton(row, text="低延迟实验", variable=self.mode,
                        value="realtime").pack(side="left", padx=(4, 20))

        ttk.Label(row, text="测试音", style="Muted.TLabel").pack(side="left")
        ttk.Radiobutton(row, text="自动中文", variable=self.source_mode,
                        value="auto").pack(side="left", padx=(8, 4))
        ttk.Radiobutton(row, text="真人说话", variable=self.source_mode,
                        value="live").pack(side="left", padx=(4, 0))

        rate_row = ttk.Frame(controls, style="Panel.TFrame")
        rate_row.pack(fill="x", pady=(10, 0))
        ttk.Label(rate_row, text="模型档位", style="Muted.TLabel").pack(side="left")
        for value, label in (("1k", "1 kbps"), ("3k", "3 kbps"), ("6k", "6 kbps")):
            button = ttk.Radiobutton(rate_row, text=label, variable=self.rate, value=value)
            button.pack(side="left", padx=(10, 0))
            self.rate_buttons.append(button)
        ttk.Label(rate_row, text="由 PC 控制台切换并启动测试", style="Muted.TLabel").pack(
            side="right"
        )

        status_row = ttk.Frame(controls, style="Panel.TFrame")
        status_row.pack(fill="x", pady=(14, 8))
        ttk.Label(status_row, textvariable=self.status_detail,
                  style="Section.TLabel").pack(side="left")
        ttk.Label(status_row, textvariable=self.elapsed,
                  style="Muted.TLabel").pack(side="right")

        metrics = ttk.Frame(controls, style="Panel.TFrame")
        metrics.pack(fill="x", pady=(0, 10))
        ttk.Label(metrics, textvariable=self.metrics_primary,
                  style="Section.TLabel").pack(anchor="w")
        ttk.Label(metrics, textvariable=self.metrics_secondary,
                  style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

        self.progress = ttk.Progressbar(controls, mode="determinate", maximum=100,
                                        style="Horizontal.TProgressbar")
        self.progress.pack(fill="x")

        actions = ttk.Frame(controls, style="Panel.TFrame")
        actions.pack(fill="x", pady=(14, 0))
        self.start_btn = ttk.Button(actions, text="开始演示", style="Accent.TButton",
                                    command=self.start_demo)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(actions, text="停止", style="Stop.TButton",
                                   command=self.stop_demo, state="disabled")
        self.stop_btn.pack(side="left", padx=(10, 0))
        ttk.Button(actions, text="打开结果目录", command=self.open_results).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(actions, text="结束后自动重复", variable=self.auto_restart).pack(side="right")

        lower = ttk.Frame(outer)
        lower.pack(fill="both", expand=True)
        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=2)
        lower.rowconfigure(0, weight=1)

        token_panel = ttk.Frame(lower, style="Panel.TFrame", padding=16)
        token_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        ttk.Label(token_panel, text="AI Token Stream", style="Section.TLabel").pack(anchor="w")
        ttk.Label(token_panel, textvariable=self.stats, style="Muted.TLabel").pack(anchor="w", pady=(3, 10))
        self.token_canvas = tk.Canvas(token_panel, height=180, bg=COLORS["panel"],
                                      highlightthickness=0)
        self.token_canvas.pack(fill="both", expand=True)
        self.token_canvas.bind("<Configure>", lambda _event: self._draw_tokens())
        ttk.Label(token_panel, textvariable=self.token_preview, style="Muted.TLabel",
                  font=("Consolas", 10)).pack(anchor="w", pady=(10, 0))

        log_panel = ttk.Frame(lower, style="Panel.TFrame", padding=16)
        log_panel.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        ttk.Label(log_panel, text="Live Events", style="Section.TLabel").pack(anchor="w")
        self.log = tk.Text(log_panel, bg="#111111", fg=COLORS["text"], bd=0,
                           insertbackground=COLORS["text"], wrap="word",
                           font=("Consolas", 9), padx=10, pady=10)
        self.log.pack(fill="both", expand=True, pady=(10, 0))
        self.log.configure(state="disabled")

    def _draw_pipeline(self):
        c = self.pipeline
        c.delete("all")
        w = max(c.winfo_width(), 800)
        h = max(c.winfo_height(), 160)
        box_w, box_h = 190, 92
        y = (h - box_h) // 2
        xs = [34, (w - box_w) // 2, w - box_w - 34]
        for x, label, sub in zip(
                xs,
                ("COM23  采集与编码", "AI Token Stream", "COM4  解码与播放"),
                ("MIC -> Neural Codec", "PC 只中转 token 数据", "Token -> Neural Decoder")):
            c.create_rectangle(x, y, x + box_w, y + box_h, fill=COLORS["panel2"],
                               outline=COLORS["line"], width=2)
            c.create_text(x + box_w / 2, y + 30, text=label, fill=COLORS["text"],
                          font=("Microsoft YaHei UI", 12, "bold"))
            c.create_text(x + box_w / 2, y + 60, text=sub, fill=COLORS["muted"],
                          font=("Microsoft YaHei UI", 9))
        for x0, x1 in ((xs[0] + box_w, xs[1]), (xs[1] + box_w, xs[2])):
            mid = (x0 + x1) / 2
            c.create_line(x0 + 8, y + box_h / 2, x1 - 8, y + box_h / 2,
                          fill=COLORS["amber"], width=3, arrow="last", arrowshape=(12, 14, 6))
            for k in range(4):
                px = mid + (k - 1.5) * 16
                c.create_oval(px - 4, y + box_h / 2 - 4, px + 4, y + box_h / 2 + 4,
                              fill=COLORS["amber"], outline="")

    def _draw_waveform(self):
        return

    def _draw_tokens(self):
        c = self.token_canvas
        c.delete("all")
        w = max(c.winfo_width(), 500)
        h = max(c.winfo_height(), 160)
        columns = 12
        rows = 5
        gap = 6
        cell_w = (w - gap * (columns + 1)) / columns
        cell_h = (h - gap * (rows + 1)) / rows
        done = min(self.token_fragments, columns * rows)
        for i in range(columns * rows):
            r, col = divmod(i, columns)
            x = gap + col * (cell_w + gap)
            y = gap + r * (cell_h + gap)
            fill = COLORS["amber"] if i < done else COLORS["panel2"]
            outline = COLORS["amber"] if i < done else COLORS["line"]
            c.create_rectangle(x, y, x + cell_w, y + cell_h, fill=fill,
                               outline=outline, width=1)

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, status, detail=None, color=None):
        self.status.set(status)
        if detail is not None:
            self.status_detail.set(detail)
        self.status_badge.configure(fg=color or COLORS["teal"])

    def start_demo(self):
        if self.running:
            return
        if self.flashing or (self.proc is not None and self.proc.poll() is None):
            self._append_log("[UI] start ignored: previous task is still running")
            return
        if not RELAY.exists():
            self._set_status("ERROR", "找不到中继脚本", COLORS["red"])
            return
        if self.source_mode.get() == "auto" and not self.wav_path.exists():
            self._set_status("ERROR", "找不到中文测试音", COLORS["red"])
            return

        self.running = True
        self.owns_relay = True
        self.selected_rate = self.rate.get()
        self.started_at = time.time()
        self.token_fragments = 0
        self.token_total = 0
        self.token_sum = 0
        self.token_bytes = 0
        self.relayed_groups = 0
        self.played = 0
        self.payload_text = "--"
        self.first_sound_text = "--"
        self.first_sound_at = None
        self.current_tokens_per_second = 0.0
        self.token_audio_seconds = 0.0
        self.token_preview.set("等待 token 数据...")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")
        for button in self.rate_buttons:
            button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        if self.source_mode.get() == "live":
            detail = f"正在准备 {self.selected_rate} 模型；完成后请在提示音后说话"
        else:
            detail = f"正在确认或切换 {self.selected_rate} 模型..."
        self._set_status("MODEL", detail, COLORS["amber"])
        self._append_log(
            f"[UI] start rate={self.selected_rate} source={self.source_mode.get()} "
            f"mode={self.mode.get()}"
        )
        self._draw_tokens()

        self.reader = threading.Thread(target=self._run_relay, daemon=True)
        self.reader.start()

    def _run_relay(self):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        flash_cmd = [
            sys.executable, "-u", str(ROOT / "_switch_codec_rate.py"),
            self.selected_rate, "--ensure", "--keep-ui",
        ]
        try:
            self.flashing = True
            self.proc = subprocess.Popen(
                flash_cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.events.put(("line", line.rstrip()))
            rc = self.proc.wait()
            self.flashing = False
            if rc != 0:
                self.events.put(("error", f"模型切换失败，退出码 {rc}"))
                return
            self.events.put(("flash_done", self.selected_rate))
        except Exception as exc:
            self.flashing = False
            self.events.put(("error", str(exc)))
            return

        chunk_ms = ("500" if self.selected_rate == "6k" or self.mode.get() == "quality"
                    else "100")
        cmd = [
            sys.executable, "-u", str(RELAY),
            "--tx-port", self.tx_port.get().strip(),
            "--rx-port", self.rx_port.get().strip(),
            "--rate", self.selected_rate,
            "--baud", "115200",
            "--seconds", "2",
            "--chunk-ms", chunk_ms,
            "--play-gain", "4",
            "--beep",
            "--no-pcm-dump",
            "--save", str(RESULT_DIR / "tokens_ui.jsonl"),
            "--timeout", "240",
        ]
        if self.source_mode.get() == "auto":
            cmd += ["--inject-wav", str(self.wav_path)]
        if self.selected_rate == "3k":
            cmd += ["--live-tx"]
        if self.selected_rate == "1k":
            # Legacy 1K firmware can intermittently stall in the final
            # dual-core decode. Keep the receiver single-core for stability.
            cmd += ["--rx-pf", "0"]
        if self.mode.get() == "quality":
            cmd += ["--buffer-play"]
        elif self.mode.get() == "realtime":
            cmd += ["--stream-play"]

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                self.events.put(("line", line.rstrip()))
            rc = self.proc.wait()
            self.events.put(("done", rc))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def stop_demo(self):
        if self.flashing:
            self._append_log("[UI] model switch is in progress; wait for it to finish")
            return
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            self._append_log("[UI] stop requested")
        self.running = False

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "line":
                    self._handle_line(payload)
                elif kind == "flash_done":
                    self.stop_btn.configure(state="normal")
                    if self.source_mode.get() == "live":
                        detail = f"{payload} 模型已就绪；提示音后请对 COM23 说话"
                    else:
                        detail = f"{payload} 模型已就绪，启动双板链路..."
                    self._set_status("RUNNING", detail, COLORS["amber"])
                elif kind == "done":
                    self._handle_done(payload)
                elif kind == "error":
                    self.running = False
                    self._finish_ui()
                    self._set_status("ERROR", payload, COLORS["red"])
        except queue.Empty:
            pass
        self.root.after(50, self._poll_events)

    def _poll_external_events(self):
        if self.proc is not None:
            self.root.after(250, self._poll_external_events)
            return
        try:
            if not UI_EVENT_FILE.exists():
                self.root.after(250, self._poll_external_events)
                return
            size = UI_EVENT_FILE.stat().st_size
            if size < self.external_offset:
                self.external_offset = 0
            if size > self.external_offset:
                with UI_EVENT_FILE.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(self.external_offset)
                    lines = handle.readlines()
                    self.external_offset = handle.tell()
                for raw in lines:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("[RUN]"):
                        self.running = True
                        run_rate = re.search(r"rate=([136]k)", line)
                        if run_rate:
                            self.selected_rate = run_rate.group(1)
                            self.rate.set(self.selected_rate)
                            self.current_rate = self.selected_rate
                        self.start_btn.configure(state="disabled")
                        self.stop_btn.configure(state="normal")
                        for button in self.rate_buttons:
                            button.configure(state="disabled")
                        self.started_at = time.time()
                        self.token_fragments = 0
                        self.token_total = 0
                        self.token_sum = 0
                        self.token_bytes = 0
                        self.relayed_groups = 0
                        self.played = 0
                        self.payload_text = "--"
                        self.first_sound_text = "--"
                        self.first_sound_at = None
                        self.current_tokens_per_second = 0.0
                        self.token_audio_seconds = 0.0
                        self._set_status("RUNNING", f"检测到 {self.selected_rate} Token 中继启动",
                                         COLORS["amber"])
                        self._draw_tokens()
                        continue
                    self._handle_line(line)
                    if "[PROGRESS]" in line and "played=1/1" in line:
                        self.running = False
                        self._finish_ui()
                        self._set_status("COMPLETE", "双板 token 语音闭环完成", COLORS["green"])
                        self._set_progress(100)
        except OSError:
            pass
        self.root.after(250, self._poll_external_events)

    def _handle_line(self, line):
        if any(key in line for key in ("[TOKEN]", "[RELAY]", "[RX]", "[PROGRESS]",
                                       "[INJECT]", "[PC-DECODE]", "[PC-STREAM]",
                                       "[FLASH]", "[RATE]", "[TIME]", "[WARN]")):
            self._append_log(line)

        rate_match = re.search(r"(?:码率|target=)\s*([136]k)", line)
        if rate_match:
            self.current_rate = rate_match.group(1)
            self.root.title(f"极讯 AI Codec - SAFE DEMO [{self.current_rate} kbps]")

        tps_match = re.search(r"tokens_per_second=([0-9.]+)", line)
        if tps_match:
            self.current_tokens_per_second = float(tps_match.group(1))

        samples_match = re.search(r"orig_samples=(\d+)", line)
        first_fragment = bool(re.search(r"frag=0/", line))
        if samples_match and first_fragment:
            self.token_audio_seconds += int(samples_match.group(1)) / 16000.0

        payload_match = re.search(r"载荷\s*:\s*(\d+)\s*字节\s*\(([0-9.]+)\s*kbps\)", line)
        if payload_match:
            self.payload_text = f"{payload_match.group(1)} B"

        first_match = re.search(r"第一块出声\D*([-0-9.]+)\s*s", line)
        if first_match:
            self.first_sound_text = first_match.group(1)

        if "[INJECT]" in line:
            self._set_status("RECORD", "PC 正在播放中文测试音到 COM23", COLORS["amber"])
            self._set_progress(8)
        elif "录音中" in line:
            self._set_status("RECORD", "COM23 正在采集语音", COLORS["amber"])
            self._set_progress(8)
        elif "[TOKEN]" in line:
            self.token_fragments += 1
            match = re.search(r"tokens=(\d+)", line)
            if match:
                tokens_now = int(match.group(1))
                self.token_total = max(self.token_total, tokens_now)
                if first_fragment:
                    self.token_sum += tokens_now
            bytes_match = re.search(r"bytes=(\d+)", line)
            if bytes_match:
                self.token_bytes += int(bytes_match.group(1))
                self.payload_text = f"{self.token_bytes} B"
            preview = re.search(r"first=\[(.*?)\]", line)
            if preview:
                self.token_preview.set("Token Preview: " + preview.group(1)[:120])
            self._set_status("TOKENIZE", "神经编解码器正在产生 AI token", COLORS["amber"])
            self._set_progress(24)
            self._draw_tokens()
        elif "[RELAY]" in line:
            self.relayed_groups += 1
            self._set_status("TRANSFER", "PC 正向 COM4 转发 token 分片", COLORS["teal"])
            self._set_progress(48)
        elif "[usb] received chunk=" in line:
            self._set_status("RECEIVE", "COM4 正在重组 token 数据", COLORS["teal"])
            self._set_progress(66)
        elif "播放块" in line:
            if self.first_sound_at is None:
                self.first_sound_at = time.time()
                self.first_sound_text = f"{self.first_sound_at - self.started_at:.2f}"
            self._set_status("PLAY", "COM4 正在播放解码音频", COLORS["green"])
            self._set_progress(88)
        elif "[PROGRESS]" in line:
            match = re.search(r"played=(\d+)/(\d+)", line)
            if match:
                self.played = int(match.group(1))
        elif "[PC-STREAM]" in line:
            if "playback started" in line:
                self._set_status("PLAY", "PC 正在连续播放 AI token 解码音频", COLORS["green"])
                self._set_progress(92)
            elif "decoded" in line:
                self._set_status("PC DECODE", "PC 正在流式解码 token", COLORS["teal"])
                self._set_progress(82)
            elif "underrun" in line:
                self._set_status("WARN", "PC 播放缓冲不足", COLORS["amber"])
        elif "[PC-DECODE]" in line:
            self._set_status("PC DECODE", "PC 正在整段解码 token", COLORS["teal"])

        self.stats.set(
            f"Token 分片 {self.token_fragments}  |  "
            f"累计 {self.token_sum or '-'} token  |  单包 {self.token_total or '-'}  |  "
            f"转发组 {self.relayed_groups}  |  播放 {self.played}"
        )
        self._refresh_metrics()

    def _refresh_metrics(self):
        aggregate_tps = (self.token_sum / self.token_audio_seconds
                         if self.token_audio_seconds > 0 else self.current_tokens_per_second)
        tps_text = f"{aggregate_tps:.1f}" if aggregate_tps > 0 else "--"
        self.metrics_primary.set(
            f"模型档位 {self.selected_rate} kbps  |  "
            f"Token 速率 {tps_text} tokens/s  |  累计 {self.token_sum or '--'}  |  "
            f"状态 {self.status.get()}"
        )
        self.metrics_secondary.set(
            f"首声 {self.first_sound_text} s  |  "
            f"载荷 {self.payload_text}  |  转发组 {self.relayed_groups}  |  播放 {self.played}"
        )

    def _handle_done(self, rc):
        self.running = False
        self.owns_relay = False
        try:
            self.external_offset = UI_EVENT_FILE.stat().st_size
        except OSError:
            self.external_offset = 0
        self._finish_ui()
        if rc == 0:
            self._set_status("COMPLETE", "双板 token 语音闭环完成", COLORS["green"])
            self._set_progress(100)
            if self.auto_restart.get():
                self.root.after(3500, self._restart_if_idle)
        else:
            self._set_status("ERROR", f"中继退出码 {rc}", COLORS["red"])
        if self.trigger_file is not None:
            done = self.trigger_file.with_suffix(".done.json")
            done.parent.mkdir(parents=True, exist_ok=True)
            done.write_text(
                __import__("json").dumps({
                    "rate": self.selected_rate,
                    "rc": rc,
                    "played": self.played,
                    "token_fragments": self.token_fragments,
                    "relayed_groups": self.relayed_groups,
                    "finished_at": time.time(),
                }, ensure_ascii=False),
                encoding="utf-8",
            )

    def _restart_if_idle(self):
        if not self.running:
            self.start_demo()

    def _finish_ui(self):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        for button in self.rate_buttons:
            button.configure(state="normal")
        self.proc = None
        self.elapsed.set(f"{time.time() - self.started_at:.1f} s")

    def _set_progress(self, value):
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = value

    def _tick(self):
        if self.running:
            self.elapsed.set(f"{time.time() - self.started_at:.1f} s")
        self.root.after(100, self._tick)

    def _poll_trigger(self):
        try:
            text = self.trigger_file.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        requested = text.splitlines()[0] if text else ""
        if (requested in {"1k", "3k", "6k"} and text != self.last_trigger
                and not self.running):
            self.last_trigger = text
            self.selected_rate = requested
            self.rate.set(requested)
            self.current_rate = requested
            self.root.title(f"极讯 AI Codec - SAFE DEMO [{self.current_rate} kbps]")
            self._set_status("READY", f"收到 {self.current_rate} kbps 演示请求", COLORS["teal"])
            self.start_demo()
        if self.trigger_file is not None:
            self.root.after(400, self._poll_trigger)

    def open_results(self):
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(RESULT_DIR))

    def on_close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    ui = SafeDemoUI(root)
    if "--autostart" in sys.argv:
        root.after(900, ui.start_demo)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
