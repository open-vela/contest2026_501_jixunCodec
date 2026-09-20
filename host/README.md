# PC Token broker and test console

The PC is a Token relay and observer only. It never decodes or plays PCM.

## Files

- `_pc_safe_demo_ui.py`: rate selector, live token view and playback status.
- `_pc_token_relay.py`: COM23 -> PC -> COM4 token relay, CRC, fragment accounting and timeout retry.
- `_switch_codec_rate.py`: safely writes the selected application image to both boards at `0x10000`.
- `_rate_stability_smoke.py`: repeated end-to-end pass/fail gate.
- `_run_three_rate_demo.py`: automated 1K -> 3K -> 6K demonstration sequence.
- `_rate_cycle_50.py`: 50-round benchmark and fingerprint consistency runner.
- `firmware/`: r276 application images for the three rates.
- `_demo_assets/ai_token_demo_cn.wav`: fixed two-second reference input.

## Run on Windows

```bat
python -m pip install -r requirements.txt
SwitchCodec_3k.cmd
```

`SwitchCodec_1k.cmd`, `SwitchCodec_3k.cmd` and `SwitchCodec_6k.cmd` select the model and launch the console. Only the application area at `0x10000` is written.

The two-second WAV is a deterministic test fixture, not a transmission limit. Audio is processed continuously in 500 ms chunks; the board UI reports measured `tokens/s`.
