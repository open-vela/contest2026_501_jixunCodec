@echo off
cd /d "%~dp0"
python -u "_pc_token_relay.py" ^
  --tx-port COM23 --rx-port COM4 --baud 115200 ^
  --rate 3k ^
  --seconds 2 --chunk-ms 500 --buffer-play ^
  --live-tx ^
  --play-gain 4 --no-pcm-dump --beep ^
  --inject-wav ".\_demo_assets\ai_token_demo_cn.wav" ^
  --timeout 240
