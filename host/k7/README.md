# K7 串口验证工具

需要 Python3、`requirements.txt` 中的 pyserial/Pillow；串口1500000、8N1，当前支持实测的CH340适配器。RX/TX交叉，GND对接，VCC不接。

在本目录设置 `K7_UART_PORT`（未设置时默认COM15），确认接的是目标板、双方已供电入网后执行：

```powershell
$env:K7_UART_PORT = 'COM15'
python display_probe.py
python check_navigation.py --board master
python test_ui_call.py --board master --samples 10
```

从板使用 `--board slave`。屏幕探针为只读检查；页面导航和通话测试会注入真实LVGL指针事件，通话测试会实际开始/结束LoRa通话。期间不要人工触屏。一次只能有一个工具占用串口。

测量结果写入本目录已忽略的 `artifacts/`、`deployment/`。显示控制器状态值3只是必要条件，不能代替真实屏幕出图和触摸验收。序号缺口、LoRa RSSI/SNR、停止后重新入网都来自实际板端数据；这些工具不调用SSH/IP控制对端。
