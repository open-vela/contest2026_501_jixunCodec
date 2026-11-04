# AiBox 3k 收发应用

目标：ESP32-S3 AiBox、ST7789 240×240 非触摸屏、ES8311 音频和 SX1262 LoRa。UI 以实体 BOOT 操作，无触摸要求。

## 源码组织

- `common/`：2026-11-04 当前 TX 实测版本的公共源码，包含 3k INT8 模型、PIE 内核、音频链路、LoRa、简约 UI。
- `roles/rx/`：7 个同路径覆盖文件，生成当前 RX 版本。不要直接把 overlay 当完整应用编译。
- `source_manifest.json`：逐文件原始与整理后 SHA256、角色差异及允许的整理变换。
- `tests/`：主机 C UI 检查与预览源码，覆盖状态分类和屏幕边界，不等于板上显示验收。

`python3 scripts/materialize_aibox.py --role tx|rx --output <新建仓外目录>` 从仓库根目录运行。工具先核对清单中的全部准备源码哈希，再复制公共文件并应用接收覆盖；输出中 `port/jx_simple_role.h` 的 TX=0、RX=1。

整理只调整构建应用名 `jixun-doc` 为 `jixun-codec`，限制此配置为已测 3k，移除未使用的 1k/6k 权重副本，将一个字符常量中的实际 NUL 改写为等价 `\0`，并清理一处空行尾空格、将源码换行统一为 LF 以保证 Git 检出后的哈希稳定。模型参数、编解码内核、按键状态机和提示音时序不因目录整理而修改。历史三档应用仍完整保留在 `app/jixun-codec/`。

模型来源核对基准为用户指定的 `sqcodec-light-A-c-i8/sqcodec-light-A-c-i8` 工程。当前编码首块沿用此前优化的 32 通道（原导出为 80），其前后适配权重形状已有差异；解码侧没有做这类裁剪。139 个解码张量、146941 个 float32 反量化存储值逐位一致；该核对不等同于 PIE 内核运行输出等价。见 [核对记录](../../docs/AiBox模型核对_20261104.json)。源码中残留历史诊断功能、过时注释与硬件调试注释；实际参数以角色源码和调用路径为准。

## 使用

发送板准备完成后，短按一次 BOOT，等待两声开始短音结束再说话，约 3 秒自动结束并播放一声低音。第二次按键可请求结束；编码、发送和完成展示期间不是新一轮录音入口。当前完成展示占用约 800 ms，偶发 BOOT 无反应尚未定位为单一原因。

接收板开机监听，接收、解码、播放依次显示。环境中其他同频设备可能干扰接收，应同时记录当时环境。串口、PC 控制入口与双路 PCM 导出仅供调试，导出组合存在未解决的阻塞轮次。

串口命令 `jixun ui send` 可走同一发送路径；实体按键仍需独立实测。角色、硬件接线、构建限制见 [板级说明](../../board/esp32s3-aibox/README.md)，真实固件与结果边界见 [当前版本记录](../../docs/AiBox当前版本与限制_20261104.md)。

## UI 检查

```bash
gcc -std=c99 -Wall -Wextra -Werror -O2 -Iapp/jixun-aibox/tests \
  app/jixun-aibox/tests/native_preview.c -o /tmp/jixun-ui-check
/tmp/jixun-ui-check 0 RECORDING 0 /tmp/jixun-tx.ppm
/tmp/jixun-ui-check 1 PLAYING 0 /tmp/jixun-rx.ppm
```

每次执行检查 17 项状态分类断言及写屏边界。输出仅为主机预览，不替代 LCD、麦克风和扬声器板测。

模型复核（需原始导出，不随本仓重复打包）：

```bash
python3 scripts/verify_aibox_model.py --reference /path/to/weights_data_3k.h --scope decoder
```

默认检查解码存储值，并列出编码差异。`--scope all` 会因既有编码裁剪而拒绝全模型一致的断言。

编译文件以 Makefile 的 `CSRCS` / `MAINSRC` 为准：实际入口是 `main/jx_main.c`，`port/jx_main.c` 和 `port/hal_i2s.c` 是保留的历史拷贝、不参加此构建；底层 I2S 使用参考板级补丁。权重由 `weights/3k/` 选择，根 weights 下保留的历史副本不作为本入口的选档依据。
