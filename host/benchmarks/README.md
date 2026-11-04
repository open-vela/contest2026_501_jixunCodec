# ESP32 测评数据离线复算

`rebuild_esp32_comparison.py` 从**已由 ESP32 产生**的日志和 PCM 重新计算延迟统计、音质评分及报告。它不执行 Codec，不刷机，也不采集新数据。

输入为完整原始测评数据集（本地默认位于忽略目录 `deliverables/esp32_metrics_comparison_20261104_035832/`）。数据目录必须包含 `evidence/`、`baseline_amr/results/` 等原始结构。

```bash
python3 -m pip install -r host/benchmarks/requirements.txt
python3 host/benchmarks/rebuild_esp32_comparison.py \
  --dataset /path/to/copy-of-original-dataset
```

**会在传入数据目录重新写入报告与 quality 输出，请先复制一份。** 原数据集可能包含原 Windows Python 的 `python_deps/`；其他平台应使用适配自身平台的隔离环境和本目录 requirements，避免混用二进制模块。

采用固定对齐：AI 16 kHz 输出前移 4118 采样，AMR 8 kHz 输出前移 40 采样；8 kHz 窄带同参考评分，不对每段重新寻找最优对齐或拟合增益。宽带补充指标及局限见公开测评报告。环境依赖版本和来源固件哈希保留于原数据集，不把 PC 评分耗时当板端编解码耗时。
