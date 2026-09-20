# 上传前人工检查清单

## 已完成

- [x] 已拉取官方 `dev-ai-contest-2026` 分支。
- [x] 应用源码已放入 `app/jixun-codec/`。
- [x] `plant-companion` 音频依赖已收敛并随应用携带，不依赖官方分支中不存在的目录。
- [x] 板级补丁已基于当前比赛分支 NuttX 基线重新生成，并通过 `git apply --check`。
- [x] 当前打包结构已用 3K 配置完成一次 clean build，错误和未解析符号均为 0。
- [x] 1K、3K、6K 固件已归位；技术报告、视频和照片等资料保留在本地 `deliverables/`。
- [x] `deliverables/` 已加入 `.gitignore` 并移出 Git 索引，不会被推送。
- [x] 已扫描并清除真实 API Key、绝对路径和临时构建产物。
- [x] 未执行 `git push`、PR、release 或任何远端写操作。

## 上传前必须人工确认

- [ ] 在完整 openvela 工作区内运行官方 AI Coding 日志采集工具，导出真实 JSONL 到 `logs/`。
- [ ] 运行 `validate-log.py logs/`，确认日志序号和格式有效。
- [ ] 确认 GitHub 登录账号是报名账号，并已接受仓库协作邀请。
- [ ] 确认 CLA 已在 openvela 官网签署。
- [ ] 人工检查本地 `deliverables/` 中的演示视频、技术报告和照片无隐私或错误内容。
- [ ] 人工 review `git diff --cached` 后再提交并推送。

## 建议提交顺序

1. 先提交代码、板级补丁、文档和交付物。
2. 再运行日志导出，单独提交 `logs/`。
3. 推送分支后创建 PR，自行 review 并通过 CLA 检查后再合入。
