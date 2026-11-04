# K7 更新准备记录

本次仅准备本地代码与暂存区。没有创建项目提交，没有推送、PR 或 Release，也没有刷写板卡。

## 基线与范围

官方 `dev-ai-contest-2026` 基线：`c6c87266eb4112732cc550fa7c1277d651200c29`，源码树 `f719c2e0a7d94e596b26d5c8f35693843de297d2`。

新增 `app/jixun-k7/`、`board/kickpi-k7/`、`host/k7/`、K7 构建/预检查工具和可选 K7 manifest。原 ESP32 应用、板级补丁、主 manifest、官方基础 manifest、已有 AI 日志和 CLA 工作流没有修改。固件与 ELF 仅留在忽略的 `deliverables/`。

业务语句和权重数组与已验收 K7 基线一致；生成字库注释中的本机路径已清理，文本行尾统一 LF。来源摘要、整理后摘要及变换列表在 `app/jixun-k7/SOURCE_PROVENANCE.json`。第三方原有空白和生成字库末尾格式按限定的属性规则保留，不对本项目编写的源码放宽空白检查。

## 已完成验证

- 固定干净上游基线上的 NuttX/BSP/apps/依赖补丁预检查与应用。
- 主、从角色分别编译；Git规范化后的配置和源码再次构建通过。
- 重复准备两次，确认不会重复应用 libuv 补丁或改写固定平台源码。
- 向隔离验证目录的文件注入未知源码变更，预检查在其他文件写入前拒绝；随后恢复。
- 原 SDK 的业务/平台源码、配置、启动脚本摘要保持不变。
- 配置值与实装基线相比仅 `CONFIG_BASE_DEFCONFIG` 的目标目录名不同。
- 源码摘要、目标 manifest、临时产物排除、路径/凭据扫描和 `git diff --cached --check`。

原始上板数据见 [VALIDATION.md](VALIDATION.md)，本次构建记录见 [build-validation.json](build-validation.json)。本次新编译产物没有上板，不用它替代历史实装版本的验证结论。

在项目根目录可以复查：

```bash
python3 scripts/check_k7_submission.py
git diff --cached --check
git diff --cached --stat
sha256sum -c SHA256SUMS.k7.txt
```

## 正式上传前仍需完成

- 按官方采集工具导出本次 K7 的真实 AI Coding 日志，进行脱敏和 `validate-log.py` 校验。保留的历史日志不代表新阶段已完成。
- 提交人确认报名账号、协作邀请、CLA、真实 `Signed-off-by` 和最终变更。
- 按既有清单检查本地技术报告、视频、照片与最终暂存区，再按团队流程创建提交和 PR。

当前偶发栈标记改写、USB枚举、原生网络/实体键等限制明确保留在验证文档，不宣称全部解决。
