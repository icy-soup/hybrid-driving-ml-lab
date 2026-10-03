# 项目结构迁移记录

迁移日期：2026-10-03

## 迁移结果

- 当前主代码：`src/driving_lab`
- 统一入口：`run.py`
- 测试目录：`tests`
- 实验产物：`experiments/data`、`experiments/models`、`experiments/results`、`experiments/logs`
- 图形资源：`assets/pygame`
- 历史实现：`legacy/pygame_rule`、`legacy/pre_refactor`
- 设计文档：`docs/design`
- 结项报告：`docs/reports/结项报告.doc`

## 外部历史备份

```text
F:\Missions And Materials\神经与规则混合型智驾项目
```

该目录保留重构前的历史版本和参考资料，本项目不删除、不覆盖它。

## 验收记录

迁移后已确认：

- 根目录 `run.py --help` 能显示 `collect`、`train`、`evaluate`、`baseline`；
- `python -m pytest -q`：34 passed；
- `python -m compileall -q src run.py legacy/pygame_rule`：通过；
- collect → train → evaluate → baseline 已在 `experiments/` 临时验证目录完成，验证产物已清理；
- baseline JSON 顶层键为 `rule`、`neural`、`hybrid`；
- 结项报告只完成位置迁移，正文未修改。

迁移过程中曾因 PowerShell 目标目录预创建产生临时的 `src/driving_lab/driving_lab` 嵌套目录，已立即修正；最终源码包路径为 `src/driving_lab`。

## 最终自检

按结构设计逐项检查了入口、主代码导入、旧 GUI 资源路径、文档旧路径、外部备份和报告文件；未发现会阻断当前基线运行的 Critical 或 Important 问题。
