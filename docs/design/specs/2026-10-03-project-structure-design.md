# 项目结构重构设计

状态：已批准并执行（2026-10-03）

## 1. 目标

将当前由学校项目历史目录演化而来的 `auto/1_rule`、`auto/2_ai` 结构，整理为按职责划分的项目结构，使代码、实验产物、历史实现、展示资源和结项材料彼此分离，便于：

- 继续运行当前无界面基线；
- 后续加入 GPU 强化学习而不再改变顶层目录；
- 在简历、答辩和代码托管平台中清楚解释项目组成；
- 最后统一编写和更新结项报告。

## 2. 当前状态与边界

迁移前主版本位于 `auto/2_ai`，核心包是 `driving_lab`，包含环境、物理计算、规则策略、行为克隆模型、奖励、场景采样、批量环境和评测接口。`auto/1_rule` 是旧版 Pygame 图形化规则方案；迁移后主代码位于 `src/driving_lab`，旧版位于 `legacy/pygame_rule`。

历史备份已经移出当前项目，位于：

```text
F:\Missions And Materials\神经与规则混合型智驾项目
```

当前项目根目录曾有根级 `images/`、`结项报告.doc` 和 `docs/superpowers/`；迁移目标是将它们分别归入 `assets/pygame`、`docs/reports` 和 `docs/design`。结项报告只移动位置，不在本次结构重构中修改内容。

## 3. 目标结构

```text
神经与规则混合型智驾项目/
├── README.md
├── requirements.txt
├── run.py                         # 当前实验统一入口
├── src/
│   └── driving_lab/               # 当前主研究代码
│       ├── config.py
│       ├── types.py
│       ├── physics.py
│       ├── environment.py
│       ├── policies.py
│       ├── dataset.py
│       ├── model.py
│       ├── rewards.py
│       ├── scenarios.py
│       ├── vector_env.py
│       ├── evaluation.py
│       └── cli.py
├── tests/                         # 当前主版本的行为和数据契约测试
├── scripts/                       # 可选的辅助脚本，不承载核心逻辑
├── experiments/
│   ├── data/                      # NPZ 等训练数据
│   ├── models/                    # PKL、后续 PyTorch checkpoint
│   ├── results/                   # JSON、CSV 和汇总结果
│   └── logs/                      # 训练日志和运行记录
├── assets/
│   └── pygame/                    # 图形界面展示资源
├── legacy/
│   ├── pygame_rule/               # 原 auto/1_rule
│   └── pre_refactor/              # 原 auto/2_ai/legacy 等旧实现
└── docs/
    ├── design/                    # 对外可读的设计和技术方案
    ├── notes/                     # 迁移记录、验收记录和实验说明
    └── reports/
        └── 结项报告.doc            # 本次只移动，最后阶段再修改
```

`docs/superpowers/` 中已有的设计稿和实施计划，在迁移阶段整理到 `docs/design/` 下的对应位置；文件内容中的路径引用同步更新。

## 4. 迁移映射

| 当前路径 | 目标路径 | 处理方式 |
|---|---|---|
| `auto/2_ai/driving_lab/` | `src/driving_lab/` | 移动当前主代码 |
| `auto/2_ai/tests/` | `tests/` | 移动测试并更新导入路径 |
| `auto/2_ai/Main.py` | `run.py` | 改为根目录薄入口 |
| `auto/2_ai/requirements.txt` | `requirements.txt` | 保留依赖入口 |
| `auto/1_rule/` | `legacy/pygame_rule/` | 作为历史 GUI 方案保存 |
| 根目录 `images/` | `assets/pygame/` | 归入展示资源，并修正旧 GUI 的资源定位 |
| `auto/2_ai/legacy/` | `legacy/pre_refactor/` | 归入旧版实现和旧模型数据 |
| `docs/superpowers/specs/` | `docs/design/specs/` | 保留设计文档并更新链接 |
| `docs/superpowers/plans/` | `docs/design/plans/` | 保留实施计划并更新链接 |
| 根目录 `结项报告.doc` | `docs/reports/结项报告.doc` | 只移动，不改内容 |

外部历史备份目录不再次复制回项目，也不删除。

## 5. 兼容性要求

迁移完成后必须保持：

1. 动作编号不变：`0=加速，1=制动，2=左变道，3=右变道，4=巡航`；
2. `HighwayEnv`、`RulePolicy`、`NeuralPolicy`、`SafetyShieldPolicy` 的接口不变；
3. `collect`、`train`、`evaluate`、`baseline` 四个命令语义不变；
4. NPZ 数据字段和模型保存格式不变；
5. 评测指标字段不变；
6. 根目录统一入口能够从项目根目录执行；
7. 旧版 Pygame 代码作为 `legacy` 保存，不影响主版本测试和运行。

## 6. 入口设计

新的推荐运行方式为：

```powershell
python run.py collect --episodes 20 --seed 7 --scenario mixed --output experiments/data/training_data.npz
python run.py train --data experiments/data/training_data.npz --model experiments/models/neural_model.pkl --epochs 30 --seed 7
python run.py evaluate --policy hybrid --model experiments/models/neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments/results/hybrid.json
python run.py baseline --model experiments/models/neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments/results/baseline.json
```

核心实现仍然位于 `src/driving_lab`，`run.py` 只负责把源码目录加入运行路径并转发到现有 CLI，不把实验逻辑塞进入口脚本。

## 7. 验证策略

迁移阶段分三次验证：

1. 迁移前记录当前 `33 passed` 的测试基线；
2. 迁移后运行 `compileall`、完整 pytest 和四条 CLI 命令；
3. 使用固定 seed 运行一次 collect → train → baseline，比较迁移前后的 JSON 指标字段和动作编号。

如果旧版 Pygame 资源无法在当前环境启动，只做静态路径检查，不把它混入主版本验收。

## 8. 非目标

- 本次不实现强化学习；
- 本次不修改奖励权重和环境语义；
- 本次不重写结项报告正文；
- 本次不删除外部历史备份；
- 本次不把旧版 GUI 伪装成当前研究主线。
