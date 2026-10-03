# 物理信息增强的数据驱动型自动驾驶算法研究项目（SITP）

这是一个以多车高速公路自动驾驶为背景的学校项目。当前主线是可复现的规则约束行为克隆基线：用物理/安全先验生成示范数据，训练神经策略，并比较规则策略、纯神经策略和带安全屏障的混合策略。

项目题目和自动驾驶背景保持不变；代码结构按职责划分，便于后续加入 GPU 强化学习，也便于在简历和答辩中说明各模块的通用 AI/ML 方法。

## 项目结构

```text
src/driving_lab/       当前主研究代码：环境、策略、数据、模型、评测
tests/                 主版本行为与数据契约测试
experiments/           训练数据、模型、结果和日志
assets/pygame/         图形界面展示资源
legacy/pygame_rule/    旧版 Pygame 规则方案
legacy/pre_refactor/   重构前源码、旧模型和旧数据
docs/design/           设计文档和实施计划
docs/notes/            迁移与验收记录
docs/reports/          结项报告
run.py                当前实验统一入口
```

历史备份已移至：

```text
F:\Missions And Materials\神经与规则混合型智驾项目
```

## 快速运行

在项目根目录执行：

```powershell
python run.py collect --episodes 20 --seed 7 --scenario mixed --output experiments/data/training_data.npz
python run.py train --data experiments/data/training_data.npz --model experiments/models/neural_model.pkl --epochs 30 --seed 7
python run.py evaluate --policy hybrid --model experiments/models/neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments/results/hybrid.json
python run.py baseline --model experiments/models/neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments/results/baseline.json
```

`baseline` 会在同一组 seed 和场景下统一评测 `rule`、`neural`、`hybrid` 三条基线。支持的场景包括 `empty`、`slow_lead`、`sudden_brake`、`obstacle`、`rear_approach`、`random` 和可复现的 `mixed`。

## 动作编号

| 编号 | 动作 |
|---:|---|
| 0 | 加速 |
| 1 | 制动 |
| 2 | 左变道 |
| 3 | 右变道 |
| 4 | 巡航 |

## 当前研究链路

```text
HighwayEnv
    ↓ observation
RulePolicy ──→ episode-aware 数据集 ──→ NumPy MLP
    │                                      ↓
    └────────────── 规则基线       NeuralPolicy
                                           ↓
                              SafetyShieldPolicy（可选）
```

观测保留原项目的 11 维特征。环境提供存活、速度跟踪、前进进度、超车、TTC 风险、变道、安全接管和碰撞等奖励分量；评测输出碰撞率、平均回报、平均存活步数、平均速度及事件统计。

强化学习属于后续阶段，目前尚未实现或训练 DQN。图形界面只作为 `legacy` 展示方案保留，后续可围绕 `HighwayEnv` 接入。

## 测试

```powershell
python -m pytest -q
python -m compileall -q src run.py legacy/pygame_rule
```

## 项目表述边界

当前阶段适合表述为“物理/安全约束增强的行为克隆基线”，可体现监督学习数据构建、规则先验、可复现仿真、奖励设计、批量环境、类别不平衡处理和工程化重构。不能将当前仿真结果表述为真实车辆控制性能或已完成的强化学习系统。
