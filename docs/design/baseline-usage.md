# 基线实验说明

当前主版本位于 `src/driving_lab`，由根目录 `run.py` 统一调用。核心环境不依赖 Pygame，图形资源和旧版 GUI 位于 `assets/pygame` 与 `legacy/pygame_rule`。

## 运行链路

```powershell
python run.py collect --episodes 20 --seed 7 --scenario mixed --output experiments/data/training_data.npz
python run.py train --data experiments/data/training_data.npz --model experiments/models/neural_model.pkl --epochs 30 --seed 7
python run.py baseline --model experiments/models/neural_model.pkl --episodes 20 --seed 7 --scenario mixed --output experiments/results/baseline.json
```

`baseline` 使用同一组可复现的种子和场景，依次评测 `rule`、`neural`、`hybrid` 三种策略，并将结果保存为一个 JSON。

## 场景

基础场景：`empty`、`slow_lead`、`sudden_brake`、`obstacle`、`rear_approach`、`random`。`mixed` 会依据 seed 生成可复现的混合场景序列。

## 数据与模型

每个观测保持 11 维特征。示范数据包含 `states`、`actions`、`episodes` 和元数据，并按 episode 划分训练集、验证集和测试集，避免同一条轨迹泄漏到不同数据集。

训练阶段会报告训练准确率、episode 验证准确率和五类动作分布。当前模型是纯 NumPy MLP，后续强化学习阶段再引入 PyTorch 和 GPU。

## 指标

评测 JSON 包含碰撞率、平均存活步数、平均速度、平均回报、超车次数、变道次数、TTC 风险事件和安全屏障接管率；神经策略还会报告相对规则策略的动作一致率。

## 测试

```powershell
python -m pytest -q
```
