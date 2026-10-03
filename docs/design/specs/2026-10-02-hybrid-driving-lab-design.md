# 物理信息增强的数据驱动型自动驾驶算法研究项目：重构设计

## 目标

保留学校项目的自动驾驶背景和官方题目，将当前 Pygame 原型重构为一个可复现、可批量运行、可评估的规则约束行为克隆实验平台。图形界面作为后置展示层，不再承担仿真、数据集和训练逻辑。

## 项目定位

项目对外仍使用“物理信息增强的数据驱动型自动驾驶算法研究项目”。技术上聚焦通用的机器学习问题：结构化状态表示、规则先验生成监督数据、行为克隆、类别不平衡、策略评测和可复现实验。高速公路是当前实验场景，不把项目能力限定为特定岗位。

## 核心边界

1. 仿真环境不依赖 Pygame，提供 `reset(seed) -> observation` 和 `step(action) -> transition`。
2. 规则策略和神经网络策略遵守同一 `Policy` 接口。
3. 动作用枚举值表达：加速、制动、左变道、右变道、巡航，编号保持 0–4 兼容现有数据语义。
4. 所有随机场景支持显式随机种子。
5. 数据采集、训练和评测可以从命令行运行，不需要打开窗口。
6. Pygame 代码暂时保存在备份中；后续只通过适配器接入新环境。

## 目标架构

```text
auto/2_ai/
├── driving_lab/
│   ├── config.py       # 环境、车辆和实验配置
│   ├── types.py        # 状态、观测、动作和转移数据结构
│   ├── physics.py      # TTC、制动距离和数值约束
│   ├── environment.py  # 无渲染高速公路环境
│   ├── policies.py     # RulePolicy / NeuralPolicy / SafetyShieldPolicy
│   ├── dataset.py      # 数据采集、保存和加载
│   ├── model.py        # 纯 NumPy MLP 与训练
│   ├── evaluation.py   # 批量评测和指标
│   └── cli.py          # collect/train/evaluate 命令
├── tests/              # 核心行为测试
├── Main.py             # 新版无界面实验入口
└── legacy/             # 重构前代码与旧资源的可恢复副本
```

## 实验能力

第一版必须支持：

- 固定种子生成同一场景；
- 前车慢车、前车突然减速、前方事故车、相邻车道有后车四类风险场景；
- 规则策略产生 `(observation, action)` 数据；
- 11 维观测与 5 类动作编号兼容旧数据；
- 按 episode 划分数据集，避免同一轨迹泄漏到验证集；
- MLP 训练输出训练/验证损失与准确率；
- 规则策略和神经策略的碰撞率、平均速度、平均存活步数、动作准确率和 TTC 风险指标。
- 神经策略外提供可选的安全屏障，在硬风险条件下强制执行制动规则，并单独报告 raw neural 与 hybrid 结果。

## 非目标

- 第一阶段不引入 PyTorch、强化学习框架或复杂 ECS；
- 不把系统描述为真实车辆控制器；
- 不把当前实现宣传为严格意义上的 PINN；
- 第一阶段不重做 GUI 美术和动画。

## 兼容性与迁移

旧版完整副本保存在 `_归档/2_ai_before_refactor_2026-10-02`。新数据文件使用 `npz`，包含 `states`、`actions`、`episodes` 和元数据；旧版 `training_data.npz` 可以读取，但若缺少 `episodes` 则按单一数据源处理并给出提示。

## 验收标准

在没有 Pygame 窗口的情况下，以下流程成功运行：

```text
python Main.py collect --episodes 20 --seed 7
python Main.py train --epochs 30
python Main.py evaluate --episodes 20 --seed 7
```

测试覆盖动作编码、TTC/制动距离、确定性 reset、场景风险、数据保存加载、训练形状和评测指标。
