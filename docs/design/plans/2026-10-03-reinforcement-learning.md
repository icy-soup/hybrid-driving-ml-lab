# GPU强化学习决策扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有无界面高速公路环境上加入可解释奖励、GPU DQN、行为克隆初始化、安全屏障和可复现实验对照。

**Architecture:** 保持 `HighwayEnv` 的 `reset/step` 接口，在环境层增加事件统计和奖励分解；在 `driving_lab/rl/` 中建立 PyTorch DQN、经验回放、并行采样和 checkpoint；在 `experiments/` 中统一运行 Rule、BC、DQN 和 shielded DQN 对照。Pygame 仍然不参与训练。

**Tech Stack:** Python 3.13、NumPy、PyTorch 2.6+（CUDA 12.4 可用时使用 RTX 4060 Laptop GPU）、pytest、JSON/CSV；CPU 作为自动回退设备。

**Spec:** `docs/design/specs/2026-10-03-reinforcement-learning-design.md`

**Execution gate:** 强化学习实现必须等“基线阶段”验收通过后才能开始。基线包括可复现环境、规则策略、行为克隆模型、场景集、数据划分和统一评测；若基线指标或环境语义不稳定，先修基线，不训练 DQN。

## Global Constraints

- 保留官方项目题目和自动驾驶高速公路背景。
- 动作编号继续使用 `0=加速, 1=制动, 2=左变道, 3=右变道, 4=巡航`。
- 核心训练环境不得依赖 Pygame。
- 所有训练和评测命令支持显式 seed、checkpoint 路径和 device 覆盖。
- raw 与 shielded 策略必须分开报告。
- 不能把仿真结果表述为真实车辆控制性能。

## Review Focus

- 碰撞终止必须同时产生 collision 惩罚和 done；测试在 `test_reward.py`。
- 超车只能在安全通过事件发生时奖励一次；测试在 `test_reward.py`。
- GPU 不可用时训练仍能在 CPU 完成一个 smoke run；测试在 `test_dqn.py`。
- target network 更新和 replay buffer 采样不能共享可变数组；测试在 `test_replay.py`。
- BC 初始化和随机初始化必须使用同一评测种子/场景；测试在 `test_experiments.py`。

## Phase 0: Baseline Gate

以下条件全部满足后，才进入 Task 3 的强化学习实现：

- `RulePolicy` 在固定 seeds 和风险场景下可复现；
- 数据集按 episode 划分，动作分布和 11 维特征通过检查；
- Behavior Cloning 有独立验证集，并报告动作分类指标；
- Rule、BC、SafetyShield 三种策略都能完成统一闭环评测；
- 评测 JSON 至少包含碰撞率、平均存活步数、平均速度、TTC 风险和变道/超车统计；
- 基线结果保存为冻结的 benchmark，后续 DQN 必须使用同一批 seeds 和场景对比。

### Task 1: 奖励配置与环境事件

**Files:**
- Create: `src/driving_lab/rewards.py`
- Modify: `src/driving_lab/config.py`
- Modify: `src/driving_lab/environment.py`
- Test: `tests/test_reward.py`

- [x] 先写奖励分量、碰撞终止、一次性超车事件和 shield 接管惩罚的失败测试。
- [x] 运行目标测试确认在缺少 `RewardConfig`/事件字段时失败。
- [x] 实现 `RewardConfig`、`EpisodeEvents` 和 `RewardBreakdown`，让 `HighwayEnv.step()` 返回奖励分量及事件计数。
- [x] 重新运行目标测试，确认奖励没有因速度或重复检测无限增长。

### Task 2: 训练场景采样与并行环境

**Files:**
- Create: `src/driving_lab/scenarios.py`
- Create: `src/driving_lab/vector_env.py`
- Modify: `src/driving_lab/environment.py`
- Test: `tests/test_vector_env.py`

- [x] 写固定 seed、场景课程、批量 reset/step 和 episode 完成边界测试。
- [x] 运行目标测试确认失败。
- [x] 实现 mixed scenario sampler 与 `VectorHighwayEnv`，采集仍在 CPU，返回 NumPy batch。
- [x] 确认并行环境不会共享 ego/traffic 可变状态。

### Baseline Gate Status (2026-10-03)

- [x] 增加 `baseline` CLI，一次性比较 Rule / Neural / Hybrid 并保存 JSON。
- [x] 增加动作分布诊断、奖励/事件指标和 episode-aware 数据划分。
- [x] 基线阶段通过 33 项测试、`compileall` 检查和一次独立的 collect → train → baseline 流水线演练；结构迁移后总测试数为 34 项。
- [ ] 等用户验收固定 seeds、场景和指标后冻结 benchmark；在此之前不实现或训练 DQN。

### Task 3: PyTorch DQN 与经验回放

**Files:**
- Create: `src/driving_lab/rl/__init__.py`
- Create: `src/driving_lab/rl/replay.py`
- Create: `src/driving_lab/rl/dqn.py`
- Test: `tests/test_replay.py`
- Test: `tests/test_dqn.py`

- [ ] 写 replay shape、target network、epsilon schedule、device fallback 和单 batch 更新测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 `QNetwork(11, 128, 128, 5)`、`ReplayBuffer`、`DQNAgent` 和 `DQNConfig`。
- [ ] 实现 Huber loss、Double-DQN 可选开关、梯度裁剪、checkpoint save/load。
- [ ] 用 CPU smoke run 和 CUDA（可用时）各运行至少一个更新步。

### Task 4: 行为克隆初始化

**Files:**
- Create: `src/driving_lab/rl/behavior_init.py`
- Modify: `src/driving_lab/dataset.py`
- Test: `tests/test_behavior_init.py`

- [ ] 写 BC classifier 训练后复制 backbone/输出层到 Q 网络的形状和预测一致性测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 PyTorch BC 训练、checkpoint 转换和随机初始化对照。
- [ ] 明确记录 BC train/validation 指标，不将其当作 RL return。

### Task 5: 训练循环、安全屏障和实验 CLI

**Files:**
- Create: `src/driving_lab/rl/trainer.py`
- Create: `src/driving_lab/rl/shield.py`
- Create: `experiments/train_dqn.py`
- Create: `experiments/evaluate_policies.py`
- Modify: `run.py`
- Test: `tests/test_experiments.py`

- [ ] 写短训练 smoke run、raw/shielded action 分离和 checkpoint 恢复测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 warmup、epsilon-greedy、replay 更新、target sync、评测周期和 best checkpoint。
- [ ] 实现 shield override 记录和原始/执行动作双日志。
- [ ] 提供 `train-dqn`、`evaluate-all` 命令，固定 seeds 并输出 JSON/CSV。

### Task 6: 对照实验、可视化数据和文档

**Files:**
- Create: `experiments/configs/*.json`
- Create: `experiments/results/.gitkeep`
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: root `README.md`

- [ ] 写规则、BC、DQN-random-init、DQN-BC-init、DQN-shielded 的统一评测配置。
- [ ] 运行固定 seeds 的 smoke/full 实验，保存 reward 曲线、指标 JSON 和汇总 CSV。
- [ ] 增加 CPU fallback、CUDA 设备、训练命令和结果解释。
- [ ] 更新简历表述，明确奖励设计、DQN、GPU 训练和安全屏障的实际边界。

### Task 7: 全量验证

**Files:**
- Test: `tests/`

- [ ] 运行 `python -m compileall -q src experiments run.py`。
- [ ] 运行 `python -m pytest -q`。
- [ ] 运行 CPU smoke train 和 CUDA smoke train（CUDA 可用时）。
- [ ] 检查相同 seed 的结果稳定、checkpoint 可恢复、raw/shielded 指标均存在。
