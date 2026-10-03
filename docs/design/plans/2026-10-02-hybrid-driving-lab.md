# Hybrid Driving Lab Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `auto/2_ai` 从 Pygame 强耦合原型重构为可复现的规则约束行为克隆实验平台，同时保留自动驾驶项目背景和旧版备份。

**Architecture:** 新建 `driving_lab` 包，分离类型、物理计算、无界面环境、策略、数据集、MLP 和评测；`Main.py` 只做命令行入口。旧版完整副本留在 `_归档/2_ai_before_refactor_2026-10-02`，GUI 后续通过适配器接入。

**Tech Stack:** Python 3.13、NumPy、pytest；第一阶段不引入深度学习框架。

**Spec:** `docs/design/specs/2026-10-02-hybrid-driving-lab-design.md`

## Global Constraints

- 保留官方项目题目和高速公路场景。
- 动作编号固定为 `0=加速, 1=制动, 2=左变道, 3=右变道, 4=巡航`。
- 核心环境不得导入 Pygame。
- 所有随机实验接受显式 seed。
- 不宣称真实车辆控制或严格 PINN。

## Review Focus

- 相同 seed 的 reset 和 episode 轨迹必须一致；对应测试放在 `test_environment.py`。
- 变道边界不能越过最左/最右车道；对应测试放在 `test_environment.py`。
- 紧急制动和 TTC 风险必须优先于加速；对应测试放在 `test_policies.py`。
- 采集数据必须保留 episode 边界，避免随机逐帧切分造成泄漏；对应测试放在 `test_dataset.py`。
- 空数据、旧格式数据和单类动作数据不能让训练流程静默成功；对应测试放在 `test_model.py`。

### Task 1: 定义核心类型与物理函数

**Files:**
- Create: `auto/2_ai/driving_lab/__init__.py`
- Create: `auto/2_ai/driving_lab/types.py`
- Create: `auto/2_ai/driving_lab/physics.py`
- Test: `auto/2_ai/tests/test_types_physics.py`

- [ ] 写动作编码、速度夹紧、TTC 和紧急制动距离的失败测试。
- [ ] 运行 `python -m pytest auto/2_ai/tests/test_types_physics.py -q` 确认测试因模块缺失失败。
- [ ] 实现 `ActionType`、`Action`、`VehicleState`、`Observation`、`Transition` 和物理函数。
- [ ] 重跑测试，确认动作编码和物理边界通过。

### Task 2: 实现无界面高速公路环境

**Files:**
- Create: `auto/2_ai/driving_lab/config.py`
- Create: `auto/2_ai/driving_lab/environment.py`
- Test: `auto/2_ai/tests/test_environment.py`

- [ ] 写固定 seed、步进、车道边界和四类风险场景的失败测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 `HighwayEnv.reset(seed=None, scenario=None)` 与 `HighwayEnv.step(action)`。
- [ ] 实现确定性车辆生成、速度更新、碰撞和 episode 终止。
- [ ] 重跑目标测试并检查核心包不导入 Pygame。

### Task 3: 实现统一策略接口与规则策略

**Files:**
- Create: `auto/2_ai/driving_lab/policies.py`
- Test: `auto/2_ai/tests/test_policies.py`

- [ ] 写动作优先级、紧急制动、跟车和安全变道的失败测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 `Policy` 协议和 `RulePolicy.act(observation)`。
- [ ] 将规则约束集中到策略中，避免策略读取环境内部对象。
- [ ] 重跑测试并确认五类动作均可由规则策略产生。

### Task 4: 实现数据集和 NumPy MLP 训练

**Files:**
- Create: `auto/2_ai/driving_lab/dataset.py`
- Create: `auto/2_ai/driving_lab/model.py`
- Test: `auto/2_ai/tests/test_dataset_model.py`

- [ ] 写数据保存/加载、episode 切分、空数据拒绝和训练形状测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 `DatasetCollector`、`load_dataset`、`MLPClassifier.fit` 和模型持久化。
- [ ] 加入 train/validation 划分、类别分布报告和最小训练指标。
- [ ] 重跑测试并确认模型可保存、加载和推理。

### Task 5: 实现批量评测和命令行入口

**Files:**
- Create: `auto/2_ai/driving_lab/evaluation.py`
- Create: `auto/2_ai/driving_lab/cli.py`
- Modify: `auto/2_ai/Main.py`
- Test: `auto/2_ai/tests/test_evaluation_cli.py`

- [ ] 写规则策略批量评测、神经策略加载和 CLI 子命令测试。
- [ ] 运行目标测试确认失败。
- [ ] 实现 `collect/train/evaluate` 三个命令。
- [ ] 将评测指标序列化为 JSON，并让 `Main.py` 仅转发到 CLI。
- [ ] 重跑目标测试。

### Task 6: 文档与全量验证

**Files:**
- Modify: `auto/2_ai/README.md`
- Create: `auto/2_ai/requirements.txt`
- Test: `auto/2_ai/tests/`

- [ ] 更新运行方式、实验解释、动作编号和简历可用的项目描述。
- [ ] 运行 `python -m pytest -q`。
- [ ] 运行三条验收命令：collect、train、evaluate。
- [ ] 检查无 Pygame 依赖的核心流程和旧版备份完整性。
