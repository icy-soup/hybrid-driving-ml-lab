# 神经与规则混合型智驾项目：最新交接文档

更新时间：2026-10-10

本文覆盖 2026-10-04 之前的交接文档。后续对话以本文的代码状态和测试结果为准；旧文档中的“71 passed”和“交通车逻辑尚未完成”等描述已经过时。

## 1. 项目目标

项目背景保持为学校正式题目的多车高速公路智能驾驶场景，最终用于人工智能/机器学习方向的简历展示。重点是可复现的多车仿真、几何感知、TTC、碰撞诊断、规则先验、MPC、行为克隆、DQN/RL、安全屏障和实验日志。

强化学习尚未进行正式训练。当前应先冻结环境、交通车行为、碰撞判定、日志和评价指标。

## 2. 当前代码入口

```text
run.py                         统一命令入口
src/driving_lab/environment.py 唯一环境状态推进者
src/driving_lab/traffic.py     NPC 交通驾驶器和交通世界
src/driving_lab/perception.py  五区域几何感知与 TTC
src/driving_lab/controllers/mpc.py  离散 MPC 基线
src/driving_lab/policies.py    Rule / Neural / SafetyShield
src/driving_lab/pygame_app.py  只读环境快照的 Pygame UI
tests/                         环境、策略、碰撞、UI 和数据契约测试
legacy/                        重构前代码和旧模型备份
docs/                          设计、审计、交接和报告
```

论文参考 Markdown：

```text
同车行.pdf-aa5b8aa8-94e1-4f2e-bd6a-5e840a3d20e6/full.md
```

## 3. 动作编号和策略接口

| 编号 | 动作 |
|---:|---|
| 0 | ACCELERATE，加速 |
| 1 | BRAKE，制动 |
| 2 | LANE_LEFT，左变道 |
| 3 | LANE_RIGHT，右变道 |
| 4 | CRUISE，巡航 |

策略接口是 `policy.act(observation) -> Action`。UI、批量评测和未来 RL 都使用这个接口，环境不会为 UI 再维护一套车辆逻辑。

## 4. 最近完成的关键修复

### 4.1 碰撞和车辆尺寸

- ego 与 NPC 碰撞记录参与者、类型、步数、间距、相对速度、TTC、碰撞前速度和碰撞前动作。
- NPC-NPC 碰撞会检测、冻结双方并保留爆炸动画状态。
- 随机 NPC 按车型设置物理长度和宽度，避免“精灵已经重叠、物理车身却没有碰撞”。
- NPC 进入车道前按最大车型车身长度检查间距，避免生成后由间距修复逻辑大幅挪动车辆。

### 4.2 高速场景和速度

- 内部动力学使用 m/s，UI 和评测显示 km/h。
- 道路最高速度：`38.8889 m/s = 140 km/h`。
- MPC 基线巡航目标：`27.7778 m/s = 100 km/h`。
- NPC 高速巡航目标：`24 m/s = 86.4 km/h`。
- `MPCConfig.target_speed` 是独立策略参数，未来可作为 RL 学习参数。
- `evaluate_policy()` 新增 `average_speed_kmh`，保留兼容用的 `average_speed`。

### 4.3 定速和连续运动

- MPC 在目标速度附近使用速度滞环，进入目标区间后保持 `CRUISE`，避免 `ACCELERATE/BRAKE` 每帧抖动。
- NPC 可以加速到高速巡航目标，不再被旧的相对坐标兼容分支锁死在出生速度。
- 手工场景继续使用低速/慢车标定行为，保证固定场景测试稳定。
- `run.py ui` 默认 FPS 已改为 30；`start_ui.bat` 显式使用 `--fps 30`。

## 5. 已验证结果

最近验证命令：

```powershell
python -m pytest -q --ignore=experiments --ignore=tests/test_dataset_model.py --ignore=tests/test_dqn.py --ignore=tests/test_evaluation_cli.py
python -m compileall -q src tests
git diff --check
```

结果：

```text
76 passed
```

长轨迹验证：

| 场景 | seed | 步数 | 碰撞 | 备注 |
|---|---:|---:|---:|---|
| mixed | 7 | 600 | 0 | MPC，车辆连续运动最大单步约 3 px |
| mixed | 0 | 1100 | 0 | NPC 速度稳定在高速目标附近 |
| mixed | 7 | 1100 | 0 | 自车约 84.6 km/h，无碰撞 |

此前第 212 步 rear-end 的原因是慢车在物理上来不及制动的距离内才被注入当前车道，相关生成约束已加入环境。

## 6. 当前运行方式

```powershell
cmd /d /c start_ui.bat
```

或者：

```powershell
python run.py ui --policy mpc --scenario mixed --seed 7 --fps 30
```

命令行长轨迹验证：

```powershell
$env:PYTHONPATH='src'
@'
from driving_lab.environment import HighwayEnv
from driving_lab.controllers.mpc import MPCPolicy
env = HighwayEnv()
obs = env.reset(seed=7, scenario='mixed')
policy = MPCPolicy()
for step in range(1, env.config.max_steps + 1):
    transition = env.step(policy.act(obs))
    obs = transition.next_observation
    if transition.done:
        print(transition.info)
        break
'@ | python -
```

重点查看 `transition.info` 中的 `collision`、`impact_speed`、`impact_ttc`、`action_before_collision`、`collision_events`、`vehicle_snapshot` 和 `traffic_decisions`。

## 7. 当前未完成事项

1. 尚未完成所有场景、seed 0–99、每个 600 步的全量生命周期审计。
2. 旧版完整 UI 的每个坐标、交互和精灵映射还需要继续与旧 `Main.py` 做逐项截图比对。
3. `BRAKE`、`EMERGENCY_BRAKE`、`LANE_CHANGE_ABORT` 和交通车之间碰撞需要继续增加固定夹具覆盖。
4. 当前 MPC 是可解释的离散短视基线，不是正式的连续优化 MPC。
5. DQN/RL 只有训练骨架和安全屏障，尚未进行正式 GPU 训练和算法对比。
6. 结项报告应在评价协议和真实实验结果冻结后再更新。

## 8. 下一次对话的工作顺序

1. 先读取本文和旧版 `legacy/pygame_rule` / `legacy/pre_refactor`，不要重新发明 UI 和车辆坐标。
2. 继续做 600 步以上的多 seed 车辆生命周期、生成、回收和碰撞审计。
3. 增加 NPC 行为模式和运动连续性的固定测试。
4. 用户验收 UI、日志和基线结果。
5. 冻结环境后再进行 DQN/RL smoke test 和 GPU 训练。
6. 最后更新结项报告和简历表述。

## 9. GitHub 交付说明

远程仓库：<https://github.com/icy-soup/hybrid-driving-ml-lab>

本次交付应包含本交接文档、当前源代码、测试和必要文档。不要提交 `ui_check.png`、pytest 临时目录、`.superpowers/` 临时状态或本地训练生成物。
