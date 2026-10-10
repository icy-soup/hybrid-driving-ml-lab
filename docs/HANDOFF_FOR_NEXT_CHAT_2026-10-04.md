# 新对话交接文档：神经与规则混合型智驾项目

更新时间：2026-10-04

## 0. 给下一个对话的最重要要求

不要重新设计或简化旧 UI/车辆生成逻辑。

用户已经明确说明：旧项目中的 UI、道路坐标、车辆参数、生成逻辑、移动逻辑和回收逻辑是用户逐项调试过的有效基准。新工作应该是：

```text
读取旧逻辑
→ 逐项审计
→ 保留有效行为
→ 建立当前 HighwayEnv 状态到旧 UI 的映射
→ 在不破坏旧逻辑的前提下增加日志、碰撞诊断、MPC、RL 等新模块
```

禁止把旧逻辑替换成“看起来差不多”的简化版本。

## 1. 项目目录

当前项目：

```text
F:\In-class Learning Materials\Aai大三\上\future\任务\项目\神经与规则混合型智驾项目
```

旧项目/有效 UI 基准：

```text
F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\Main.py
F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\ai_module.py
F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\enemy_ai_module.py
F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\images
F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\font
```

论文/参考 Markdown：

```text
F:\In-class Learning Materials\Aai大三\上\future\任务\项目\神经与规则混合型智驾项目\同车行.pdf-aa5b8aa8-94e1-4f2e-bd6a-5e840a3d20e6\full.md
```

只使用 `full.md` 作为论文参考，不要重新转换 PDF。

## 2. 用户真正要的产品目标

项目背景保持学校的多车高速公路智能驾驶题目，但最终用于 AI/ML 简历。重点不是把项目定死为智能驾驶岗位，而是展示：

- 多车交通仿真
- 规则先验和可解释决策
- 几何感知和碰撞诊断
- MPC、行为克隆、DQN/RL 的可比基线
- GPU 训练与 CPU 可复现验证
- 可靠日志和实验评价

最终顺序：

```text
恢复旧 UI/旧车辆行为基准
→ 修复并审计车辆生成、移动和回收
→ 完成日志与碰撞诊断
→ Rule/MPC/BC 基线
→ 用户验收
→ DQN/RL 正式训练
→ 结项报告
→ 用户确认后批量同步 GitHub
```

## 3. 旧 UI 中已经调好的关键参数和逻辑

旧 `Main.py` 明确包含：

- `GAME_WIDTH = 910`
- `PANEL_WIDTH = 240`
- `bg_height = 450`
- `LANE_CENTERS_BOTTOM = [200, 250, 300]`
- 车辆从道路外位置进入，而不是直接出生在自车旁边
- 不同车辆类型、颜色和尺寸
- 前向生成、后向生成、生成间隔和最大车辆数
- 不同速度策略、变道速度和回收边界
- `EnemyAI` 的跟车、紧急制动、变道检查和 warning 状态
- 右侧控制中心、策略、AI 模式、距离、车流密度、状态等信息
- 旧车辆精灵资源不是单一矩形，而是不同车型和颜色的 PNG

旧代码中的这些逻辑必须先读懂再迁移。不能用当前简化的 `_make_random_traffic()` 直接替代全部旧生成机制。

## 4. 当前仓库已经做过的事情

已存在的本地提交包括：

```text
2e83f61  deterministic traffic audit fixtures
912eb78  geometric traffic state and perception
464d70d  explainable traffic driver model
da9639e  collision events and episode diagnostics
53a49fb  traffic-aware evaluation metrics
6b44f0c  discrete MPC policy
7c611b0  GPU-capable DQN and safety shield
19903a5  PyTorch/NumPy runtime compatibility
a0ab386  state-only pygame renderer contract
31fdd03  directly launchable UI
707ef1d  nonnegative audit score and detailed logs
08a5bc8  integrate traffic drivers into highway environment
```

最近一次全量测试曾达到：

```text
71 passed
```

但这不等于车辆逻辑已经可靠。很多测试是模块级测试，不能替代长时间、多 seed、端到端车辆生命周期测试。

## 5. 当前明确暴露的问题

### P0：车辆生成逻辑未按旧代码完整迁移

当前随机生成逻辑仍是简化版本，尚未证明：

- 车辆一定从屏幕外进入
- 出生时一定避开自车和其他车辆
- 后车生成安全窗正确
- 车辆数量和生成间隔与旧逻辑一致
- 车辆 600 步连续生成/回收稳定

### P0：旧 UI 左侧只完成了部分映射

当前新 `pygame_app.py` 曾经错误地把道路做成三等分，导致车辆视觉上跑出道路。后来改回了旧道路高度和车道中心，但仍不能声称与旧 UI 等价。

正确的旧坐标基准是：

```text
lane 0 → y=200
lane 1 → y=250
lane 2 → y=300
ego x anchor ≈ 455
road width = 910
```

### P1：车辆精灵映射不完整

曾经错误地把所有车辆取同一张精灵。新对话必须实现稳定的车辆类型/颜色映射，不能只按文件列表第一张图片显示。

### P1：交通 AI 行为覆盖仍不完整

当前日志曾观察到 `FOLLOW`，但以下状态必须通过固定夹具真正触发并记录：

- `BRAKE`
- `EMERGENCY_BRAKE`
- `PREPARE_LANE_CHANGE`
- `LANE_CHANGE`
- `LANE_CHANGE_ABORT`
- 交通车之间碰撞

### P1：初始碰撞和长时间碰撞不能用“前三步没撞”证明

必须至少跑：

- 每个场景 seed `0–99`
- 每个 episode 至少 `600` 步或直到终止
- 输出初始重叠、最小间距、最小 TTC、碰撞事件和生命周期

## 6. 当前日志证据及其局限

已有日志示例：

```text
experiments/logs/slow_lead_ai_v2.jsonl
experiments/logs/rear_ai_v2.jsonl
experiments/logs/obstacle_debug_v2.jsonl
```

日志已经包含速度、距离、reward components、车辆记录和碰撞事件，但当前下一个对话必须继续检查：

- vehicle mode 是否能解释每次速度变化
- vehicle ID 是否贯穿生成、运动、回收
- spawn/recycle 事件是否记录
- vehicle y 是否始终等于旧 UI 车道中心或变道插值
- collision 前一步是否有对应 TTC/间距变化

## 7. 新对话的强制测试计划

### A. 旧逻辑基准测试

先只读旧 `Main.py`，整理出：

- 生成入口
- 生成间隔公式
- 前向/后向生成条件
- 车道选择
- 最大车辆数
- 删除边界
- 车辆速度更新公式
- 变道积分公式
- 旧碰撞矩形定义

不要先写新代码。

### B. 生成测试

新增测试必须覆盖：

1. 初始车辆全部在有效的屏幕外/道路范围。
2. 初始车辆和自车无矩形重叠。
3. 同车道前后间距满足旧逻辑安全窗。
4. 后车生成不会直接出现在自车碰撞区。
5. 每辆车 ID 唯一。
6. 连续 600 步生成与回收后数量有上限。
7. 不同 seed 不能出现初始碰撞。

### C. 运动测试

每步保存：

```text
step, vehicle_id, x, y, lane, speed, acceleration,
target_lane, mode, decision_reason
```

检查：

- x 单调性符合车辆相对自车方向
- y 在车道中心或合法变道插值
- 加速度有上下限
- 车辆不会瞬移
- 车辆不会穿透

### D. 长时间端到端测试

每个固定 seed 至少 600 步，不能只测前三步。输出：

- 初始车辆列表
- 每辆车的首次出现和最后出现步
- 每辆车进入道路的步数
- 每辆车离开道路的步数
- 最小同车道间距
- 最小 TTC
- 所有碰撞事件
- 行为模式计数

### E. UI 映射测试

使用真实 `HighwayEnv` 快照验证：

- lane 映射到 `200/250/300`
- x=0 映射到 ego anchor
- x<0 由左侧进入
- x>910 后进入回收范围
- 不同 vehicle_id 使用不同稳定精灵
- UI 不改变快照和环境状态

### F. 评分和日志测试

必须分别显示：

- 原始距离
- 归一化安全分
- shaped reward
- 碰撞惩罚

不能把负的 shaped reward 叫作主分，也不能只输出最后一行摘要而没有逐步日志。

## 8. 新对话的实施顺序

严格按照下面顺序，不要跳步：

1. 阅读旧 `Main.py` 生成/移动/回收代码。
2. 写旧逻辑审计表，不改代码。
3. 写生成和生命周期测试，先确认测试能抓住当前错误。
4. 实现旧生成逻辑到当前环境的适配，不重写 UI。
5. 跑 100 seed × 600 步日志。
6. 修复初始碰撞、跑道外车辆、回收和间距问题。
7. 写 UI 左侧坐标/精灵映射测试。
8. 只调整右侧文字和信息面板。
9. 全量测试和长序列日志通过后，再给用户验收。
10. 用户验收后，才继续 DQN 正式训练和 GitHub 同步。

## 9. 当前禁止事项

- 不要重新画左侧道路
- 不要用矩形替代旧车辆精灵
- 不要把旧生成逻辑简化成随机 x/lane/speed
- 不要只跑前三步
- 不要只看 pytest 通过就宣称车辆逻辑完成
- 不要在 P0 问题未解决前训练 RL
- 不要向 GitHub 推送未经用户验收的版本

## 10. 本交接的结论

当前版本是“部分模块已完成，但旧车辆生命周期和旧 UI 左侧行为尚未可靠迁移”的中间状态。下一个对话必须从旧代码审计和测试计划开始，而不是继续凭感觉修改 `pygame_app.py` 或重新设计车辆生成器。
