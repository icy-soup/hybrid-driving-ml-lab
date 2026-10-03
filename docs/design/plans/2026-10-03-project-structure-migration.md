# 项目结构迁移 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将历史目录式项目整理为职责型结构，同时保持当前基线代码、命令、数据格式和评测行为不变。

**Architecture:** 当前主代码迁移到 `src/driving_lab`，根目录 `run.py` 作为唯一实验入口；测试、实验产物、图形资源、历史实现和文档分别放入独立目录。旧版 GUI 只作为 `legacy/pygame_rule` 保存，不参与主版本测试。

**Tech Stack:** Python 3.13、NumPy、pytest、PowerShell 文件迁移；不新增运行时依赖，不启动强化学习训练。

**Spec:** `docs/design/specs/2026-10-03-project-structure-design.md`

## Global Constraints

- 动作编号继续使用 `0=加速, 1=制动, 2=左变道, 3=右变道, 4=巡航`。
- `HighwayEnv`、策略接口、NPZ 数据字段、模型格式和评测指标保持不变。
- `collect`、`train`、`evaluate`、`baseline` 四个 CLI 子命令语义保持不变。
- 不实现强化学习，不修改奖励权重和环境语义。
- 不修改结项报告正文；只移动到 `docs/reports/结项报告.doc`。
- 外部历史备份 `F:\Missions And Materials\神经与规则混合型智驾项目` 不复制、不删除。

## Review Focus

- 根目录运行入口必须能找到 `src/driving_lab`，测试覆盖 `run.py --help`。
- 测试迁移后不能依赖旧的 `auto/2_ai` 路径，测试覆盖源码目录导入。
- 旧 GUI 资源路径必须指向 `assets/pygame`，静态检查覆盖字体、背景和车辆资源引用。
- CLI 输出路径必须能创建嵌套的 `experiments/data`、`models`、`results` 目录。
- 文档中的旧路径不能误导用户，扫描覆盖 `README.md`、设计文档和实施计划。

### Task 1: 迁移前布局契约

**Files:**
- Create: `tests/test_project_layout.py`

- [x] **Step 1: Write the failing layout test**

  测试根目录必须包含 `run.py`、`src/driving_lab`、`tests`、`experiments`、`assets`、`legacy` 和 `docs/reports`，并断言旧的 `auto/2_ai/driving_lab` 不再作为主代码路径。

- [x] **Step 2: Run the test to verify it fails**

  Run: `python -m pytest tests/test_project_layout.py -q`

  Expected: FAIL，因为目标结构尚未建立。

### Task 2: 迁移当前主代码和测试

**Files:**
- Move: `auto/2_ai/driving_lab/` → `src/driving_lab/`
- Move: `auto/2_ai/tests/` → `tests/`
- Move: `auto/2_ai/requirements.txt` → `requirements.txt`
- Move: `auto/2_ai/README.md` → `docs/design/baseline-usage.md`
- Create: `run.py`
- Modify: `tests/*.py`

**Interfaces:**
- Produces: 根目录 `run.py`，将项目 `src` 加入 `sys.path` 后调用 `driving_lab.cli.main`。

- [x] **Step 1: 创建目标目录并移动文件**
- [x] **Step 2: 更新测试导入路径为根目录 `src`**
- [x] **Step 3: 创建 `run.py` 薄入口**
- [x] **Step 4: 运行 `python run.py --help`，确认出现四个 CLI 子命令**
- [x] **Step 5: 运行 `python -m pytest tests -q`，确认主测试通过**

### Task 3: 整理 GUI 资源和历史实现

**Files:**
- Move: `auto/1_rule/` → `legacy/pygame_rule/`
- Move: `images/` → `assets/pygame/images/`
- Move: `auto/1_rule/font/` → `assets/pygame/font/`
- Move: `auto/2_ai/legacy/` → `legacy/pre_refactor/2_ai/`
- Modify: `legacy/pygame_rule/Main.py`

- [x] **Step 1: 移动旧 GUI、资源和旧模型文件**
- [x] **Step 2: 让旧 GUI 按 `__file__` 定位 `assets/pygame`，不依赖当前工作目录**
- [x] **Step 3: 静态检查字体、道路、车辆资源路径均已指向 `assets/pygame`**
- [x] **Step 4: 确认 `legacy` 不被主版本导入**

### Task 4: 整理实验产物和结项材料

**Files:**
- Create: `experiments/data/`
- Create: `experiments/models/`
- Create: `experiments/results/`
- Create: `experiments/logs/`
- Create: `docs/design/`
- Create: `docs/notes/`
- Create: `docs/reports/`
- Move: `docs/superpowers/specs/` → `docs/design/specs/`
- Move: `docs/superpowers/plans/` → `docs/design/plans/`
- Move: `结项报告.doc` → `docs/reports/结项报告.doc`

- [x] **Step 1: 创建实验和文档目录**
- [x] **Step 2: 移动设计稿、实施计划和结项报告，不修改报告正文**
- [x] **Step 3: 在 `docs/notes/` 记录迁移日期、外部备份路径和验证结果**

### Task 5: 更新公开文档和路径引用

**Files:**
- Modify: `README.md`
- Modify: `docs/design/**/*.md`
- Create: `docs/notes/project-layout.md`

- [x] **Step 1: 更新根 README 的目录树、运行命令和文件链接**
- [x] **Step 2: 将文档中的 `auto/2_ai`、`auto/1_rule` 和 `docs/superpowers` 改为目标路径或历史说明**
- [x] **Step 3: 写入新的项目结构说明和外部归档位置**
- [x] **Step 4: 扫描旧路径引用并确认只剩迁移说明**

### Task 6: 全量验证

- [x] **Step 1: 运行 `python -m compileall -q src run.py legacy/pygame_rule`**
- [x] **Step 2: 运行 `python -m pytest -q`**
- [x] **Step 3: 在 `experiments/` 下跑 collect → train → evaluate → baseline**
- [x] **Step 4: 比较迁移前后的指标字段、动作编号和 baseline JSON 顶层键**
- [x] **Step 5: 确认外部历史备份仍然存在，且结项报告内容未被修改**
