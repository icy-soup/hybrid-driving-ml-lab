# Traffic Simulation and UI Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the passive traffic approximation with an explainable multi-vehicle simulation, expose collision diagnostics, and reconnect the original visual UI to the canonical environment without changing the current policy interface.

**Architecture:** Keep `HighwayEnv` as the single state transition owner. Add focused perception, traffic-driver, event-diagnostics, and rendering modules; traffic vehicles use independent driver models while the ego vehicle continues to use Rule/Neural/Hybrid policies. The UI renders environment snapshots only and never advances its own copy of traffic.

**Tech Stack:** Python 3.11+, NumPy, pytest, optional Pygame 2.5+, JSONL diagnostics, existing PNG/TTF assets.

**Spec:** `docs/design/specs/2026-10-04-traffic-simulation-and-ui-rebuild.md`

## Global Constraints

- Do not push to GitHub during implementation; remote `main` remains at `368d02f` until user acceptance.
- Use the paper principles from `同车行.../full.md` as design references, not as copied code or a claim of reproducing its experiments.
- `policy.act(observation) -> Action` remains the ego-policy interface.
- Pygame is optional; headless tests and training must run without opening a window.
- Use fixed seeds for all reproducibility tests.
- Every behavior change gets a failing test before implementation.

## Review Focus

- Vehicles initialized in the same lane with insufficient geometric gap: test deterministic rejection or spacing correction in Task 2.
- A faster rear vehicle catches a slower front vehicle: test speed convergence and no penetration in Task 3.
- A lane change begins while the target lane is occupied or closing quickly: test abort/pause in Task 3.
- Collision detection at the exact boundary and during lateral overlap: test event type and pre-collision snapshot in Task 4.
- UI and headless environment diverge after one step: test that the renderer consumes a snapshot and never updates simulation state in Task 6.

### Task 1: Freeze the baseline and define audit fixtures

**Files:**
- Create: `tests/fixtures/traffic_cases.py`
- Create: `docs/notes/traffic-ai-audit-2026-10-04.md`
- Test: `tests/test_traffic_fixtures.py`

**Interfaces:**
- Produces deterministic vehicle configurations for empty road, following, rear approach, lane-change conflict, and collision cases.
- Records the old implementation's known differences without importing legacy Pygame modules into the new tests.

- [ ] **Step 1: Write failing fixture tests** for deterministic IDs, lane assignments, and expected relative positions.
- [ ] **Step 2: Run `E:\Anaconda3\python.exe -m pytest -q tests/test_traffic_fixtures.py` and verify the fixture module is missing.
- [ ] **Step 3: Implement the fixture factory with `make_case(name: str) -> list[VehicleState]`.
- [ ] **Step 4: Run the fixture tests and verify they pass.
- [ ] **Step 5: Write the audit note comparing legacy `EnemyAI` behavior with the canonical model; do not modify legacy files.
- [ ] **Step 6: Commit locally: `test: add deterministic traffic audit fixtures`.

### Task 2: Introduce geometric vehicle state and perception

**Files:**
- Modify: `src/driving_lab/types.py`
- Modify: `src/driving_lab/config.py`
- Create: `src/driving_lab/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Add `BehaviorMode` enum and extend `VehicleState` with continuous lateral position, target lane, desired speed, acceleration, dimensions, and mode while preserving compatible constructors where practical.
- Add `VehicleGeometry(vehicle_id, x, y, lane, length, width, speed, mode)`.
- Add `PerceptionSnapshot.front_center`, `front_left`, `front_right`, `rear_left`, `rear_right` and `lane_occupancy`.
- Implement `perceive(ego: VehicleGeometry, vehicles: Sequence[VehicleGeometry], config: EnvironmentConfig) -> PerceptionSnapshot`.

- [ ] **Step 1: Write failing tests** for bumper-to-bumper distances, relative speeds, TTC, lane overlap, and exact boundary classification.
- [ ] **Step 2: Run `pytest tests/test_perception.py -q`; verify failures are caused by missing geometry/perception APIs.
- [ ] **Step 3: Implement the state additions and geometry-based perception using the paper's five sensor regions.
- [ ] **Step 4: Run the focused perception tests and verify pass.
- [ ] **Step 5: Run the existing type/physics tests and fix compatibility without changing action numbering.
- [ ] **Step 6: Commit locally: `feat: add geometric traffic state and perception`.

### Task 3: Implement independent traffic-driver behavior

**Files:**
- Create: `src/driving_lab/traffic.py`
- Modify: `src/driving_lab/environment.py`
- Modify: `src/driving_lab/config.py`
- Test: `tests/test_traffic_behavior.py`

**Interfaces:**
- Add `TrafficProfile(name, desired_speed, headway, max_acceleration, comfortable_deceleration, lane_change_threshold, lane_change_cooldown)`.
- Add `TrafficDriver(profile: TrafficProfile).decide(vehicle, perception, time_step) -> TrafficDecision`.
- Add `TrafficDecision(target_speed, target_lane, behavior_mode, reason)`.
- Add `TrafficWorld.step_traffic(vehicles, ego, scenario) -> TrafficStepResult`.

- [ ] **Step 1: Write failing tests** for cruise, following, emergency braking, safe lane change, blocked lane change, continuous lane change, and profile differences.
- [ ] **Step 2: Run the focused tests and verify the missing `TrafficDriver`/`TrafficWorld` APIs fail.
- [ ] **Step 3: Implement longitudinal control with bounded acceleration and headway/TTC checks.
- [ ] **Step 4: Implement target-lane selection and lane-change pause/abort using front/rear safety windows.
- [ ] **Step 5: Implement continuous integration of x/y and enforce vehicle geometry spacing after integration.
- [ ] **Step 6: Run focused behavior tests and verify no same-lane penetration across multiple steps.
- [ ] **Step 7: Integrate `TrafficWorld` into `HighwayEnv.step` while leaving ego policy selection unchanged.
- [ ] **Step 8: Run environment, policy, reward, and vector-environment regression tests.
- [ ] **Step 9: Commit locally: `feat: add explainable traffic driver model`.

### Task 4: Add collision events and episode diagnostics

**Files:**
- Create: `src/driving_lab/events.py`
- Modify: `src/driving_lab/rewards.py`
- Modify: `src/driving_lab/environment.py`
- Modify: `src/driving_lab/evaluation.py`
- Modify: `src/driving_lab/cli.py`
- Test: `tests/test_collision_events.py`
- Test: `tests/test_diagnostics_cli.py`

**Interfaces:**
- Add immutable `CollisionEvent` with IDs, collision type, step, scenario, lanes, gaps, speeds, TTC, behavior modes, and reason.
- Add `StepTrace` and `EpisodeDiagnostics` with `record_step`, `record_collision`, `write_jsonl`, and `summary` methods.
- Extend `Transition.info` with `collision_events`, `vehicle_snapshot`, and `diagnostics` fields while keeping existing keys.
- Add CLI command `diagnose` that runs a seeded episode and writes JSONL plus summary JSON.

- [ ] **Step 1: Write failing tests** for rear-end, side, ego-traffic, and traffic-traffic event classification and required fields.
- [ ] **Step 2: Run focused tests and verify event types and CLI command are absent.
- [ ] **Step 3: Implement event classification after state integration and before episode termination.
- [ ] **Step 4: Implement step snapshots and JSONL/summary serialization with stable JSON types.
- [ ] **Step 5: Integrate diagnostics into `HighwayEnv` and `evaluate_policy` without changing existing metric names.
- [ ] **Step 6: Implement `run.py diagnose --scenario ... --seed ... --output ...`.
- [ ] **Step 7: Run focused diagnostics tests and inspect one generated trace manually.
- [ ] **Step 8: Commit locally: `feat: add collision events and episode diagnostics`.

### Task 5: Rebuild evaluation around traffic-aware metrics

**Files:**
- Modify: `src/driving_lab/evaluation.py`
- Modify: `src/driving_lab/rewards.py`
- Modify: `src/driving_lab/cli.py`
- Test: `tests/test_evaluation_metrics.py`

**Interfaces:**
- Preserve `evaluate_policy` return keys and add traffic collision counts, collision-type counts, minimum TTC, comfort effort, and diagnostic path when requested.

- [ ] **Step 1: Write failing tests** for metric aggregation across multiple collision types and episodes.
- [ ] **Step 2: Run focused tests and verify new metrics are absent.
- [ ] **Step 3: Implement aggregation from `EpisodeDiagnostics.summary`.
- [ ] **Step 4: Verify baseline output remains backward-compatible and includes the new fields.
- [ ] **Step 5: Commit locally: `feat: extend traffic-aware evaluation metrics`.

### Task 6: Replace the experimental UI with a state-only renderer

**Files:**
- Delete/recreate: `src/driving_lab/pygame_app.py`
- Delete: `start_ui.bat`
- Modify: `src/driving_lab/cli.py`
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Test: `tests/test_pygame_renderer.py`

**Interfaces:**
- Add `render_snapshot(screen, snapshot, assets, metrics)` with no environment mutation.
- Add `run_ui(policy_name, model_path, scenario, seed, fps)` that advances only `HighwayEnv` and passes snapshots to the renderer.
- Reuse `assets/pygame` vehicle/road/font resources and keep UI optional.

- [ ] **Step 1: Write failing renderer tests** proving the renderer does not change vehicle positions and can render a snapshot without a policy.
- [ ] **Step 2: Run focused tests and verify the state-only renderer contract is missing.
- [ ] **Step 3: Implement asset loading with graceful fallback and old-layout-inspired road/panel composition.
- [ ] **Step 4: Implement vehicle sprites, behavior labels, TTC, recent-event panel, and collision overlay from snapshots.
- [ ] **Step 5: Integrate UI command and add a launcher only after the UI smoke test passes.
- [ ] **Step 6: Run headless Pygame smoke test with `SDL_VIDEODRIVER=dummy` and a posted QUIT event.
- [ ] **Step 7: Commit locally: `feat: render canonical traffic state in pygame UI`.

### Task 7: Full audit, reproducibility run, and user acceptance package

**Files:**
- Modify: `README.md`
- Modify: `docs/design/baseline-usage.md`
- Create: `docs/notes/traffic-ai-acceptance-2026-10-04.md`
- Test: all `tests/`

- [ ] **Step 1: Run `pytest -q` with the project-local temporary directory and require all tests to pass.
- [ ] **Step 2: Run `compileall` for `src`, `run.py`, and retained legacy code.
- [ ] **Step 3: Run seeded `collect`, `train`, `baseline`, and `diagnose` commands.
- [ ] **Step 4: Inspect generated JSONL traces for at least one follow, lane-change, and collision case.
- [ ] **Step 5: Run the actual UI locally and compare its state/metrics with the corresponding headless trace.
- [ ] **Step 6: Record known limitations and acceptance evidence in the note.
- [ ] **Step 7: Ask the user to run and accept the local version.
- [ ] **Step 8: Only after acceptance, create one grouped commit and request confirmation before pushing GitHub.

### Task 8: Add the discrete short-horizon MPC baseline

**Files:**
- Create: `src/driving_lab/controllers/__init__.py`
- Create: `src/driving_lab/controllers/mpc.py`
- Modify: `src/driving_lab/cli.py`
- Test: `tests/test_mpc_policy.py`

**Interfaces:**
- Add `MPCConfig(horizon=5, candidate_actions=(0, 1, 2, 3, 4), collision_cost, progress_cost, speed_cost, lane_change_cost)`.
- Add `MPCPolicy(env_model, config).act(observation) -> Action`.
- Add `MPCPolicy.last_plan` for diagnostics without exposing renderer internals.

- [ ] **Step 1: Write failing tests** for safe braking selection, deterministic action selection, horizon-one equivalence, and plan recording.
- [ ] **Step 2: Run focused tests and verify the MPC module is absent.
- [ ] **Step 3: Implement a pure NumPy rollout model that does not mutate the live environment.
- [ ] **Step 4: Implement sequence scoring for collision risk, progress, speed tracking, lane-change effort, and terminal safety.
- [ ] **Step 5: Add `--policy mpc` to evaluation and baseline output.
- [ ] **Step 6: Run MPC tests and compare one fixed-seed episode with RulePolicy.
- [ ] **Step 7: Commit locally: `feat: add discrete model predictive policy`.

### Task 9: Add GPU-capable DQN and behavior-cloning initialization

**Files:**
- Create: `src/driving_lab/rl/__init__.py`
- Create: `src/driving_lab/rl/replay.py`
- Create: `src/driving_lab/rl/dqn.py`
- Create: `src/driving_lab/rl/behavior_init.py`
- Create: `src/driving_lab/rl/trainer.py`
- Create: `src/driving_lab/rl/shield.py`
- Create: `experiments/train_dqn.py`
- Modify: `src/driving_lab/cli.py`
- Modify: `requirements.txt`
- Modify: `environment.yml`
- Test: `tests/test_dqn.py`
- Test: `tests/test_rl_smoke.py`

**Interfaces:**
- Add `DQNConfig`, `QNetwork`, `ReplayBuffer`, `DQNAgent`, and `DQNTrainer` with explicit device selection (`cuda` when available, otherwise `cpu`).
- Add `initialize_from_behavior_cloning(agent, model_path) -> None` with shape validation.
- Add `SafetyShieldPolicy(learned_policy, fallback=RulePolicy())` integration that records override events.
- Add CLI commands `train-dqn` and `evaluate-all` with fixed seeds and JSON outputs.

- [ ] **Step 1: Write failing tests** for replay capacity, Q-network shape, checkpoint round-trip, CPU smoke training, BC initialization shape checks, and shield override recording.
- [ ] **Step 2: Run focused tests and verify the RL modules are absent.
- [ ] **Step 3: Implement replay buffer and Q-network with the existing 5-action contract.
- [ ] **Step 4: Implement DQN loss, target network, epsilon schedule, gradient clipping, checkpoint save/load, and device selection.
- [ ] **Step 5: Implement behavior-cloning initialization from the NumPy model only through a documented weight conversion path; reject incompatible models.
- [ ] **Step 6: Implement the trainer and a short CPU smoke run before any long GPU training.
- [ ] **Step 7: Implement shielded action selection and diagnostic override metrics.
- [ ] **Step 8: Add fixed-seed evaluation comparing Rule, BC, MPC, DQN-random-init, DQN-BC-init, and shielded DQN.
- [ ] **Step 9: Run all RL tests and one bounded training smoke run; do not claim performance improvements until repeated evaluation is complete.
- [ ] **Step 10: Commit locally: `feat: add dqn training and safety-shield comparisons`.

### Task 10: Final AI experiment package and resume-ready documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/design/baseline-usage.md`
- Create: `docs/notes/ai-method-comparison-2026-10-04.md`
- Create: `experiments/configs/benchmark.yaml`
- Test: `tests/test_benchmark_reproducibility.py`

- [ ] **Step 1: Write failing tests** for benchmark configuration validation and same-seed metric reproducibility.
- [ ] **Step 2: Implement one benchmark configuration covering Rule, BC, MPC, DQN, and shielded DQN.
- [ ] **Step 3: Run repeated evaluations and save metrics, traces, and model metadata.
- [ ] **Step 4: Document what is actually implemented, GPU/CPU environment, reward definition, limitations, and ablations.
- [ ] **Step 5: Prepare a concise resume-oriented project summary without claiming real-vehicle performance.
- [ ] **Step 6: Let the user run the complete local acceptance package.
- [ ] **Step 7: Only after explicit acceptance, create a grouped release commit and ask before pushing GitHub.
