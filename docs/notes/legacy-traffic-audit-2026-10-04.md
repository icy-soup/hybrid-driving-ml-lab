# Legacy traffic audit — 2026-10-04

## Scope

Compared the legacy Pygame implementation in `F:\Missions And Materials\神经与规则混合型智驾项目\2_ai_before_refactor_2026-10-02\Main.py` with the current headless `HighwayEnv` and `TrafficWorld`.

## Frozen legacy constants

| Item | Legacy value / rule |
|---|---|
| Window | `GAME_WIDTH=910`, `PANEL_WIDTH=240`, `bg_height=450`, `FPS=60` |
| Lane centers | `tail_y=150`, `base_position_y=50`, therefore `[200, 250, 300]` |
| Ego anchor | `left = GAME_WIDTH / 2 - width / 2`; lane 1 center is `250` |
| Enemy cap | `MAX_ENEMIES=24` |
| Forward spawn | `x=GAME_WIDTH+50` (`960`), random vehicle index `0..5`, random lane `0..2` |
| Forward interval | starts at `800 ms`, lower bound `150 ms`; score decay, density, strategy and speed factors modify it |
| Rear spawn | every `3000 ms` at `x=-150`, probability `0.25 * strategy enemy_interval_factor`, vehicle index from `[0,1,6]` |
| Removal | normal `x <= -400` or `x >= 1310`; slow-speed `x <= -350` or `x >= 1260` |
| Collision rectangle | `Rect(left, bottom - 20, width, 28)` for both ego and traffic |
| Traffic longitudinal motion | `speed = current_base_speed + relative_speed`; then `left += speed` |
| Relative-speed smoothing | move toward target by at most `0.08` per frame |
| Enemy lateral motion | lane centers, `3 px/frame`; target lane set by `EnemyAI` |
| Enemy collision | player collision ends game; traffic-traffic collision marks both crashed unless fully off-screen |

## Legacy control flow

1. Player action is computed and applied.
2. Score is updated from `player.left > enemy.right`.
3. Spawn interval is calculated from score, traffic density, strategy and current speed.
4. Forward spawn and optional rear spawn are performed only while count is below `24`.
5. Every enemy calls `Enemy.move(all_vehicles, base_speed, player)`.
6. Player and traffic collision is checked.
7. Pairwise traffic collision is checked.
8. Rendering uses each vehicle's type/color/state sprite.

## Legacy vehicle construction

`Enemy` chooses a type-specific sprite and color, stores width/height, uses different base relative-speed rules for forward vs rear spawn, and initializes `bottom` from `[200,250,300]`. It keeps independent state for `relative_speed`, `target_bottom`, lane-change phase, warning timer, crash state, and overtaking.

## Current implementation gaps

- `HighwayEnv._make_random_traffic()` creates `4..8` vehicles at random `x` in `[-250,520]`; it does not model the legacy off-screen forward/rear spawn process, interval clocks, vehicle cap, vehicle type or sprite identity.
- Current traffic `y` is derived from `lane * lane_width`, which gives `[0,50,100]` by default. The renderer adapter must map these logical lanes to legacy screen centers `[200,250,300]` without mutating the environment.
- Current `_advance_traffic()` removes vehicles at `detection_distance` (`config` default), not legacy speed-dependent removal margins.
- Current `TrafficWorld` resolves same-lane penetration after integration, but there is no legacy-style spawn safety window or per-vehicle lifecycle record.
- Current `TrafficWorld` stores behavior modes and reasons, but lane changes are represented by a target lane update rather than continuous lateral interpolation at `3 px/frame`.
- Current `VehicleState` does not yet carry legacy vehicle type/color/sprite state.
- Existing spacing tests validate initial gaps and post-step clamping, but do not cover `100 seeds × 600 steps`, spawn/recycle events, or legacy removal boundaries.

## Required next tests before migration

1. Deterministic initial layout: no ego overlap and no same-lane penetration for seeds `0..99`.
2. Lifecycle run: each seed runs at least `600` steps and records first/last appearance, spawn/recycle, minimum gap, TTC and collision events.
3. Spawn contract: forward vehicles start outside the right edge; rear vehicles start at `x=-150`; IDs are unique; count never exceeds `24`.
4. Motion contract: relative longitudinal motion is monotonic according to the legacy sign convention; lateral `y` stays at a lane center or a bounded lane-change interpolation.
5. UI contract: renderer maps logical lane `0/1/2` to `y=200/250/300`, uses stable type/color sprites, and does not mutate an environment snapshot.

## Decision

Do not replace the current UI or random traffic generator yet. Add failing lifecycle/spawn tests first, then implement an adapter or environment-side lifecycle model that preserves the legacy behavior while keeping the headless state deterministic.
