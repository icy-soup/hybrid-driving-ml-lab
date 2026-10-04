# Traffic AI audit — 2026-10-04

This note freezes the known gap between the legacy Pygame traffic logic and the
canonical headless environment. It is an audit record, not an instruction to
copy legacy code.

## Findings

- The canonical environment previously moved traffic using only relative `x`
  and speed. Traffic vehicles had no independent decision step, continuous
  lateral state, dimensions, behavior mode, or collision cause.
- The legacy `EnemyAI` contains useful ideas (driver profiles, following,
  emergency braking, and lane-change phases), but its perception is based on
  sprite-relative positions rather than bumper geometry. It also mixes speed
  units/sign conventions and uses global lane-change configuration.
- Spawn spacing and lane-change rear conflicts were not represented as a
  shared invariant. A collision was a boolean side effect, not an event with
  participants and a reason.

## Canonical direction

The new tests use deterministic `VehicleState` fixtures and keep the existing
action numbering unchanged. Later tasks will add geometry and perception,
independent traffic drivers, typed collision events, and a renderer that reads
the environment snapshot without advancing simulation state. The paper
reference in `同车行.../full.md` informs sensor regions and lane-change timing;
no legacy module is imported by these tests.
