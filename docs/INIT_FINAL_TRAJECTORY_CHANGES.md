# Init / Final / Trajectory Change Record

Date: 2026-06-13

## Why this change exists

The previous anchor-init pass fixed the most serious issue: an initial view
should not be rewarded for making the question target invisible. A good init
view should keep the task visually grounded while still preventing immediate
submission.

This follow-up tightens three remaining areas:

1. Some init validators still had old "invisible target" branches after early
   returns, so the source code no longer matched the intended semantics.
2. `coverage_mode: trajectory` templates were storing the final trajectory
   frame as `target_view`, while their score was computed from evidence
   accumulated across the trajectory. That made `target_view` look like a
   submit-only answer view even when it was not.
3. The expert path was produced by one narrow beam-search run. That is useful
   for generation, but it is weaker than a navigation-style ground-truth path.

## Init-view changes

- `target_invisible_at_init` is now treated as a legacy field name with
  visible-anchor semantics:
  the target must be relaxed-visible, but the init sampler still rejects views
  that satisfy all evidence slots.
- `portal_invisible_at_init` is similarly overridden to require a visible
  portal anchor instead of rewarding an off-screen portal.
- `target_invisible_from_room_a` is overridden for T23:
  the camera must start in `room_a`, must not already see the room-B target
  well enough to answer, and should face the connecting portal.
- T23 instantiation now chooses a real DOOR portal and records
  `portal_0_id` / `portal_0_proxy_id`.
- T23 `init_view.visible_anchors` now points to `portal_0_id`, not
  `target_id`.

## Final-view and coverage changes

Generated items now expose three coverage fields:

- `coverage`: the authoritative score coverage, preserving template
  `coverage_mode`.
- `submit_view_coverage`: coverage computed on the final frame only.
- `trajectory_evidence_coverage`: coverage computed over the full trajectory.

For `coverage_mode: submit`, `coverage == submit_view_coverage`.
For `coverage_mode: trajectory`, `coverage == trajectory_evidence_coverage`,
and `submit_view_coverage` may be lower by design.

This makes the distinction explicit:

- submit-final templates expect the final frame to be answerable.
- trajectory-memory templates allow evidence gathered over multiple views.

## Trajectory reliability changes

Generation now calls `find_robust_expert_trajectory`, which runs two
deterministic beam/depth budgets by default and selects the best result by:

1. highest real coverage,
2. shortest view sequence,
3. highest potential.

Each item records `trajectory_reliability`, including:

- selected beam/depth config,
- per-config coverage and step count,
- whether every config found full coverage with the same length,
- coverage and full-coverage step ranges.

This is not a proof of global optimality. It is a sensitivity audit that makes
beam fragility visible and usually avoids choosing a path that only the
original narrow beam happened to find.

## Audit script

`testbench/audit_trajectory_reliability.py` reads generated jsonl files and
flags:

- submit-mode items whose final frame does not meet minimum coverage,
- trajectory-mode items whose full evidence coverage is insufficient,
- trajectory-memory items whose final frame alone is not answerable,
- unstable beam sensitivity,
- trajectories dominated by strafe/backward motion.

Use:

```bash
python testbench/audit_trajectory_reliability.py out/batch_anchor_init
```

## Sweep timeout guard

`testbench/sweep_all_templates.py` now supports a per-template timeout:

```bash
python testbench/sweep_all_templates.py \
    --scene C:/Users/user/Desktop/0267_840790 \
    --n 1 \
    --out out/batch_anchor_init_v2_timeout_sweep_n1 \
    --timeout-per-template 180
```

Each template runs in an isolated worker process when the timeout is enabled.
If one template exceeds the budget, the worker is terminated, the row is marked
`timeout`, and the sweep continues to later templates. When `--out` is provided,
the script also writes `sweep_summary.json`.

The first full-template diagnostic sweep with `n=1` and a 180s timeout produced:

- ok: 16 templates
- empty: T14, T15, T16 (scene annotations unavailable)
- timeout: T05, T06, T27
- errors/stubs: 0

The output batch is `out/batch_anchor_init_v2_timeout_sweep_n1`.

## Waypoint-stitch expert trajectories

`evaluation/expert.py` now tries a lightweight waypoint-stitch expert before
beam search:

1. expand evidence slots,
2. sample candidate views from each slot's `region_generator`,
3. keep candidates that satisfy the slot,
4. order the evidence waypoints by a short nearest-route heuristic,
5. linearly interpolate valid camera positions between `init_view` and the
   waypoints,
6. accept only if `compute_coverage` passes under the template's own
   `coverage_mode`.

If stitching succeeds, `trajectory_reliability.method` is `waypoint_stitch`.
If it fails for most trajectory-memory templates, a short beam fallback may be
used. For the previously slow T05/T06/T27 templates, failed stitched attempts
return quickly so the generator can sample a new instance instead of spending
minutes in full robust beam search.

T27 has a dedicated `t27_target_scan` stitch variant: it creates waypoints for
the concrete target instances inside each zone, so zone-counting evidence is
accumulated over the trajectory instead of requiring one room-level frame to see
all targets at once.

Clean example outputs:

- `out/batch_waypoint_stitch_examples_clean_n1`: 19 usable templates x 1 item;
  T14/T15/T16 are empty because this scene lacks the required annotations.
- `out/batch_waypoint_stitch_key_templates_clean_n3`: T05/T06/T27 x 3 items.

Audit passed for coverage. T27 still reports that the final frame alone is not
answerable, which is expected for trajectory-memory counting: the full
trajectory evidence coverage is 1.0.
