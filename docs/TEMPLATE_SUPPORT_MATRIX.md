# Template support matrix

This matrix records the expected release status of the template-driven APL
task system. "Supported" means the template has an instantiator and can produce
tasks when the scene has the required geometry/metadata.

| Template | Task family | v0.1 status | Notes |
| --- | --- | --- | --- |
| T01 | Category recognition | Supported | Weighted lower because category labels can be visually ambiguous. |
| T04 | True size comparison | Supported | Requires same-room, shape/semantic-near objects with non-trivial but non-extreme volume ratio. |
| T05 | True distance comparison | Supported | Trajectory-mode evidence can span multiple pair views. |
| T06 | Clearance assessment | Supported | Requires portal/passage geometry and a movable object. |
| T08 | Configuration judgment | Supported | Requires a compact same-room group of meaningful, distinct-label objects. |
| T11 | Single vs multiple | Supported | Includes single and multiple cases. |
| T13 | Post-occlusion continuation | Excluded from current release | Dropped because AABB-only data cannot provide balanced continuous/separate labels. |
| T14 | Back/front face acquisition | Metadata-gated | Requires per-object orientation metadata such as `front_normal`. |
| T15 | Label-face acquisition | Metadata-gated | Requires `label_side` or equivalent per-object annotation. |
| T16 | Front/back difference | Metadata-gated | Requires richer face/orientation annotations. |
| T17 | Post-occlusion existence | Supported | Includes yes/no hidden-region cases with plausible same-room no-case decoys. |
| T18 | Post-occlusion category | Supported | Requires a recoverable hidden target and shape/semantic-confusable distractors. |
| T19 | Post-occlusion completeness | Excluded from current release | Dropped because current data lacks balanced completeness/damage annotations. |
| T20 | Local bearing search | Supported | Init hard-rejects clear target views; allows at most a weak visual anchor before search. |
| T21 | Local nearest target | Supported | Requires two candidate targets with absolute and relative distance asymmetry. |
| T23 | Cross-room existence | Excluded from current release | Dropped from this training-data version to keep the task set focused on object-level active perception. |
| T24 | Portal direction | Excluded from current release | Dropped because room-name inference is noisy without explicit room labels. |
| T26 | Occluded counting | Supported | Trajectory-mode; requires countable same-label objects and occlusion. |
| T27 | Zone counting | Excluded from current release | Dropped from this training-data version to avoid trajectory-memory counting ambiguity. |
| T29 | Contact relationship | Supported with caveat | Uses plausible support/contact labels and geometry; rerun beam audit before public release. |
| T32 | Connectivity judgment | Excluded from current release | Dropped from this training-data version; it is closer to room-graph reasoning than visual evidence acquisition. |
| T33 | Passage passability | Supported | Requires passage width and reference object/agent-width proxy. |

## Coverage-mode notes

- `submit` mode: only the final submitted frame is scored for evidence.
- `trajectory` mode: evidence can be collected across the trajectory when no
  single final view can contain all required evidence.
- Release documentation and downstream evaluation scripts must preserve this
  distinction for any trajectory-mode templates that remain in a run.

## Current local verification snapshot

- Core regression suite: 7 tests passed.
- Template handler/schema checks: 22 handlers loaded, 19 p0 YAML templates
  loaded.
- Historical local JSONL audit: 25 items checked, with warnings for the now
  excluded T27 trajectory-memory final-frame coverage and one T29
  beam-sensitivity case. Re-audit after the tightened T29 selector.
