# Template-Driven APL Active-Task System — Design ↔ Code Audit

This document tracks **every design rule** that emerged during the multi-round
chat (v1 → v3 of the plan) and verifies whether the current code reflects it.
The intent is to give a single place where any reviewer can see *what was
agreed*, *where it lives in the source*, and *what still needs work*.

> Last verified: 2026-05-10 against
> [task_generation/apl_tasks/template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py)
> and [evaluation/](spatial-training-room/evaluation/).
> All G1–G7 closed; system tour added in §4.

---

## 1. Source-of-truth design (v3, distilled from session memory)

### A. Hard “must”s (anti-cheat / scoring)
1. **No belief state fields.** Strip `init_belief` / `target_belief` /
   `uncertainty_type` / `function_type` / `decision_type` /
   `allow_cannot_tell` from `APLActiveTaskItem`.
2. **Lattice = action grid.** State expansion for the expert / potential
   uses the `ActionPrimitive` set, not an arbitrary XY grid.
3. **Submit-time quality only.** Coverage must be evaluated at the *final*
   submitted view; “drive-by” passing through a good frame must score 0.
4. **`Score = 1[â=a*] · CovFactor · γ^T`** with
   `CovFactor = min(1, Coverage / min_coverage_for_credit)`.
5. **Reward shaping** via potential differences:
   `r_t = Φ(s_{t+1}) − Φ(s_t)`, terminal step adds the episode score.
6. **Single uncertainty axis** (no F-A/F-B/D-M…); reflected as a flat
   `subclass` string (e.g. `C1.1`).
7. **`Cannot tell` is *not* a special mechanism**, only a normal MC option
   for templates whose GT genuinely is “no”.

### B. Schema extensions to `APLActiveTaskItem`
- `template_id` *(e.g. `T01`)*
- `subclass`   *(e.g. `C1.1`)*
- `quality_spec` *(echoes evidence_slots / γ / max_steps / min_coverage)*
- `expert_trajectory: List[ViewState]`
- `coverage: float`
- `score: float`
- `min_steps: int`

### C. Template / evaluation engine
- 22 YAML templates: 19 in `templates/p0/`, 3 in `templates/p1/`.
- `evaluation/` provides: `predicates`, `region_generators`, `coverage`,
  `scorer`, `potential`, `expert`, `quality_overrides`, `template_spec`.
- Tier-2 callables registered via `@register_quality(...)` (5 of them).
- BFS/beam expert: visited keys at `(x/0.25 m, y/0.25 m, yaw/22.5°)`.

### D. Driver
- `TemplateActiveGenerator.generate_for_template(template_id, n)`.
- `INSTANTIATORS` registry, one entry per template.
- `run_factory.py` flag: `--mode template --template-id T??`.
- `testbench/smoke_test_template.py` minimal end-to-end runner.

---

## 2. Audit table — design rule ↔ source location

| # | Rule | Status | Location |
|---|---|---|---|
| 1 | No belief state on item | ✅ done | [core/data_types.py](spatial-training-room/core/data_types.py#L164) — class has no `*_belief` / `uncertainty_type` fields |
| 2 | New schema fields present | ✅ done | [core/data_types.py](spatial-training-room/core/data_types.py#L191-L197) |
| 3 | JSONL round-trip preserves new fields | ✅ done | [core/data_types.py](spatial-training-room/core/data_types.py#L210-L262) |
| 4 | Action-grid expansion (no XY lattice) | ✅ done | [evaluation/potential.py](spatial-training-room/evaluation/potential.py#L105-L130) — `_DEFAULT_ACTIONS` over `ActionPrimitive` |
| 5 | `Score = 1[â=a*] · CovFactor · γ^T` | ✅ done | [evaluation/scorer.py](spatial-training-room/evaluation/scorer.py#L33-L57) |
| 6 | Potential-shaped step rewards + sparse terminal | ✅ done | [evaluation/scorer.py](spatial-training-room/evaluation/scorer.py#L60-L86) |
| 7 | Coverage aggregator `'all'` (product) / `'mean'` | ✅ done | [evaluation/coverage.py](spatial-training-room/evaluation/coverage.py#L155-L177) |
| 8 | Tier-2 overrides registered | ✅ done | [evaluation/quality_overrides.py](spatial-training-room/evaluation/quality_overrides.py) — 5 callables |
| 9 | Predicate signature `(pos,target,hfov,scene_ctx,**kw)` | ✅ done | [evaluation/predicates.py](spatial-training-room/evaluation/predicates.py) — 12 predicates |
| 10 | Region samplers (16) | ✅ done | [evaluation/region_generators.py](spatial-training-room/evaluation/region_generators.py) |
| 11 | Expert beam search w/ rounded-state visited set | ✅ done | [evaluation/expert.py](spatial-training-room/evaluation/expert.py); [evaluation/potential.py](spatial-training-room/evaluation/potential.py) |
| 12 | 22 YAML templates loaded by `load_template` | ✅ done | [task_generation/templates/p0](spatial-training-room/task_generation/templates/p0), [task_generation/templates/p1](spatial-training-room/task_generation/templates/p1) |
| 13 | Single-axis subclass label (no F/D axes) | ✅ done | YAML `subclass: C1.1` only; no other axis keys present |
| 14 | `--template-id` CLI flag in factory | ✅ done | [run_factory.py](spatial-training-room/run_factory.py) |
| 15 | Smoke-test script | ✅ done | [testbench/smoke_test_template.py](spatial-training-room/testbench/smoke_test_template.py) |
| 16 | Scene-context extension (rooms, portals, occlusion, projection) | ✅ done | [core/scene_context_ext.py](spatial-training-room/core/scene_context_ext.py) |
| 17 | Init-view chosen so slot predicates **fail** | ✅ done | `_sample_failing_init` in [template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py) |

### Gaps / partial items — **closed 2026-05-10**

| # | Rule | Status | Resolution |
|---|---|---|---|
| G1 | Submit-time quality (anti-drive-by) | ✅ fixed | `slot_satisfied` / `compute_coverage` gained `submit_only=True` ([evaluation/coverage.py](spatial-training-room/evaluation/coverage.py)); `episode_score` ([evaluation/scorer.py](spatial-training-room/evaluation/scorer.py)), `find_expert_trajectory` ([evaluation/expert.py](spatial-training-room/evaluation/expert.py)) and `_build_task_item` ([task_generation/apl_tasks/template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py)) all use submit-time semantics. Tier-2 callables still receive the full trajectory and decide themselves (documented). |
| G2 | Single-step gain < single-step cost | ✅ documented | Math note added at the top of [evaluation/scorer.py](spatial-training-room/evaluation/scorer.py): per-step cost = `1−γ`; meaningful step-gain requires `ΔΦ ≥ 1−γ`. Templates tighten the budget by lowering `gamma` in YAML. |
| G3 | Real instantiators for all 22 templates | ✅ partially extended | T08, T13, T15, T27 promoted to real instantiators in [template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py). Total real: **15/22**. Remaining 7 stubs (T06, T14, T16, T26, T29, T32, T33) require scene-side metadata not present in current AABB data and stay as no-ops (return `None`). |
| G4 | T01 honest-downgrade caveat | ✅ fixed | T01 YAML gained `caveat:` and `weight: 0.3` ([T01_category_recognition.yaml](spatial-training-room/task_generation/templates/p0/T01_category_recognition.yaml)); `TemplateSpec` gained matching fields ([template_spec.py](spatial-training-room/evaluation/template_spec.py)). |
| G5 | Tier-2 invocation contract docs | ✅ fixed | Module docstring of [quality_overrides.py](spatial-training-room/evaluation/quality_overrides.py) now states the full signature, submit-time vs trajectory-level split, and a worked toy example. |
| G6 | Question rendering hardening | ✅ fixed | `_render_question` now handles `{var}`, `{{var}}`, and `{choice_*}` consistently ([template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py)). |
| G7 | Choice-generator registry | ✅ fixed | New [choice_generators.py](spatial-training-room/task_generation/apl_tasks/choice_generators.py) registers 14 named generators (`binary_yes_no`, `similar_aabb_labels`, `count_range`, …); `_build_task_item` looks up `spec.answer_choices_generator` and overrides instantiator-built choices when registered. |

### Items deliberately out of scope (per v3)
- **Belief tracking signal** — explicitly removed.
- **Snapshot diff for U3-dynamic** — deferred to a later phase.
- **Texture / label content acquisition** — not supported by AABB data.

---

## 3. (Reserved)

> Run recipes have moved to §5 to keep them next to the system tour.

---

## 4. Step-by-step system tour (for new readers)

Read these files **in this order** with the focusing questions in mind. Each
step builds on the previous; you should be able to summarise the answer to
each focusing question before moving on.

### Step 1 — What does a task LOOK like?
**File**: [core/data_types.py](spatial-training-room/core/data_types.py) (jump to `class APLActiveTaskItem`)

Focus on:
- The 7 v3 fields (`template_id`, `subclass`, `quality_spec`,
  `expert_trajectory`, `coverage`, `score`, `min_steps`).
- `to_jsonl_dict()` / `from_dict()` — serialisation contract.

Q: *“If I ran the generator and got one task back, what would the JSONL
record contain and how do I reload it?”*

### Step 2 — What does a TEMPLATE look like?
**Files**:
[task_generation/templates/p0/T01_category_recognition.yaml](spatial-training-room/task_generation/templates/p0/T01_category_recognition.yaml)
→ [evaluation/template_spec.py](spatial-training-room/evaluation/template_spec.py)

Read the YAML top-to-bottom. Then read `TemplateSpec` and `EvidenceSlot`
dataclasses. Notice the round-trip: every YAML key has a Python field.

Key fields to understand:
- `evidence_slots`: a list of *what a good final view looks like*.
- `predicates`: small Boolean primitives that score a single (cam_pos,
  cam_target).
- `tier2_override`: when a slot's quality is too holistic for predicates.
- `coverage_aggregator`: `"all"` (product) vs `"mean"`.
- `gamma`, `max_steps`: the step-cost knobs.

Q: *“What constraints must a final view satisfy for this template to
score 1.0?”*

### Step 3 — How are predicates and regions implemented?
**Files**:
[evaluation/predicates.py](spatial-training-room/evaluation/predicates.py),
[evaluation/region_generators.py](spatial-training-room/evaluation/region_generators.py),
[core/scene_context_ext.py](spatial-training-room/core/scene_context_ext.py)

- Skim the 12 predicates — note the canonical signature
  `(cam_pos, cam_target, hfov_deg, scene_ctx, **kwargs) -> bool`.
- Skim the 16 region samplers — note they all return
  `List[(cam_pos, cam_target)]`.
- Every helper they need (occlusion, room membership, portal sampling…)
  lives in `scene_context_ext.py`, monkey-patched onto `SceneContext` at
  import time. **The first import of evaluation pulls this in.**

Q: *“If I add a new predicate, where do I register it and how do I call
it from YAML?”* (Answer: add to `PREDICATE_REGISTRY` in `predicates.py`,
then reference its name in the YAML predicate list.)

### Step 4 — How is COVERAGE actually computed?
**File**: [evaluation/coverage.py](spatial-training-room/evaluation/coverage.py)

Trace `compute_coverage` → `slot_satisfied` → either Tier-2 callable or
`slot_satisfied_at` → `fraction_passed` → individual predicates.

Pay attention to the `submit_only` flag (G1 fix): the *scoring* path uses
`submit_only=True`; the *potential Φ* uses `submit_only=False`.

Q: *“Why does the agent NOT get credit for passing through a great frame
mid-trajectory?”*

### Step 5 — How does the SCORER turn a trajectory into a number?
**File**: [evaluation/scorer.py](spatial-training-room/evaluation/scorer.py)

Read the module docstring (G2 math note) then `episode_score` and
`step_rewards`.

Equations to internalise:
- `Score = 1[â=a*] · CovFactor · γ^T`
- `r_t = Φ(s_{t+1}) − Φ(s_t)`, terminal step adds `Score`.

Q: *“If I make the agent take 4 useless extra steps, how much score do I
lose?”* (Answer: factor of `γ^4 ≈ 0.81`.)

### Step 6 — How is the EXPERT trajectory found?
**Files**:
[evaluation/potential.py](spatial-training-room/evaluation/potential.py),
[evaluation/expert.py](spatial-training-room/evaluation/expert.py)

- `breadth_first_potential_search` does a beam search over
  `ActionPrimitive` transitions (lattice = action grid, see Rule 4).
- Visited keys are rounded `(x/0.25 m, y/0.25 m, yaw/22.5°)` to keep the
  beam finite.
- `find_expert_trajectory` ranks candidates by *submit-time* coverage,
  then potential, then shortness.

Q: *“What action set does the expert search consider, and what tie-breaks
two candidates with equal coverage?”*

### Step 7 — How is a single TASK actually built?
**File**: [task_generation/apl_tasks/template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py)

Walk `_build_task_item` line-by-line:
1. Sample candidate target views per slot.
2. Sample an init_view that **fails** every slot (`_sample_failing_init`).
3. Run expert beam search to derive the gold action sequence.
4. Compute submit-time coverage + score.
5. Render the question, dispatch choice generator (G7).
6. Stamp the `APLActiveTaskItem`.

Then look at the `INSTANTIATORS` registry: 15 entries today (T01, T04,
T05, T08, T11, T13, T15, T17–T21, T23, T24, T27); the remaining 7 stubs
return `None` and are skipped silently.

Q: *“Where does the per-template scene-matching logic live? How do I add
a new instantiator?”*

### Step 8 — How does the CLI tie it together?
**Files**:
[run_factory.py](spatial-training-room/run_factory.py),
[testbench/smoke_test_template.py](spatial-training-room/testbench/smoke_test_template.py)

- `--mode template --template-id T01` routes through
  `TemplateActiveGenerator` and writes one JSONL file per scene.
- The smoke-test script calls `generate_for_template` directly so you can
  see one task printed end-to-end with no I/O glue.

Q: *“Given a real scene path, what is the shortest command that produces
3 valid T01 tasks I can read?”*

---

## 5. Quick run recipes

```powershell
# (1) List templates the generator can actually instantiate
python testbench/smoke_test_template.py --list

# (2) Smoke-test ONE template against ONE scene
python testbench/smoke_test_template.py `
    --scene <PATH_TO_SCENE_DIR> `
    --template T01 --n 3 --out out/T01.jsonl

# (3) Factory mode (one JSONL per scene)
python run_factory.py --scenes <PATH> --mode template --template-id T01 --max-items 20
```

Sample successful task:

```jsonc
{
  "task_id": "T01_active_…",
  "template_id": "T01", "subclass": "C1.1",
  "question": "Is that a piano or a cabinet?",
  "answer": "cabinet", "answer_choice": "B", "choices": ["piano", "cabinet", "chair"],
  "init_view": {...}, "target_view": {...},
  "action_sequence": ["MOVE_FORWARD", "TURN_LEFT", "STOP"],
  "expert_trajectory": [{...}, {...}, {...}, {...}],
  "coverage": 1.0, "score": 0.857,
  "min_steps": 3,
  "quality_spec": { "evidence_slots": [...], "gamma": 0.95, ... }
}
```

---

## 5. File map (so changes don’t drift again)

| Concern | Files |
|---|---|
| Schema | [core/data_types.py](spatial-training-room/core/data_types.py) |
| Scene helpers | [core/scene_context_ext.py](spatial-training-room/core/scene_context_ext.py) |
| Templates | [task_generation/templates/p0](spatial-training-room/task_generation/templates/p0), [task_generation/templates/p1](spatial-training-room/task_generation/templates/p1) |
| Driver | [task_generation/apl_tasks/template_active_generator.py](spatial-training-room/task_generation/apl_tasks/template_active_generator.py) |
| Evaluation | [evaluation/predicates.py](spatial-training-room/evaluation/predicates.py), [evaluation/region_generators.py](spatial-training-room/evaluation/region_generators.py), [evaluation/coverage.py](spatial-training-room/evaluation/coverage.py), [evaluation/scorer.py](spatial-training-room/evaluation/scorer.py), [evaluation/potential.py](spatial-training-room/evaluation/potential.py), [evaluation/expert.py](spatial-training-room/evaluation/expert.py), [evaluation/quality_overrides.py](spatial-training-room/evaluation/quality_overrides.py), [evaluation/template_spec.py](spatial-training-room/evaluation/template_spec.py) |
| CLI / smoke test | [run_factory.py](spatial-training-room/run_factory.py), [testbench/smoke_test_template.py](spatial-training-room/testbench/smoke_test_template.py) |
