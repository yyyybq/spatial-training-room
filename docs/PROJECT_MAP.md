# Project map

Spatial Training Room is a data factory for spatial reasoning tasks in
reconstructed indoor 3D scenes. The project can be read as three connected
levels of supervision:

1. Static QA: answer from one view.
2. Passive APL: follow an instruction to move to a target view.
3. Active APL: move because the current view is insufficient to answer a
   question.

## Core scene and data model

| Path | Role |
| --- | --- |
| `core/data_types.py` | JSONL task schemas: `QATaskItem`, `APLPassiveTaskItem`, `APLActiveTaskItem`, and `ViewState`. |
| `core/scene_context.py` | Loads scene geometry, object AABBs, room polygons, walls, camera intrinsics, and visibility helpers. |
| `core/scene_context_ext.py` | Adds room IDs, portals, object centers, occlusion/projection helpers, and other scene queries used by templates. |
| `core/task_base.py` | Shared generator base and JSONL save/load utilities. |
| `configs/` | Default generation configs for QA and APL. |

## Action and trajectory layer

| Path | Role |
| --- | --- |
| `action_space/action_primitives.py` | Discrete move, turn, look, and stop actions plus action configuration. |
| `action_space/action_executor.py` | Applies one primitive action to a `ViewState`. |
| `action_space/action_sequences.py` | Goal-directed planning and sequence validation for passive/legacy active tasks. |
| `evaluation/expert.py` | Expert trajectory search for template-driven active APL, including waypoint stitching and beam sensitivity diagnostics. |
| `evaluation/trajectory_metrics.py` | Step efficiency, SPL, information gain, and counterfactual regret metrics. |

## Static QA generation

| Path | Role |
| --- | --- |
| `task_generation/qa_tasks/qa_generator.py` | Thin wrapper that exposes the legacy batch QA pipeline as typed `QATaskItem` objects. |
| `bench_generation/qa_batch_generator.py` | Legacy batch QA pipeline: samples views, computes visible objects, and emits single-view MCQ items. |
| `sampler/question_generator/` | Standalone single-view question generators such as counting, nearest object, relative position, distance, rotation, and navigation-action MCQ. |
| `bench_generation/camera_generation.py` | Camera/view sampling utilities used by older QA generation paths. |
| `bench_generation/preview.py` | Optional rendering and review image helpers; depends on heavier render stack. |

Static QA asks: given this view, what can be inferred now? It teaches the
visual/spatial semantics of one frame: visibility, count, distance, nearest
object, relative direction, and simple view-local navigation facts.

## Passive APL generation

| Path | Role |
| --- | --- |
| `task_generation/apl_tasks/passive_generator.py` | Instruction-following tasks: move to a distance, face an object, or stand on a side of an object. |
| `task_generation/apl_tasks/apl_types.py` | Task type constants, natural-language templates, and difficulty helpers. |

Passive APL asks: given an instruction, can the agent execute the right action
sequence? It turns scene geometry into supervised navigation demonstrations:
`instruction -> init_view -> action_sequence -> target_view`.

## Active APL generation

| Path | Role |
| --- | --- |
| `task_generation/apl_tasks/active_generator.py` | Legacy active tasks: question-driven navigation for visibility and spatial relationship questions. |
| `task_generation/apl_tasks/template_active_generator.py` | Current template-driven active APL generator. Instantiates YAML templates, samples init/target views, runs expert search, computes coverage/score, and writes `APLActiveTaskItem`. |
| `task_generation/templates/p0/` | P0 active templates. |
| `task_generation/templates/p1/` | P1 active templates that need richer annotations. |
| `task_generation/apl_tasks/choice_generators.py` | Registered multiple-choice option builders. |
| `task_generation/apl_tasks/init_validators.py` | Template-specific checks that keep initial views non-answerable. |

Active APL asks: the question cannot be answered from the initial view, so what
should the agent do to gather enough evidence? It is the main release story for
active perception: `question + init_view -> expert actions -> evidence view(s)
-> answer`.

## Template evaluation layer

| Path | Role |
| --- | --- |
| `evaluation/template_spec.py` | Loads YAML templates into `TemplateSpec` and expands pattern-based evidence slots. |
| `evaluation/predicates.py` | View-level predicates such as visibility, pair visibility, centeredness, portal visibility, room membership, and contact checks. |
| `evaluation/region_generators.py` | Samples candidate camera poses for each evidence slot. |
| `evaluation/coverage.py` | Computes submit-time or trajectory-level evidence coverage. |
| `evaluation/scorer.py` | Episode score: correct answer times coverage factor times step penalty. |
| `evaluation/potential.py` | Potential function used for expert search and reward shaping. |
| `evaluation/quality_overrides.py` | Tier-2 quality checks for templates that need holistic evidence logic. |

## Testbench and release utilities

| Path | Role |
| --- | --- |
| `testbench/test_regression_core.py` | Core regression tests for registries, scoring, coverage semantics, and trajectory metrics. |
| `testbench/test_items_6_and_7.py` | Template schema projection and handler inventory checks. |
| `testbench/smoke_test_template.py` | Minimal one-template generation smoke test. |
| `testbench/sweep_all_templates.py` | Runs registered templates against one scene and reports produced/empty/error/timeout. |
| `testbench/audit_trajectory_reliability.py` | Audits generated JSONL for coverage-mode consistency, beam sensitivity, and unusual motion patterns. |
| `testbench/visualize_tasks.py`, `trace_T05.py`, `render_real_tasks.py` | Debugging and visual review tools. |
| `docs/ACTIVE_APL_TASK_ANALYSIS.md` | Release-facing analysis of each Active APL task, including object-selection requirements and scoring/init-view rationale. |
| `docs/TEMPLATE_SUPPORT_MATRIX.md` | Template support/exclusion status for the current release. |

## Output and local artifacts

| Path | Role |
| --- | --- |
| `out/` | Generated JSONL, rendered previews, review HTML, and experiment outputs. Ignored by `.gitignore`; should not be pushed as source. |
| `*.log` | Local debug logs. Ignored by `.gitignore`; should not be pushed as source. |
| `*.egg-info/`, `__pycache__/` | Local packaging/cache artifacts. Ignored by `.gitignore`; should not be pushed as source. |

## How QA, passive APL, and active APL fit together

The clean release narrative is a curriculum:

- Static QA teaches what can be answered from a single observation.
- Passive APL teaches how actions change observations under explicit
  navigation instructions.
- Active APL combines both: the agent receives a question, recognizes that the
  initial observation is insufficient, navigates to collect evidence, then
  answers.

In this framing, QA is not a side project. It is the single-view baseline and
the perception vocabulary. Passive APL is the action grounding layer. Template
active APL is the flagship task family that tests deliberate evidence
acquisition.
