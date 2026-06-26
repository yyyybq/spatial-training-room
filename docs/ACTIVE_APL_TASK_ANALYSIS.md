# Active APL task analysis

This note describes the template-driven Active APL tasks that are relevant for
the current training-data release. T13, T19, T23, T24, T27, and T32 are
intentionally excluded from this release.

## Scoring and trajectory length

`gamma` is not the main knob that makes generated SFT trajectories longer or
shorter. It is a score/reward discount applied after an expert trajectory has
already been found:

`score = correct_answer * coverage_factor * gamma ** num_steps`

For SFT, trajectory length is controlled mainly by:

- the chosen task template and evidence slots,
- `max_steps`,
- `action_config.move_m` and `action_config.turn_deg`,
- how far the initial view is from the required evidence view(s),
- expert-search and waypoint-stitching behavior.

So if the goal is longer, meaningful SFT demonstrations, adjust init/target
sampling and step granularity. Do not rely on `gamma` for that. A lower `gamma`
only marks long trajectories as lower quality; a higher `gamma` makes the
score less sensitive to length.

## Legacy active generator

`task_generation/apl_tasks/active_generator.py` is a legacy generator for simple
visibility/navigation questions. It is kept only for compatibility with
`run_factory.py --mode apl`. The current release story should point readers to
`template_active_generator.py` and the YAML templates.

## Task table

| ID | Task content | Example | Object/scene selection requirements | Initial-view principle | Evidence view / scoring principle | Current assessment |
| --- | --- | --- | --- | --- | --- | --- |
| T01 | Category recognition from a better view. | "Is the central object a piano or cabinet?" | Should choose category-ambiguous objects: similar AABB size/aspect, non-trivial labels, preferably confusing back/side silhouettes. Avoid common/trivial objects. | Target should be partially visible or oblique/far enough that at least one identity predicate fails. | Submit view must satisfy visible, distance band, aspect exposed, centered, and scale band. | Conceptually useful but weak with AABB-only data. Current instantiator is too loose; improve by selecting confusing shape families. |
| T04 | True physical size comparison. | "Which is larger, the stool or side table?" | Two same-room objects with different true volume but not obvious by common sense; prefer semantic/shape-near pairs and reject extreme volume ratios. | Init should make apparent size misleading or incomplete. | Submit view(s) must reveal enough geometry for both objects; score is submit-only. | Strengthened: instantiator now requires same-room, semantically/shape-close candidates, skips fine clutter, and uses both min/max true-volume ratio. |
| T05 | Distance comparison from a subject to two references. | "Is the sofa closer to the lamp or bookshelf?" | Subject and two references in the same room, distinct labels, distance ratio >= 1.3. | Init should not clearly show both required pairs. | Trajectory coverage must include AB evidence and AC evidence; final frame alone need not contain all evidence. | Strong task. Good use of trajectory-mode evidence. |
| T06 | Object-clearance through a bottleneck. | "Can the sofa pass through this doorway?" | Door/portal width in a usable range; movable object with max XY dimension close to portal width. | Init should not be orthogonal to the bottleneck; object anchor remains relevant. | Evidence requires orthogonal close view of each bottleneck and visible left/right gap proxies. `min_coverage=0.8`. | Keep as the object-affordance line: can this specific object pass through this specific bottleneck? |
| T08 | Geometric configuration judgment. | "Are these objects arranged in a line?" | Three or more distinct-label, meaningful objects in one compact local group; avoid tiny decorative clutter and arbitrary cross-room groupings. | Init should show a projected/ambiguous arrangement rather than the clean configuration. | Submit view should break projection ambiguity and show the group clearly. | Strengthened: instantiator now builds compact same-room groups around a seed and rejects tiny/fine objects. |
| T11 | Single vs multiple object disambiguation. | "Is this one object or two separate objects?" | Ideally same or semantically similar label, close distance, similar size/height, and image-gap small from init but separable from another view. | Init should see the group as merged or nearly merged. | Evidence view must reveal separation on image, visible member(s), and proper distance band. | Current implementation is too loose: it randomly groups objects and does not guarantee semantic/size/distance similarity. Needs tighter pair/group selection. |
| T13 | Post-occlusion continuation. | "Is the object behind the occluder continuous with the front object?" | Needs a credible front/back occlusion pair plus real continuous/separate labels. | Init sees an occlusion relation, not enough to know continuation. | Evidence view moves around occluder to inspect hidden part. | Excluded from this release: AABB-only data cannot prove continuity, and current GT defaults to `continuous`. |
| T17 | Hidden-region existence. | "Is there a chair behind the cabinet?" | Occluder-target pair where target is hidden from an init-like viewpoint; no-case labels should be plausible for the same room and hidden region. | Init sees occluder/region but target is invisible or highly occluded. It should not provide an unobstructed prior view of the target. | Evidence view clears the sightline behind/around occluder; tier-2 checks hidden region and target if GT=yes. | Strengthened: no-case decoys are now sampled from same-room shape/semantic-plausible labels rather than arbitrary scene labels. |
| T18 | Hidden-object category. | "What type of object is hidden behind the wardrobe?" | Same as T17, but target category must become visually identifiable; distractors should be shape/semantic-confusable. | Init target invisible or heavily occluded by occluder. | Evidence view requires stricter target visibility, low occlusion, aspect exposed, centered, scale band. | Strengthened: distractors now reuse the T01 shape/semantic-confusable selection instead of arbitrary other labels. |
| T20 | Local bearing search. | "Find the target around you." | Target in same room with at least one same-room distractor; prefer shape/semantic-near distractors when available. | Init may keep a weak anchor, but must not expose 2+ corners or a clearly answerable target view. | Evidence view centers and reveals the target. | Strengthened: init validator now hard-rejects clear/2-corner target views; instantiator avoids singleton rooms and prefers local plausible distractors. |
| T21 | Local nearest-target comparison. | "Which is closer to the reference, A or B?" | Reference plus two distinct-label candidates in same room; distance difference >= 0.4 m and distance ratio >= 1.25. | Init should not expose both candidate-reference relations clearly. | Evidence must show the relevant pairs/reference relation. | Strengthened: instantiator and YAML now require both absolute and relative distance separation. |
| T24 | Portal direction from current pose. | "Is the door to the bedroom left or right?" | Door portal connecting rooms; room names inferred from objects. | Init orientation matters; portal should be relevant but not trivially centered if possible. | GT is recomputed after init is selected; evidence checks portal visibility/direction. | Excluded from this release: room-name inference is too noisy without explicit room labels. |
| T26 | Occluded local counting. | "How many chairs are around the table, including hidden ones?" | Same-label group with at least three instances, compact local cluster, and a nearby sizable occluder. At least one instance should be hidden at init. | Init sees occluder/cluster but misses at least one instance. | Trajectory should reveal hidden instances around the occluder; count includes initially hidden ones. | Strengthened: instantiator now selects compact clusters, a larger nearby occluder, and multiple hidden-instance evidence slots. Still needs data audit on real scenes. |
| T29 | Contact/proximity relationship. | "Is the lamp resting on the table or beside it?" | Two AABBs with plausible support/contact labels plus geometric support/adjacency checks. | Init should view along an ambiguous axis so vertical/contact relation is unclear. | Lateral evidence view must show pair visibility, orthogonal relation, distance band, and image separation. | Strengthened: instantiator now restricts support surfaces, supported objects, beside labels, support area, and near-contact gap; rerun beam audit before release. |
| T33 | Passage-passability for an agent/reference width. | "Can a wheelchair fit through this doorway?" | Door/passage width near an agent/reference threshold. No movable target object needed. | Init should not be orthogonal enough to read width directly. | Submit view must be close and orthogonal to passage plane with visible separation. | Keep as the passage-affordance line: is this passage wide enough for a reference agent/type? Distinct from T06's object-clearance story. |

## Recommended exclusions for this release

| ID | Reason |
| --- | --- |
| T14 | Requires face/orientation annotations such as `front_normal`. |
| T15 | Requires label-side/text annotations. |
| T16 | Requires richer front/back visual annotations. |
| T13 | Current implementation has unbalanced GT (`continuous` only); should become metadata-gated. |
| T19 | Current implementation has unbalanced GT (`complete` only); should become metadata-gated. |
| T23 | Excluded by design for this release. |
| T24 | Excluded by design for this release because room-name inference is noisy without explicit room labels. |
| T27 | Excluded by design for this release. |
| T32 | Excluded by design for this release. |

## Key implementation gaps to fix before a stronger public release

- T01: choose confusing categories by shape family, not arbitrary objects.
  Current instantiator now requires shape/semantic-confusable distractors.
- T11: choose near, similar-size, same/similar-label groups; reject random
  unrelated groups. Current instantiator now uses same-room, close, shape/label
  similar pairs.
- T13/T19: require real labels or synthetic controlled variants rather than
  constant GT.
- T26: explicitly enumerate hidden instances, prove at least one is hidden at
  init, and require evidence slots for all hidden instances counted in GT.
  Current instantiator now creates multiple hidden-instance slots, but still
  needs a real-scene audit to verify init occlusion.
- T06/T33: keep both only under distinct names: object-clearance vs
  passage-passability.

## Remaining object-selection risks

| ID | Risk | Suggested tightening |
| --- | --- | --- |
| T04 | Pair quality still depends on init apparent-size validator finding a misleading view. | Audit generated samples for apparent-size ambiguity and tune max volume ratio if too many tasks become empty. |
| T08 | Compact groups can still be semantically loose if scene labels are noisy. | Add optional scene-specific semantic family maps if the dataset exposes cleaner categories. |
| T13 | AABB-only pairs do not prove continuation vs separation. | Excluded until synthetic labels or geometric continuity annotations are added. |
| T17 | No-case plausibility now depends on available same-room confusable labels. | If scenes produce too few no-cases, add room-type priors for absent labels. |
| T18 | Shape-confusable distractors can reduce yield in sparse scenes. | Keep yield audit; fallback only to documented metadata-gated mode if needed. |
| T20 | Current design permits 0-1 weak visible corner as an anchor, not a fully unseen target. | This is intentional for local search; `<2 corners` is enforced as a hard reject for answerable init views. |
| T21 | Ratio threshold can reduce yield in small rooms. | Tune `min_dist_ratio` after sweep; keep both absolute and ratio thresholds. |
| T24 | Room names inferred from objects can be noisy. | Excluded until explicit room labels or high-confidence room-name metadata are available. |
| T29 | Whitelists may miss valid long-tail contacts. | Expand support/contact label sets based on audited real-scene failures; rerun beam-sensitivity audit. |
