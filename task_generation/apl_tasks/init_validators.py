"""
task_generation/apl_tasks/init_validators.py

YAML ``trigger:`` field -> init-view validator dispatch table.

Why this file exists
====================
Many YAML templates declare init-view constraints in their ``trigger:`` block
(e.g. ``max_init_back_angle_deg: 60``, ``init_apparent_size_ratio_min: 0.7``,
``target_invisible_at_init: true``).  Historically NONE of these fields were
read by Python; the only enforced rule was the generic "submit would fail
right now" check inside ``_sample_failing_init``.  As a result init views
often satisfied the letter of "predicate fails" while violating the spirit
("the question must actually be hard to answer from this view").

This module establishes a single source of truth: the YAML field name itself
becomes the dispatch key.  Per-template validators register against the field
names their templates declare, and ``score_init_view`` walks the spec.trigger
dict at runtime, summing each validator's contribution.

Validator contract
==================
A validator is::

    fn(value, spec, task_instance, cp, ct, hfov, scene_ctx) -> float

It returns a non-negative ``preference score`` (typically in ``[0, 1]``) —
higher means "this candidate fits the template's design intent better".  A
validator may also return ``float('-inf')`` to **hard-reject** a candidate
when the template's strict invariant is violated (e.g. ``target_invisible``
is declared but the target is in fact visible).  ``-inf`` is treated as an
internal sentinel; callers see a clean ``(passes, preference)`` tuple via
``score_init_view`` instead.

Public dispatch API
-------------------
``score_init_view`` returns ``(passes, preference, breakdown)``:

    passes      : bool     — False iff any validator hard-rejected.
    preference  : float    — Sum of per-field preference scores (0.0 if rejected).
    breakdown   : dict     — Per-field contributions, useful for diagnostics.

If no validator is registered for a trigger field, it is silently ignored
(but logged at DEBUG once).  Fields known to be consumed elsewhere (e.g. at
instantiation time, not at init-view selection) are listed in
``_NON_INIT_TRIGGER_FIELDS`` so they are skipped without warning.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Tuple

LOGGER = logging.getLogger(__name__)

# (value, spec, task_instance, cp, ct, hfov, scene_ctx) -> float
InitValidator = Callable[..., float]

INIT_VALIDATORS: Dict[str, InitValidator] = {}


# Trigger fields that are NOT init-view constraints — consumed by instantiator
# logic, scene_requirement matching, or pure documentation.  Listing them here
# silences the "no validator registered" debug-log noise.
_NON_INIT_TRIGGER_FIELDS = {
    # Instantiation-time geometry constraints
    "min_dist_ratio",            # T05 / T21 — referenced in _instantiate_T05
    "min_dist_difference_m",     # T21
    "min_true_volume_ratio",     # T04
    "max_true_volume_ratio",     # T04
    "group_size_min",            # T08 — group size picked at instantiation
    "group_size_max",            # T08
    "min_true_instance_count",   # T11 — instance count enforced by instantiator
    "min_total_count",           # T26
    "min_zones",                 # T27
    "zones_mutually_non_visible",  # T27 — scene-level invariant, not init pose
    "min_clearance_margin_m",    # T06
    "max_clearance_margin_m",    # T06
    "min_passage_width_m",       # T33
    "max_passage_width_m",       # T33
    "rooms_count_min",           # T23
    # Question-rendering metadata (consumed by question generator, not init)
    "bearing_hint_accuracy_deg", # T20 — hint sector half-width for hint text
}

# One-shot warning bookkeeping
_WARNED_FIELDS: set = set()


def register_init_validator(field_name: str):
    """Decorator: bind a YAML trigger field name to a scoring function."""
    def deco(fn: InitValidator) -> InitValidator:
        if field_name in INIT_VALIDATORS:
            LOGGER.warning("Re-registering init validator for %r", field_name)
        INIT_VALIDATORS[field_name] = fn
        return fn
    return deco


def score_init_view(
    spec,
    task_instance: Dict[str, Any],
    cp,
    ct,
    hfov: float,
    scene_ctx,
) -> Tuple[bool, float, Dict[str, float]]:
    """Evaluate every registered trigger field against this candidate view.

    Returns
    -------
    passes : bool
        ``True`` iff no validator hard-rejected the candidate.
    preference : float
        Sum of per-field preference scores.  Always ``0.0`` when ``passes`` is
        ``False`` — callers should branch on ``passes`` first.
    breakdown : dict[str, float]
        Per-field score, useful for diagnostics / tracing.  A hard-rejecting
        field is recorded as ``-inf`` in the breakdown.
    """
    trigger = getattr(spec, "trigger", None) or {}
    if not isinstance(trigger, dict):
        return True, 0.0, {}

    total = 0.0
    breakdown: Dict[str, float] = {}
    for field, value in trigger.items():
        if field in _NON_INIT_TRIGGER_FIELDS:
            continue
        fn = INIT_VALIDATORS.get(field)
        if fn is None:
            if field not in _WARNED_FIELDS:
                LOGGER.debug(
                    "No init validator registered for trigger field %r; ignored "
                    "(template=%s).  Add one via @register_init_validator(%r).",
                    field, getattr(spec, "template_id", "?"), field,
                )
                _WARNED_FIELDS.add(field)
            continue
        try:
            contrib = float(fn(value, spec, task_instance, cp, ct, hfov, scene_ctx))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "Init validator %r raised %s; treating as 0.0", field, exc,
            )
            contrib = 0.0
        breakdown[field] = contrib
        if contrib == float("-inf"):
            return False, 0.0, breakdown
        total += contrib
    return True, total, breakdown


# ---------------------------------------------------------------------------
# Registered validators
# ---------------------------------------------------------------------------
#
# Design notes
# ------------
# Each validator is keyed by the YAML trigger field name (not template id), so
# the same field declared by multiple templates reuses the same logic. Each
# validator must read what it needs from ``task_instance`` defensively (keys
# vary across templates) and must NEVER raise — return 0.0 on any uncertainty.
#
# Score conventions:
#   * 1.0 — ideal init: question is meaningful AND non-trivial AND visually anchored
#   * 0.5-0.7 — acceptable: non-trivial but partial anchoring
#   * 0.1-0.3 — last resort: non-trivial but agent has weak context (facing wall)
#   * 0.0 — neutral: field present but no information gained
#   * -inf — hard reject: the template's strict invariant is violated
# ---------------------------------------------------------------------------

import math
import numpy as np


def _forward_xy(cp, ct):
    """2D camera forward vector in world XY plane (ignores Z)."""
    f = np.asarray(ct, dtype=float) - np.asarray(cp, dtype=float)
    f[2] = 0.0
    n = float(np.linalg.norm(f))
    return f, n


def _bearing_to_object_deg(cp, ct, ctx, obj_id):
    """Absolute angle (deg) between camera forward and direction-to-object,
    in the world XY plane. 0° = directly ahead, 180° = directly behind.
    Returns ``None`` if the object centre is unavailable or the camera is
    degenerate."""
    centre = ctx.get_object_centre(obj_id)
    if centre is None:
        return None
    f, fn = _forward_xy(cp, ct)
    d = np.asarray(centre, dtype=float) - np.asarray(cp, dtype=float)
    d[2] = 0.0
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return None
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))


# -------------------------------------------------------------------
# T20 (Local Bearing Search) — target_invisible_at_init
# -------------------------------------------------------------------
# Strict: if the target is fully visible from the init view, the agent
# could answer immediately and the task is trivial → hard reject.
# Soft: even when invisible, the QUALITY of the init varies:
#   * target roughly to the SIDE (60°-120° off-forward) → ideal
#     ("turn ~90° and you'll find it" — the canonical bearing-search task)
#   * target ahead-but-invisible (occluded) → suspicious / weak signal
#   * target directly behind → trivial 180° turn, no meaningful search
# -------------------------------------------------------------------

@register_init_validator("target_invisible_at_init")
def _v_target_invisible_at_init(value, spec, task_instance, cp, ct, hfov, scene_ctx):
    # YAML may declare ``target_invisible_at_init: false`` for a few templates
    # that explicitly want the target visible at init.  Honour that.
    want_invisible = bool(value)
    # Try common key names; fall back to whatever the instantiator produced.
    tgt = (task_instance.get("target_id")
           or task_instance.get("subject_id")
           or task_instance.get("query_id"))
    if not tgt:
        return 0.0
    # Lazy import to avoid pulling evaluation.* during module load.
    from ...evaluation.predicates import Visible

    relaxed_visible = Visible(cp, ct, hfov, scene_ctx,
                              obj=tgt, min_corners=1, max_occ=0.95)
    answer_visible = Visible(cp, ct, hfov, scene_ctx,
                             obj=tgt, min_corners=4, max_occ=0.30)
    if not relaxed_visible:
        return float("-inf")
    if answer_visible:
        return 0.2
    ang = _bearing_to_object_deg(cp, ct, scene_ctx, tgt)
    if ang is None:
        return 0.6
    if ang <= 20.0:
        return 1.0
    if ang <= 35.0:
        return 0.7
    if ang <= 50.0:
        return 0.4
    return 0.1

    is_visible = Visible(cp, ct, hfov, scene_ctx,
                         obj=tgt, min_corners=4, max_occ=0.30)
    if want_invisible and is_visible:
        return float("-inf")
    if (not want_invisible) and (not is_visible):
        return float("-inf")
    # From here on the strict invariant is satisfied — compute soft reward.
    ang = _bearing_to_object_deg(cp, ct, scene_ctx, tgt)
    if ang is None:
        return float("-inf")
    # Sweet spot: target is to the side (bearing-search task is meaningful)
    if 60.0 <= ang <= 120.0:
        return 1.0
    if 45.0 <= ang <= 135.0:
        return 0.7
    if 30.0 <= ang <= 150.0:
        return 0.4
    # Either nearly directly ahead (occluded — weak / suspicious) or nearly
    # directly behind (trivial 180° turn).
    return 0.1


# -------------------------------------------------------------------
# T05 (Actual Distance Comparison) — max_both_pairs_visible_at_init
# -------------------------------------------------------------------
# Strict: if BOTH (subject, ref_b) and (subject, ref_c) are simultaneously
# fully visible at init, the agent can answer by AABB-eyeballing without
# moving → trivial → hard reject.
# Soft: the most informative inits show the SUBJECT clearly so the agent
# knows what the question is about, while at most ONE of the two reference
# pairs is satisfied — creating a clear "now I need to find the other ref"
# tension.  Views facing a wall (no in-frame projection of any of the three
# objects) get a small positive but uninformative score.
# -------------------------------------------------------------------

@register_init_validator("max_both_pairs_visible_at_init")
def _v_max_both_pairs_visible_at_init(value, spec, task_instance, cp, ct, hfov, scene_ctx):
    # YAML declares this as ``max_both_pairs_visible_at_init: false`` to mean
    # "it is NOT allowed for both pairs to be visible".  If the template
    # author flips it to True, this validator becomes a no-op.
    if bool(value) is True:
        return 0.0
    subj = task_instance.get("subject_id")
    ref_b = task_instance.get("ref_b_id")
    ref_c = task_instance.get("ref_c_id")
    if not subj or not ref_b or not ref_c:
        return 0.0
    from ...evaluation.predicates import PairVisible, Visible

    pv_ab = PairVisible(cp, ct, hfov, scene_ctx,
                        obj_a=subj, obj_b=ref_b, min_corners=4, max_occ=0.30)
    pv_ac = PairVisible(cp, ct, hfov, scene_ctx,
                        obj_a=subj, obj_b=ref_c, min_corners=4, max_occ=0.30)
    if pv_ab and pv_ac:
        return float("-inf")

    subj_visible = Visible(cp, ct, hfov, scene_ctx,
                           obj=subj, min_corners=3, max_occ=0.50)
    if subj_visible and (pv_ab or pv_ac):
        # Best: agent sees A + exactly one reference; must navigate to find the other.
        return 1.0
    if subj_visible:
        # Good: agent sees what the question is about, neither pair completed yet.
        return 0.7

    # Subject not clearly visible — fall back to "is anything projected on screen?"
    try:
        _, in_s = scene_ctx.project_aabb_corners(subj, cp, ct)
        _, in_b = scene_ctx.project_aabb_corners(ref_b, cp, ct)
        _, in_c = scene_ctx.project_aabb_corners(ref_c, cp, ct)
        n_partial = int(in_s.sum() > 0) + int(in_b.sum() > 0) + int(in_c.sum() > 0)
    except Exception:  # noqa: BLE001
        n_partial = 0
    if n_partial >= 2:
        return 0.4
    if n_partial == 1:
        return 0.3
    # Facing a wall — possible but uninformative; let other slots decide.
    return 0.1


# ---------------------------------------------------------------------------
# Shared helpers for Batch A/B validators below
# ---------------------------------------------------------------------------

def _has_any_in_frame_corner(scene_ctx, obj_id, cp, ct) -> bool:
    """True iff at least one AABB corner of ``obj_id`` projects inside the image."""
    try:
        _, in_frame = scene_ctx.project_aabb_corners(obj_id, cp, ct)
        return bool(in_frame.sum() > 0)
    except Exception:
        return False


def _is_visible(scene_ctx, obj_id, cp, ct, hfov,
                min_corners: int = 3, max_occ: float = 0.40) -> bool:
    """Wrapper around ``predicates.Visible`` that never raises."""
    try:
        from ...evaluation.predicates import Visible
        return bool(Visible(cp, ct, hfov, scene_ctx,
                            obj=obj_id, min_corners=min_corners, max_occ=max_occ))
    except Exception:
        return False


def _bbox_pixel_area(scene_ctx, obj_id, cp, ct) -> float:
    """Bounding-box pixel area of in-frame AABB corners.  0.0 if not on screen."""
    try:
        uv, in_frame = scene_ctx.project_aabb_corners(obj_id, cp, ct)
        pts = uv[in_frame]
        if pts.shape[0] < 2:
            return 0.0
        du = float(pts[:, 0].max() - pts[:, 0].min())
        dv = float(pts[:, 1].max() - pts[:, 1].min())
        return du * dv
    except Exception:
        return 0.0


# ===========================================================================
# Batch A — validators built on existing predicates / scene_ctx helpers
# ===========================================================================

# -------------------------------------------------------------------
# T11 (Single-vs-multiple Split) — max_init_separation_frac
# -------------------------------------------------------------------
# Intent: at init the two same-label instances must appear MERGED, so the
# agent cannot count them just by looking forward.  ``max_init_separation_frac``
# is the maximum allowed on-image separation as fraction of image width.
# -------------------------------------------------------------------

@register_init_validator("max_init_separation_frac")
def _v_max_init_separation_frac(value, spec, ti, cp, ct, hfov, ctx):
    try:
        max_frac = float(value)
    except Exception:
        return 0.0
    a = ti.get("pair_a_id") or ti.get("member_id")
    b = ti.get("pair_b_id")
    if not a or not b or a == b:
        # Single-instance proxy — soft skip; let other signals decide.
        return 0.0
    try:
        uva, fa = ctx.project_aabb_corners(a, cp, ct)
        uvb, fb = ctx.project_aabb_corners(b, cp, ct)
    except Exception:
        return 0.0
    if fa.sum() == 0 or fb.sum() == 0:
        # At least one off-screen — separation is "infinite", soft positive
        # because the agent still has to move to count.
        return 0.3
    ca = uva[fa].mean(axis=0)
    cb = uvb[fb].mean(axis=0)
    try:
        W = float(ctx._image_width())
    except Exception:
        W = 0.0
    if W <= 0:
        return 0.0
    sep_frac = float(np.linalg.norm(ca - cb)) / W
    if sep_frac > max_frac:
        return float("-inf")
    # Reward: smaller separation = stronger illusion of "single object"
    return 1.0 - 0.5 * (sep_frac / max(max_frac, 1e-6))


# -------------------------------------------------------------------
# T13 (Post-Occlusion Continuation) — min_init_occlusion_fraction
# T17/T18/T19 (Post-Occlusion *)     — min_target_occlusion_at_init
# -------------------------------------------------------------------
# These two field names express the same intent: at init the target/back
# object must be ≥X occluded (so the agent must reposition to disambiguate).
# T13 uses ``back_id``; T17-19 use ``target_id``.
# -------------------------------------------------------------------

def _check_min_occlusion(value, ti, cp, ct, ctx, *, obj_key_priority):
    try:
        min_occ = float(value)
    except Exception:
        return 0.0
    obj = None
    for k in obj_key_priority:
        v = ti.get(k)
        if v:
            obj = v
            break
    if not obj:
        return 0.0
    try:
        occ = float(ctx.occlusion_fraction(obj, cp, ct))
    except Exception:
        return 0.0
    if occ < min_occ:
        return float("-inf")
    # Graded reward in (~0.5, 1.0]: deeper occlusion is more illusory.
    return 0.5 + 0.5 * min(1.0, occ)


@register_init_validator("min_init_occlusion_fraction")
def _v_min_init_occlusion_fraction(value, spec, ti, cp, ct, hfov, ctx):
    return _check_min_occlusion(
        value, ti, cp, ct, ctx,
        obj_key_priority=("back_id", "target_id"),
    )


@register_init_validator("min_target_occlusion_at_init")
def _v_min_target_occlusion_at_init(value, spec, ti, cp, ct, hfov, ctx):
    return _check_min_occlusion(
        value, ti, cp, ct, ctx,
        obj_key_priority=("target_id", "back_id"),
    )


# -------------------------------------------------------------------
# T14 (Back-Face Acquisition) — max_init_back_angle_deg
# -------------------------------------------------------------------
# Init must view the object from the FRONT side: angle between camera-to-
# object direction and the object's front normal is ≤ max.  Requires
# ``front_normal`` annotation; instantiator raises SceneRequirementUnmet
# when missing, so this validator is mostly defensive.
# -------------------------------------------------------------------

@register_init_validator("max_init_back_angle_deg")
def _v_max_init_back_angle_deg(value, spec, ti, cp, ct, hfov, ctx):
    try:
        max_deg = float(value)
    except Exception:
        return 0.0
    tgt = ti.get("target_id") or ti.get("subject_id")
    if not tgt:
        return 0.0
    try:
        n = ctx._object_face_normal(tgt, "front")
        centre = ctx.get_object_centre(tgt)
    except Exception:
        return 0.0
    if n is None or centre is None:
        # Scene has no front-normal annotation — cannot evaluate; neutral.
        return 0.0
    d = np.asarray(cp, dtype=float) - np.asarray(centre, dtype=float)
    dn = float(np.linalg.norm(d))
    nn = float(np.linalg.norm(n))
    if dn < 1e-6 or nn < 1e-6:
        return 0.0
    cos_a = float(np.dot(d, n) / (dn * nn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    if ang > max_deg:
        return float("-inf")
    return 1.0 - 0.5 * (ang / max(max_deg, 1e-6))


# -------------------------------------------------------------------
# T15 (Label-Face Acquisition) — label_side_visible_at_init: false
# -------------------------------------------------------------------
# At init the labelled side must NOT face the camera, so the agent has to
# walk around to read it.  Uses ``_object_face_normal(obj, label_side)``.
# -------------------------------------------------------------------

@register_init_validator("label_side_visible_at_init")
def _v_label_side_visible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    want_visible = bool(value)
    tgt = ti.get("target_id") or ti.get("subject_id")
    side = ti.get("label_side") or ti.get("side_name")
    if not tgt or not side:
        return 0.0
    try:
        n = ctx._object_face_normal(tgt, side)
        centre = ctx.get_object_centre(tgt)
    except Exception:
        return 0.0
    if n is None or centre is None:
        return 0.0
    d = np.asarray(cp, dtype=float) - np.asarray(centre, dtype=float)
    dn = float(np.linalg.norm(d))
    nn = float(np.linalg.norm(n))
    if dn < 1e-6 or nn < 1e-6:
        return 0.0
    cos_a = float(np.dot(d, n) / (dn * nn))
    side_faces_camera = cos_a > 0.5  # camera roughly on positive normal side
    if want_visible and not side_faces_camera:
        return float("-inf")
    if (not want_visible) and side_faces_camera:
        return float("-inf")
    # Reward: clearer rejection of the unwanted side = better init.
    return 0.5 + 0.5 * (abs(cos_a))


# -------------------------------------------------------------------
# T16 (Front-Back Difference) — only_one_face_visible_at_init
# -------------------------------------------------------------------
# At init, only ONE of {front, back} faces of the target should be visible.
# Without per-face geometry we can only check the front-normal sign:
# strictly positive (front faces camera) XOR strictly negative (back faces).
# Mid-range (side view) shows neither cleanly → reject.
# T16 instantiator is stubbed for scenes without annotations.
# -------------------------------------------------------------------

@register_init_validator("only_one_face_visible_at_init")
def _v_only_one_face_visible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    if not bool(value):
        return 0.0
    tgt = ti.get("target_id") or ti.get("subject_id")
    if not tgt:
        return 0.0
    try:
        n = ctx._object_face_normal(tgt, "front")
        centre = ctx.get_object_centre(tgt)
    except Exception:
        return 0.0
    if n is None or centre is None:
        return 0.0
    d = np.asarray(cp, dtype=float) - np.asarray(centre, dtype=float)
    dn = float(np.linalg.norm(d))
    nn = float(np.linalg.norm(n))
    if dn < 1e-6 or nn < 1e-6:
        return 0.0
    cos_a = abs(float(np.dot(d, n) / (dn * nn)))
    # cos≈1: pure front/back view (good); cos≈0: pure side view (bad).
    if cos_a < 0.30:
        return float("-inf")
    return cos_a  # in [0.3, 1.0]


# -------------------------------------------------------------------
# T24 (Portal Direction) — portal_invisible_at_init
# -------------------------------------------------------------------
# At init the portal must not be visible (outside FOV or far behind agent).
# Uses ``_portal_dict(portal_id)`` to fetch position + width; FOV inclusion
# is approximated by the half-angle of the portal subtended at the camera.
# -------------------------------------------------------------------

@register_init_validator("portal_invisible_at_init")
def _v_portal_invisible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    want_invisible = bool(value)
    pid = ti.get("portal_id")
    if not pid:
        return 0.0
    try:
        pd = ctx._portal_dict(pid)
    except Exception:
        return 0.0
    if pd is None:
        return 0.0
    pos = np.asarray(pd.get("position"), dtype=float)
    cp_a = np.asarray(cp, dtype=float)
    ct_a = np.asarray(ct, dtype=float)
    f = ct_a - cp_a; f[2] = 0.0
    d = pos - cp_a;  d[2] = 0.0
    fn = float(np.linalg.norm(f))
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return 0.3
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    half_fov = 0.5 * float(hfov)
    in_fov = ang <= half_fov
    if not in_fov:
        return float("-inf")
    if ang <= 15.0:
        return 0.2
    if ang <= 35.0:
        return 0.6
    return 1.0
    if want_invisible and in_fov:
        return float("-inf")
    if (not want_invisible) and (not in_fov):
        return float("-inf")
    # Reward: portal at side bearing (90°-150°) is the canonical case where
    # the agent must turn to find it — the search task is well-formed.
    if want_invisible:
        if 60.0 <= ang <= 150.0:
            return 1.0
        if ang > 150.0:
            return 0.5  # portal behind: trivial 180° turn
        return 0.3
    return 0.7  # portal in FOV as desired


# -------------------------------------------------------------------
# T26 (Occluded Counting) — min_init_occluded_count
# -------------------------------------------------------------------
# At least N instances of ``target_label`` must NOT be visible at init,
# so the agent cannot just count what's in front of them.
# -------------------------------------------------------------------

@register_init_validator("min_init_occluded_count")
def _v_min_init_occluded_count(value, spec, ti, cp, ct, hfov, ctx):
    try:
        min_count = int(value)
    except Exception:
        return 0.0
    label = ti.get("target_label")
    if not label:
        return 0.0
    try:
        items = ctx.get_objects_by_label(label)
    except Exception:
        items = []
    if not items:
        return 0.0
    invisible = 0
    visible = 0
    for item in items:
        obj_id = getattr(item, "id", None)
        if not obj_id:
            continue
        if _is_visible(ctx, obj_id, cp, ct, hfov,
                       min_corners=1, max_occ=0.95):
            visible += 1
        if not _is_visible(ctx, obj_id, cp, ct, hfov,
                           min_corners=3, max_occ=0.40):
            invisible += 1
    if visible <= 0:
        return float("-inf")
    if invisible < min_count:
        return float("-inf")
    # Reward proportional to fraction occluded
    return min(1.0, invisible / max(1, len(items)))


# -------------------------------------------------------------------
# T27 (Zone Counting) — min_zones_invisible_at_init
# -------------------------------------------------------------------
# At least N zones (rooms) must be invisible from the init pose.
# A zone is "invisible" if NO instance inside it is visible.
# -------------------------------------------------------------------

@register_init_validator("min_zones_invisible_at_init")
def _v_min_zones_invisible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    try:
        min_count = int(value)
    except Exception:
        return 0.0
    zones = []
    za = ti.get("zone_a_targets")
    zb = ti.get("zone_b_targets")
    if isinstance(za, list):
        zones.append(za)
    if isinstance(zb, list):
        zones.append(zb)
    if not zones:
        return 0.0
    invisible_zones = 0
    visible_zones = 0
    for zone_ids in zones:
        if not zone_ids:
            # Empty zone is trivially "invisible".
            invisible_zones += 1
            continue
        any_visible = any(
            _is_visible(ctx, oid, cp, ct, hfov, min_corners=1, max_occ=0.95)
            for oid in zone_ids
        )
        if any_visible:
            visible_zones += 1
        if not any_visible:
            invisible_zones += 1
    if visible_zones <= 0:
        return float("-inf")
    if invisible_zones < min_count:
        return float("-inf")
    return min(1.0, invisible_zones / max(1, len(zones)))


# ===========================================================================
# Batch B — validators that need bespoke geometry computations
# ===========================================================================

# -------------------------------------------------------------------
# T04 (Actual Size Comparison) — init_apparent_size_ratio_{min,max}
# -------------------------------------------------------------------
# At init, the on-image apparent sizes of obj_a and obj_b must produce a
# ratio inside [min, max].  Numbers near 1 mean "looks the same size", so
# the agent cannot answer by eyeballing.  Both bounds are checked in one
# validator (the ``_max`` validator returns 0 to avoid double-counting).
# -------------------------------------------------------------------

def _t04_size_ratio_check(value, spec, ti, cp, ct, hfov, ctx):
    trigger = getattr(spec, "trigger", {}) or {}
    try:
        lo = float(trigger.get("init_apparent_size_ratio_min", 0.0))
        hi = float(trigger.get("init_apparent_size_ratio_max", float("inf")))
    except Exception:
        return 0.0
    a = ti.get("obj_a_id")
    b = ti.get("obj_b_id")
    if not a or not b:
        return 0.0
    area_a = _bbox_pixel_area(ctx, a, cp, ct)
    area_b = _bbox_pixel_area(ctx, b, cp, ct)
    if area_a <= 1.0 or area_b <= 1.0:
        return float("-inf")
    if area_a <= 1.0 or area_b <= 1.0:
        # At least one is essentially off-screen — can't evaluate apparent
        # size; treat as non-trivial (must navigate) with soft positive.
        return 0.3
    ratio = area_a / area_b
    # Compare against the bound that puts the ratio in [lo, hi].
    if ratio < lo or ratio > hi:
        return float("-inf")
    # Sweet spot: ratio ≈ midpoint of [lo, hi] in log space.
    log_lo = math.log(max(lo, 1e-6))
    log_hi = math.log(max(hi, 1e-6))
    log_mid = 0.5 * (log_lo + log_hi)
    span = max(0.5 * (log_hi - log_lo), 1e-6)
    deviation = abs(math.log(ratio) - log_mid) / span  # 0 at midpoint, 1 at bound
    return 1.0 - 0.5 * deviation  # in [0.5, 1.0]


@register_init_validator("init_apparent_size_ratio_min")
def _v_init_apparent_size_ratio_min(value, spec, ti, cp, ct, hfov, ctx):
    return _t04_size_ratio_check(value, spec, ti, cp, ct, hfov, ctx)


@register_init_validator("init_apparent_size_ratio_max")
def _v_init_apparent_size_ratio_max(value, spec, ti, cp, ct, hfov, ctx):
    # Both bounds are evaluated under the ``_min`` validator.  Returning 0
    # here keeps each candidate's total un-inflated.
    return 0.0


# -------------------------------------------------------------------
# T08 (Configuration Judgment) — min_projection_ambiguity_deg
# -------------------------------------------------------------------
# The group's principal XY axis defines a "look-along" direction where
# objects project ~collinearly (the illusion that makes the task hard).
# At init, camera forward must be WITHIN ``min_projection_ambiguity_deg``
# of that axis — viewing perpendicular destroys the illusion.
# -------------------------------------------------------------------

@register_init_validator("min_projection_ambiguity_deg")
def _v_min_projection_ambiguity_deg(value, spec, ti, cp, ct, hfov, ctx):
    try:
        max_off_axis = float(value)
    except Exception:
        return 0.0
    group_ids = ti.get("group_ids")
    if not group_ids or len(group_ids) < 2:
        return 0.0
    centres = []
    for gid in group_ids:
        try:
            c = ctx.get_object_centre(gid)
        except Exception:
            c = None
        if c is None:
            continue
        centres.append(np.asarray(c, dtype=float)[:2])
    if len(centres) < 2:
        return 0.0
    centres = np.stack(centres)
    centroid = centres.mean(axis=0)
    pca = centres - centroid
    try:
        _, _, vt = np.linalg.svd(pca, full_matrices=False)
        axis = vt[0]   # principal direction in XY
    except Exception:
        return 0.0
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-6:
        return 0.0
    axis = axis / axis_n
    f, fn = _forward_xy(cp, ct)
    if fn < 1e-6:
        return 0.0
    cos_a = abs(float(np.dot(f[:2] / fn, axis)))  # axis is unsigned → take abs
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    if ang > max_off_axis:
        return float("-inf")
    # Reward: smaller angle from axis = stronger projection illusion.
    return 1.0 - (ang / max(max_off_axis, 1e-6)) * 0.5  # in [0.5, 1.0]


# -------------------------------------------------------------------
# T21 (Distance Ordering) — init_ordering_ambiguous
# -------------------------------------------------------------------
# At init, the relative depth ordering of obj_a vs obj_b (with respect to
# the reference object) must be ambiguous — i.e. their bearings from the
# camera are nearly identical, so the agent cannot rank distances by
# eyeballing.  We measure the angular separation (in XY) between bearing
# to A and bearing to B.  Small angle → ambiguous → reward.
# -------------------------------------------------------------------

@register_init_validator("init_ordering_ambiguous")
def _v_init_ordering_ambiguous(value, spec, ti, cp, ct, hfov, ctx):
    want_ambiguous = bool(value)
    a = ti.get("obj_a_id")
    b = ti.get("obj_b_id")
    if not a or not b:
        return 0.0
    ca = ctx.get_object_centre(a)
    cb = ctx.get_object_centre(b)
    if ca is None or cb is None:
        return 0.0
    cp_a = np.asarray(cp, dtype=float)
    da = np.asarray(ca, dtype=float) - cp_a; da[2] = 0.0
    db = np.asarray(cb, dtype=float) - cp_a; db[2] = 0.0
    dan = float(np.linalg.norm(da))
    dbn = float(np.linalg.norm(db))
    if dan < 1e-6 or dbn < 1e-6:
        return 0.0
    cos_a = float(np.dot(da, db) / (dan * dbn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    # "Ambiguous" threshold: bearings within 8° of each other (objects appear
    # stacked / overlapping on screen).
    AMBIG_DEG = 8.0
    is_ambig = ang <= AMBIG_DEG
    if want_ambiguous and not is_ambig:
        return float("-inf")
    if (not want_ambiguous) and is_ambig:
        return float("-inf")
    if want_ambiguous:
        # Reward: tighter overlap = stronger ambiguity.
        return 1.0 - 0.5 * (ang / AMBIG_DEG)
    # want_unambiguous: reward wide separation, up to 90°.
    return min(1.0, 0.4 + 0.6 * (ang / 90.0))


# -------------------------------------------------------------------
# T23 (Inter-Room Visibility) — target_invisible_from_room_a
# -------------------------------------------------------------------
# Agent must START in room_a, AND target (in room_b) must not be visible
# from that pose.  Hard-reject if cam outside room_a OR if target visible.
# -------------------------------------------------------------------

@register_init_validator("target_invisible_from_room_a")
def _v_target_invisible_from_room_a(value, spec, ti, cp, ct, hfov, ctx):
    if not bool(value):
        return 0.0
    room_a = ti.get("room_a_id")
    target = ti.get("target_id")
    if not room_a or not target:
        return 0.0
    if not _is_visible(ctx, target, cp, ct, hfov, min_corners=1, max_occ=0.95):
        return float("-inf")
    if _is_visible(ctx, target, cp, ct, hfov, min_corners=4, max_occ=0.40):
        return 0.2
    return 1.0
    try:
        in_room = bool(ctx.is_position_in_room_id(cp, room_a))
    except Exception:
        in_room = False
    if not in_room:
        return float("-inf")
    if _is_visible(ctx, target, cp, ct, hfov, min_corners=3, max_occ=0.40):
        return float("-inf")
    # Reward higher when target is also off-screen entirely (not just occluded
    # behind a wall): agent definitely has to navigate to room_b.
    if not _has_any_in_frame_corner(ctx, target, cp, ct):
        return 1.0
    return 0.7


# -------------------------------------------------------------------
# T29 (Lateral Bearing) — max_init_lateral_angle_deg
# -------------------------------------------------------------------
# Camera forward must be roughly aligned (within ``max_init_lateral_angle_deg``)
# with the A→B axis in XY, so the two objects appear stacked along the depth
# direction and the agent cannot judge their lateral separation from view.
# -------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Runtime overrides for legacy field names whose old implementations still
# contain pre-anchor semantics. These assignments intentionally replace the
# decorator registrations above without changing YAML field names.
# ---------------------------------------------------------------------------

def _v_target_visible_anchor_at_init(value, spec, task_instance, cp, ct, hfov, scene_ctx):
    tgt = (task_instance.get("target_id")
           or task_instance.get("subject_id")
           or task_instance.get("query_id"))
    if not tgt:
        return 0.0
    from ...evaluation.predicates import Visible

    relaxed_visible = Visible(cp, ct, hfov, scene_ctx,
                              obj=tgt, min_corners=1, max_occ=0.95)
    two_corner_visible = Visible(cp, ct, hfov, scene_ctx,
                                 obj=tgt, min_corners=2, max_occ=0.95)
    answer_visible = Visible(cp, ct, hfov, scene_ctx,
                             obj=tgt, min_corners=4, max_occ=0.30)
    if bool(value) and (two_corner_visible or answer_visible):
        return float("-inf")
    if (not bool(value)) and not answer_visible:
        return float("-inf")
    ang = _bearing_to_object_deg(cp, ct, scene_ctx, tgt)
    if ang is None:
        return 0.6 if relaxed_visible else 0.4
    if not relaxed_visible:
        return 0.8 if ang <= 50.0 else 0.3
    if ang <= 20.0:
        return 1.0
    if ang <= 35.0:
        return 0.7
    if ang <= 50.0:
        return 0.4
    return 0.1


def _v_portal_visible_anchor_at_init(value, spec, ti, cp, ct, hfov, ctx):
    pid = ti.get("portal_id")
    if not pid:
        return 0.0
    try:
        pd = ctx._portal_dict(pid)
    except Exception:
        return 0.0
    if pd is None or pd.get("position") is None:
        return 0.0
    pos = np.asarray(pd.get("position"), dtype=float)
    cp_a = np.asarray(cp, dtype=float)
    ct_a = np.asarray(ct, dtype=float)
    f = ct_a - cp_a; f[2] = 0.0
    d = pos - cp_a;  d[2] = 0.0
    fn = float(np.linalg.norm(f))
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return 0.3
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    if ang > 0.5 * float(hfov):
        return float("-inf")
    if ang <= 15.0:
        return 1.0
    if ang <= 35.0:
        return 0.7
    return 0.5


def _v_start_room_with_portal_anchor(value, spec, ti, cp, ct, hfov, ctx):
    if not bool(value):
        return 0.0
    room_a = ti.get("room_a_id")
    if not room_a:
        return 0.0
    try:
        if not bool(ctx.is_position_in_room_id(cp, room_a)):
            return float("-inf")
    except Exception:
        return float("-inf")

    target = ti.get("target_id")
    if target and _is_visible(ctx, target, cp, ct, hfov,
                              min_corners=3, max_occ=0.40):
        return float("-inf")

    portal = ti.get("portal_0_id") or ti.get("portal_id")
    if not portal:
        return 0.4
    try:
        pd = ctx._portal_dict(portal)
    except Exception:
        pd = None
    if pd is None or pd.get("position") is None:
        return 0.4
    pos = np.asarray(pd["position"], dtype=float)
    f, fn = _forward_xy(cp, ct)
    d = pos - np.asarray(cp, dtype=float)
    d[2] = 0.0
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return 0.4
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    if ang > 0.5 * float(hfov):
        return float("-inf")
    if ang <= 15.0:
        return 1.0
    if ang <= 35.0:
        return 0.7
    return 0.5


INIT_VALIDATORS["target_invisible_at_init"] = _v_target_visible_anchor_at_init
INIT_VALIDATORS["portal_invisible_at_init"] = _v_portal_visible_anchor_at_init
INIT_VALIDATORS["target_invisible_from_room_a"] = _v_start_room_with_portal_anchor


@register_init_validator("max_init_lateral_angle_deg")
def _v_max_init_lateral_angle_deg(value, spec, ti, cp, ct, hfov, ctx):
    try:
        max_deg = float(value)
    except Exception:
        return 0.0
    a = ti.get("obj_a_id")
    b = ti.get("obj_b_id")
    if not a or not b:
        return 0.0
    ca = ctx.get_object_centre(a)
    cb = ctx.get_object_centre(b)
    if ca is None or cb is None:
        return 0.0
    axis = np.asarray(cb, dtype=float) - np.asarray(ca, dtype=float)
    axis[2] = 0.0
    an = float(np.linalg.norm(axis))
    f, fn = _forward_xy(cp, ct)
    if an < 1e-6 or fn < 1e-6:
        return 0.0
    cos_a = abs(float(np.dot(f, axis) / (fn * an)))  # axis is unsigned
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    if ang > max_deg:
        return float("-inf")
    # Reward: closer to axis-aligned = stronger stacking illusion.
    return 1.0 - 0.5 * (ang / max(max_deg, 1e-6))


# -------------------------------------------------------------------
# T32 (Multi-Room Path) — path_portals_not_all_visible_at_init
# -------------------------------------------------------------------
# At least one portal on the path (room_a → room_b → ... → end_room) must
# be invisible at init, so the agent has to navigate to discover it.
# task_instance currently exposes only ``portal_0_id`` (first hop); we check
# that one — if it's visible, the trivial "look ahead" answers the task.
# -------------------------------------------------------------------

@register_init_validator("path_portals_not_all_visible_at_init")
def _v_path_portals_not_all_visible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    if not bool(value):
        return 0.0
    portal_id = ti.get("portal_0_id")
    if not portal_id:
        return 0.0
    try:
        pd = ctx._portal_dict(portal_id)
    except Exception:
        pd = None
    if pd is None:
        return 0.0
    pos = np.asarray(pd.get("position"), dtype=float)
    cp_a = np.asarray(cp, dtype=float)
    f, fn = _forward_xy(cp, ct)
    d = pos - cp_a; d[2] = 0.0
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return 0.3
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    half_fov = 0.5 * float(hfov)
    portal_in_fov = ang <= half_fov
    if portal_in_fov:
        # First portal already visible — task may still be valid if there are
        # multiple hops, but quality is lower.  Soft reward, no hard reject.
        return 0.2
    if ang > 150.0:
        # Portal essentially behind agent — trivial 180° turn.
        return 0.4
    return 1.0


# -------------------------------------------------------------------
# T33 (Passage Width) — passage_not_orthogonally_visible_at_init
# -------------------------------------------------------------------
# At init the camera must NOT be looking head-on through the passage — if
# it were, the agent could measure clearance from view alone.  We approximate
# "orthogonal view" as: passage centre is in FOV AND camera-forward is
# nearly aligned with camera→passage direction (i.e. agent looking straight
# at the passage).  Reject those poses; reward oblique / off-axis views.
# -------------------------------------------------------------------

@register_init_validator("passage_not_orthogonally_visible_at_init")
def _v_passage_not_orthogonally_visible_at_init(value, spec, ti, cp, ct, hfov, ctx):
    if not bool(value):
        return 0.0
    portal_id = ti.get("passage_id") or ti.get("portal_id")
    if not portal_id:
        return 0.0
    try:
        pd = ctx._portal_dict(portal_id)
    except Exception:
        pd = None
    if pd is None:
        return 0.0
    pos = np.asarray(pd.get("position"), dtype=float)
    cp_a = np.asarray(cp, dtype=float)
    f, fn = _forward_xy(cp, ct)
    d = pos - cp_a; d[2] = 0.0
    dn = float(np.linalg.norm(d))
    if fn < 1e-6 or dn < 1e-6:
        return 0.3
    cos_a = float(np.dot(f, d) / (fn * dn))
    cos_a = max(-1.0, min(1.0, cos_a))
    ang = math.degrees(math.acos(cos_a))
    # "Looking straight through": passage centre within ~15° of forward.
    if ang <= 15.0:
        return 0.3
    if ang <= 30.0:
        return 0.4  # near-frontal — still some clearance cue from view
    # Oblique or side view: agent must approach to measure width.
    return 1.0
