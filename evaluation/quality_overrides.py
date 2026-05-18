"""
evaluation/quality_overrides.py

Tier-2 callables registered via QUALITY_REGISTRY. Each takes:

    fn(slot:           EvidenceSlot,        # already resolved (no {{var}})
       trajectory:     Sequence[(np.ndarray, np.ndarray)],   # full trajectory
       task_instance:  dict,                # the per-task variables
       scene_ctx:      SceneContext) -> bool

Tier-2 callables encode template-specific *holistic* checks that the simple
"fraction-of-predicates-passed" rule cannot capture (e.g., balanced distance
to two reference objects, or counting all instances inside a zone).

Submit-time semantics
---------------------
The coverage layer ALWAYS hands the FULL trajectory to a Tier-2 callable.
Each callable is responsible for choosing what to look at:

    * "submit-time" check  → read `trajectory[-1]` only.
      Example::
          cp, ct = trajectory[-1]
          return slot_satisfied_at(slot, cp, ct, hfov, scene_ctx)

    * "trajectory-level" check → scan all frames.
      Example: T23 "did the agent cover ≥270° of yaw inside room_b?"

Worked example — toy override::

    @register_quality("two_objects_visible_together")
    def two_objects_visible_together(slot, trajectory, task_instance, ctx):
        a = task_instance["obj_a_id"]; b = task_instance["obj_b_id"]
        if not trajectory:
            return False
        cp, ct = trajectory[-1]
        hfov = ctx.default_hfov_deg()
        return (Visible(cp, ct, hfov, ctx, obj=a, min_corners=3, max_occ=0.4)
                and Visible(cp, ct, hfov, ctx, obj=b, min_corners=3, max_occ=0.4))

In a YAML evidence slot, opt in via::

    evidence_slots:
      - slot_id: pair_view
        region_generator: around_pair
        region_args: {a: "{{obj_a_id}}", b: "{{obj_b_id}}"}
        predicates: []                                 # ignored when override set
        threshold: 1.0
        tier2_override: two_objects_visible_together
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from .coverage import register_quality, slot_satisfied_at
from .predicates import Visible
from ..core import scene_context_ext  # noqa: F401
from ..core.scene_context import SceneContext


# ---------------------------------------------------------------------------
# T04 — distance_balance_check
# ---------------------------------------------------------------------------

@register_quality("distance_balance_check")
def distance_balance_check(slot, trajectory, task_instance, scene_ctx):
    """
    For T04 (actual_size_comparison):
    requires |log(d_a/d_b)| ≤ 0.5 at the submit (last) view, with both
    objects visible.
    """
    if not trajectory:
        return False
    obj_a = task_instance.get("obj_a_id") or slot.region_args.get("obj_a")
    obj_b = task_instance.get("obj_b_id") or slot.region_args.get("obj_b")
    if obj_a is None or obj_b is None:
        return False
    ca = scene_ctx.get_object_centre(obj_a)
    cb = scene_ctx.get_object_centre(obj_b)
    if ca is None or cb is None:
        return False
    cp, ct = trajectory[-1]
    cp = np.asarray(cp)
    d_a = float(np.linalg.norm(cp - ca))
    d_b = float(np.linalg.norm(cp - cb))
    if d_a < 0.2 or d_b < 0.2:
        return False
    if abs(math.log(d_a / d_b)) > 0.5:
        return False
    hfov = scene_ctx.default_hfov_deg()
    return (
        Visible(cp, np.asarray(ct), hfov, scene_ctx,
                obj=obj_a, min_corners=3, max_occ=0.40)
        and
        Visible(cp, np.asarray(ct), hfov, scene_ctx,
                obj=obj_b, min_corners=3, max_occ=0.40)
    )


# ---------------------------------------------------------------------------
# T08 — axis_view_angle_check
# ---------------------------------------------------------------------------

@register_quality("axis_view_angle_check")
def axis_view_angle_check(slot, trajectory, task_instance, scene_ctx):
    """
    For T08 (configuration_judgment):
    Requires angle between (camera→group_centroid) and the group's principal
    XY axis to lie in [30°, 75°] at the submit view, AND all group objects
    visible.
    """
    if not trajectory:
        return False
    group = task_instance.get("group_ids") or slot.region_args.get("group")
    if group is None:
        return False
    if isinstance(group, str):
        group = [group]
    centres = [scene_ctx.get_object_centre(i) for i in group]
    centres = [c for c in centres if c is not None]
    if len(centres) < 2:
        return False
    pts = np.stack([c[:2] for c in centres])
    centroid = pts.mean(axis=0)
    pca = pts - centroid
    _, _, vh = np.linalg.svd(pca)
    axis = vh[0]

    cp, ct = trajectory[-1]
    cp = np.asarray(cp)
    d = centroid - cp[:2]
    if np.linalg.norm(d) < 1e-6:
        return False
    d = d / np.linalg.norm(d)
    cos_a = float(np.clip(abs(np.dot(d, axis)), 0.0, 1.0))
    angle = math.degrees(math.acos(cos_a))
    if not (30.0 <= angle <= 75.0):
        return False
    # All group objects visible
    hfov = scene_ctx.default_hfov_deg()
    for oid in group:
        if not Visible(cp, np.asarray(ct), hfov, scene_ctx,
                        obj=oid, min_corners=3, max_occ=0.40):
            return False
    return True


# ---------------------------------------------------------------------------
# T17 — hidden_region_visibility_check
# ---------------------------------------------------------------------------

@register_quality("hidden_region_visibility_check")
def hidden_region_visibility_check(slot, trajectory, task_instance, scene_ctx):
    """
    For T17 (post_occlusion_existence):
    Branches on GT.
      • GT='yes' (target exists): Visible(target_id) on the submit view.
      • GT='no'  (target absent): occluder back-region is now in clear sight,
        i.e. the standard slot predicates pass at submit view.
    """
    if not trajectory:
        return False
    cp, ct = trajectory[-1]
    cp = np.asarray(cp)
    ct = np.asarray(ct)
    hfov = scene_ctx.default_hfov_deg()
    gt = (task_instance.get("gt_answer") or task_instance.get("answer") or "").lower()
    if gt == "yes":
        target = task_instance.get("target_id")
        if not target:
            return False
        return Visible(cp, ct, hfov, scene_ctx,
                       obj=target, min_corners=3, max_occ=0.40)
    # GT == "no": confirm we're in a clear-sight position past the occluder
    return slot_satisfied_at(slot, cp, ct, hfov, scene_ctx)


# ---------------------------------------------------------------------------
# T23 — cross_room_existence_check
# ---------------------------------------------------------------------------

@register_quality("cross_room_existence_check")
def cross_room_existence_check(slot, trajectory, task_instance, scene_ctx):
    """
    For T23 (cross_room_existence):
      • Camera must be inside `room_b_id` at submit view (InRoom).
      • GT='yes': Visible(target_id) on some frame.
      • GT='no':  agent has scanned the room — at submit view at least one
        view inside room_b spans ≥ 270° cumulative yaw (approximated by
        sampling unique yaw bins across the trajectory inside room_b).
    """
    if not trajectory:
        return False
    room_b = task_instance.get("room_b_id") or slot.region_args.get("room")
    if room_b is None:
        return False
    cp_last = np.asarray(trajectory[-1][0])
    if not scene_ctx.is_position_in_room_id(cp_last, room_b):
        return False

    gt = (task_instance.get("gt_answer") or task_instance.get("answer") or "").lower()
    hfov = scene_ctx.default_hfov_deg()
    target = task_instance.get("target_id")
    if gt == "yes":
        if not target:
            return False
        for cp, ct in trajectory:
            cp_a = np.asarray(cp)
            if not scene_ctx.is_position_in_room_id(cp_a, room_b):
                continue
            if Visible(cp_a, np.asarray(ct), hfov, scene_ctx,
                       obj=target, min_corners=3, max_occ=0.40):
                return True
        return False
    # GT == "no": measure yaw coverage in room_b
    yaw_bins = set()
    for cp, ct in trajectory:
        cp_a = np.asarray(cp)
        if not scene_ctx.is_position_in_room_id(cp_a, room_b):
            continue
        f = np.asarray(ct) - cp_a
        if np.linalg.norm(f[:2]) < 1e-6:
            continue
        yaw = math.degrees(math.atan2(f[1], f[0])) % 360.0
        yaw_bins.add(int(yaw // 30))   # 12 bins of 30°
    return len(yaw_bins) >= 9   # ≥ 270° coverage


# ---------------------------------------------------------------------------
# T27 — zone_count_check
# ---------------------------------------------------------------------------

@register_quality("zone_count_check")
def zone_count_check(slot, trajectory, task_instance, scene_ctx):
    """
    For T27 (zone_counting):
      All target instances inside the room/zone declared in slot.region_args
      must be visible at SOME frame of the trajectory (not necessarily simultaneously).
      Only targets in the SAME zone as this slot are required (not targets from
      other zones, which would be impossible to see from inside this zone).
    """
    if not trajectory:
        return False
    zone = slot.region_args.get("room")
    # Use zone-specific target list if available (keyed by zone room id)
    if zone is not None:
        zone_a_id = task_instance.get("zone_a_id")
        zone_b_id = task_instance.get("zone_b_id")
        if zone == zone_a_id and "zone_a_targets" in task_instance:
            targets = task_instance["zone_a_targets"]
        elif zone == zone_b_id and "zone_b_targets" in task_instance:
            targets = task_instance["zone_b_targets"]
        else:
            targets = task_instance.get("zone_targets") or task_instance.get("target_ids")
    else:
        targets = task_instance.get("zone_targets") or task_instance.get("target_ids")
    if not targets:
        return False
    hfov = scene_ctx.default_hfov_deg()
    seen = set()
    for cp, ct in trajectory:
        cp_a = np.asarray(cp)
        ct_a = np.asarray(ct)
        if zone is not None and not scene_ctx.is_position_in_room_id(cp_a, zone):
            continue
        for tid in targets:
            if tid in seen:
                continue
            if Visible(cp_a, ct_a, hfov, scene_ctx,
                       obj=tid, min_corners=3, max_occ=0.40):
                seen.add(tid)
    return seen == set(targets)
