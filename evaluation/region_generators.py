"""
evaluation/region_generators.py

Region samplers for evidence-slot positions. Each generator returns:

    List[Tuple[np.ndarray, np.ndarray]]   # (cam_pos, cam_target) pairs

Generators are NAMED — register them in REGION_REGISTRY so YAML templates
can reference them by string.

All generators accept (scene_ctx, rng, **region_args). They may filter for
basic feasibility (in-room, valid position) but do not verify slot predicates.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import numpy as np

from ..core import scene_context_ext  # noqa: F401
from ..core.scene_context import SceneContext, DEFAULT_CAMERA_HEIGHT


Vec3 = np.ndarray
Sample = Tuple[Vec3, Vec3]


# ---------------------------------------------------------------------------
# Bounded sampling helper
# ---------------------------------------------------------------------------

def bounded_sampling(target_n: int, max_attempts_factor: int, sample_fn,
                     filter_fn=None) -> list:
    """Run ``sample_fn`` until ``target_n`` accepted samples are collected
    or ``target_n * max_attempts_factor`` attempts elapse.

    All region generators in this module share this loop shape; centralising
    it avoids the "broken break on ``len(out) > n*6``" class of bug (the
    cap is unreachable because ``out`` only grows on success) by making the
    *attempt* count the authoritative termination condition.

    Args:
        target_n: desired number of samples.
        max_attempts_factor: cap on attempts = target_n * factor.
        sample_fn: callable() -> sample or None.  Called once per attempt.
        filter_fn: optional callable(sample) -> bool.  Applied to non-None.

    Returns:
        List of accepted samples (length ≤ target_n).
    """
    out: list = []
    attempts = 0
    cap = max(1, target_n * max_attempts_factor)
    while len(out) < target_n and attempts < cap:
        attempts += 1
        s = sample_fn()
        if s is None:
            continue
        if filter_fn is not None and not filter_fn(s):
            continue
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ring_sample(centre_xy: np.ndarray, dist: float, yaw_deg: float,
                 height: float) -> np.ndarray:
    rad = math.radians(yaw_deg)
    return np.array([
        float(centre_xy[0] + dist * math.cos(rad)),
        float(centre_xy[1] + dist * math.sin(rad)),
        float(height),
    ])


def _pos_facing(pos: Vec3, target_xy: np.ndarray, height: float) -> Sample:
    target = np.array([float(target_xy[0]), float(target_xy[1]), height])
    return pos, target


def _sample_height(rng: random.Random, hmin: float, hmax: float) -> float:
    return rng.uniform(hmin, hmax)


def _check_valid(scene_ctx: SceneContext, pos: Vec3) -> bool:
    try:
        return scene_ctx.is_position_valid(pos)
    except Exception:
        return True


def _room_centroid(scene_ctx: SceneContext, room_id: Optional[str]) -> Optional[np.ndarray]:
    scene_ctx._ensure_room_index()
    poly = scene_ctx._room_polygons_by_id.get(room_id) if room_id else None
    if poly is None and scene_ctx.room_polygons:
        poly = scene_ctx.room_polygons[0]
    if poly is None or len(poly) == 0:
        return None
    return np.array([float(poly[:, 0].mean()), float(poly[:, 1].mean())])


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def around_object(scene_ctx: SceneContext, rng: random.Random, *,
                  target: str, dist_min: float, dist_max: float,
                  height_min: float = DEFAULT_CAMERA_HEIGHT,
                  height_max: float = DEFAULT_CAMERA_HEIGHT,
                  n: int = 32) -> List[Sample]:
    centre = scene_ctx.get_object_centre(target)
    if centre is None:
        return []
    centre_xy = centre[:2]
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        yaw = rng.uniform(0.0, 360.0)
        h = _sample_height(rng, height_min, height_max)
        pos = _ring_sample(centre_xy, d, yaw, h)
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, centre_xy, h))
    return out


def equidistant_to_pair(scene_ctx: SceneContext, rng: random.Random, *,
                        obj_a: str, obj_b: str, dist_min: float, dist_max: float,
                        max_log_dist_ratio: float = 0.5,
                        n: int = 32) -> List[Sample]:
    ca = scene_ctx.get_object_centre(obj_a)
    cb = scene_ctx.get_object_centre(obj_b)
    if ca is None or cb is None:
        return []
    midpoint = 0.5 * (ca[:2] + cb[:2])
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        yaw = rng.uniform(0.0, 360.0)
        h = DEFAULT_CAMERA_HEIGHT
        pos = _ring_sample(midpoint, d, yaw, h)
        if not _check_valid(scene_ctx, pos):
            continue
        d_a = float(np.linalg.norm(pos[:2] - ca[:2]))
        d_b = float(np.linalg.norm(pos[:2] - cb[:2]))
        if d_a < 0.2 or d_b < 0.2:
            continue
        if abs(math.log(d_a / d_b)) > max_log_dist_ratio:
            continue
        target_xy = 0.5 * (ca[:2] + cb[:2])
        out.append(_pos_facing(pos, target_xy, h))
    return out


def around_pair(scene_ctx: SceneContext, rng: random.Random, *,
                obj_a: str, obj_b: str, dist_min: float, dist_max: float,
                n: int = 32) -> List[Sample]:
    ca = scene_ctx.get_object_centre(obj_a)
    cb = scene_ctx.get_object_centre(obj_b)
    if ca is None or cb is None:
        return []
    midpoint = 0.5 * (ca[:2] + cb[:2])
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        yaw = rng.uniform(0.0, 360.0)
        h = DEFAULT_CAMERA_HEIGHT
        pos = _ring_sample(midpoint, d, yaw, h)
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, midpoint, h))
    return out


def orthogonal_to_portal(scene_ctx: SceneContext, rng: random.Random, *,
                         portal: str, dist_min: float = 0.3, dist_max: float = 2.0,
                         ortho_max_deg: float = 15.0,
                         n: int = 32) -> List[Sample]:
    return scene_ctx.sample_orthogonal_to_portal(
        portal, dist_min=dist_min, dist_max=dist_max,
        ortho_max_deg=ortho_max_deg, n=n, rng=rng,
    )


def view_breaking_projection(scene_ctx: SceneContext, rng: random.Random, *,
                             group, axis_angle_min_deg: float = 30.0,
                             axis_angle_max_deg: float = 75.0,
                             n: int = 32) -> List[Sample]:
    """
    For T08 — view group from an oblique angle relative to the principal axis
    fitted through group object centres.
    """
    ids = list(group) if isinstance(group, (list, tuple)) else [group]
    centres = [scene_ctx.get_object_centre(i) for i in ids]
    centres = [c for c in centres if c is not None]
    if len(centres) < 2:
        return []
    pts = np.stack([c[:2] for c in centres])
    centroid = pts.mean(axis=0)
    # Principal axis via SVD
    pca = pts - centroid
    _, _, vh = np.linalg.svd(pca)
    axis = vh[0]                      # 1st principal direction in XY
    perp = np.array([-axis[1], axis[0]])
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        # Sample target angle relative to axis in [min, max]
        ang = math.radians(rng.uniform(axis_angle_min_deg, axis_angle_max_deg))
        sign = rng.choice([1.0, -1.0])
        d = rng.uniform(1.0, 4.0)
        direction = math.cos(ang) * axis + sign * math.sin(ang) * perp
        pos_xy = centroid + d * direction
        pos = np.array([pos_xy[0], pos_xy[1], DEFAULT_CAMERA_HEIGHT])
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, centroid, DEFAULT_CAMERA_HEIGHT))
    return out


def view_with_separation(scene_ctx: SceneContext, rng: random.Random, *,
                         group, target_separation_frac: float = 0.05,
                         n: int = 32) -> List[Sample]:
    """
    For T11 — sample views around the group centroid; downstream
    SeparationOnImage predicate filters.
    """
    ids = list(group) if isinstance(group, (list, tuple)) else [group]
    centres = [scene_ctx.get_object_centre(i) for i in ids]
    centres = [c for c in centres if c is not None]
    if not centres:
        return []
    centroid = np.mean([c[:2] for c in centres], axis=0)
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(1.5, 4.0)
        yaw = rng.uniform(0.0, 360.0)
        pos = _ring_sample(centroid, d, yaw, DEFAULT_CAMERA_HEIGHT)
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, centroid, DEFAULT_CAMERA_HEIGHT))
    return out


def behind_front_object(scene_ctx: SceneContext, rng: random.Random, *,
                        front_obj: str, back_obj: str,
                        side_angle_min_deg: float = 30.0,
                        n: int = 32) -> List[Sample]:
    """
    For T13 — sample sideways views that put both front_obj and back_obj
    in frame with at least side_angle_min_deg off the front-back axis.
    """
    cf = scene_ctx.get_object_centre(front_obj)
    cb = scene_ctx.get_object_centre(back_obj)
    if cf is None or cb is None:
        return []
    axis = cb[:2] - cf[:2]
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-6:
        return []
    axis = axis / axis_n
    perp = np.array([-axis[1], axis[0]])
    midpoint = 0.5 * (cf[:2] + cb[:2])
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        ang = math.radians(rng.uniform(side_angle_min_deg, 90.0))
        sign = rng.choice([1.0, -1.0])
        d = rng.uniform(1.0, 3.0)
        direction = math.cos(ang) * axis + sign * math.sin(ang) * perp
        pos_xy = midpoint + d * direction
        pos = np.array([pos_xy[0], pos_xy[1], DEFAULT_CAMERA_HEIGHT])
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, midpoint, DEFAULT_CAMERA_HEIGHT))
    return out


def around_occluder_back(scene_ctx: SceneContext, rng: random.Random, *,
                         occluder: str, target: Optional[str] = None,
                         target_region=None,
                         dist_min: float = 0.5, dist_max: float = 3.0,
                         n: int = 32) -> List[Sample]:
    """
    Sample positions around the BACK of the occluder (opposite the front face)
    so the occluder no longer blocks the target zone. Looks toward the region
    behind the occluder.
    """
    centre = scene_ctx.get_object_centre(occluder)
    if centre is None:
        return []
    # If target known, the "back" direction is from occluder centre toward target
    look_target_xy = None
    if target is not None:
        ct = scene_ctx.get_object_centre(target)
        if ct is not None:
            look_target_xy = ct[:2]
    if look_target_xy is None and target_region is not None:
        look_target_xy = np.asarray(target_region)[:2]
    if look_target_xy is None:
        # Fall back to ring sampling
        return around_object(scene_ctx, rng, target=occluder,
                             dist_min=dist_min, dist_max=dist_max, n=n)

    back_dir = look_target_xy - centre[:2]
    bn = float(np.linalg.norm(back_dir))
    if bn < 1e-6:
        return []
    back_dir = back_dir / bn

    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        # Sample within ±60° of back direction, beyond the occluder
        ang = math.radians(rng.uniform(-60.0, 60.0))
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        side_dir = np.array([
            back_dir[0] * cos_a - back_dir[1] * sin_a,
            back_dir[0] * sin_a + back_dir[1] * cos_a,
        ])
        d = rng.uniform(dist_min, dist_max)
        pos_xy = centre[:2] + side_dir * d
        pos = np.array([pos_xy[0], pos_xy[1], DEFAULT_CAMERA_HEIGHT])
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, look_target_xy, DEFAULT_CAMERA_HEIGHT))
    return out


def bearing_sector_in_room(scene_ctx: SceneContext, rng: random.Random, *,
                           target: str, room: Optional[str] = None,
                           bearing_sector_deg: float = 45.0,
                           n: int = 32) -> List[Sample]:
    """
    Position cameras inside the room; orient so target lies within
    ±bearing_sector_deg of forward.
    """
    centre = scene_ctx.get_object_centre(target)
    if centre is None:
        return []
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    positions = scene_ctx.sample_positions_in_room(num_points=n * 3, rng=rs)
    out: List[Sample] = []
    for pos in positions:
        if room is not None and not scene_ctx.is_position_in_room_id(pos, room):
            continue
        d = centre - pos
        if np.linalg.norm(d) < 1e-6:
            continue
        d_xy = d[:2] / (np.linalg.norm(d[:2]) + 1e-9)
        # Build a forward roughly aligned with target direction (within sector)
        offset = math.radians(rng.uniform(-bearing_sector_deg + 5.0,
                                           bearing_sector_deg - 5.0))
        cos_a, sin_a = math.cos(offset), math.sin(offset)
        f_xy = np.array([
            d_xy[0] * cos_a - d_xy[1] * sin_a,
            d_xy[0] * sin_a + d_xy[1] * cos_a,
        ])
        target_pt = np.array([
            pos[0] + f_xy[0],
            pos[1] + f_xy[1],
            DEFAULT_CAMERA_HEIGHT,
        ])
        out.append((pos, target_pt))
        if len(out) >= n:
            break
    return out


def view_of_triple(scene_ctx: SceneContext, rng: random.Random, *,
                   obj_a: str, obj_b: str, reference: str,
                   dist_min: float = 0.8, dist_max: float = 5.0,
                   n: int = 32) -> List[Sample]:
    centres = [scene_ctx.get_object_centre(i) for i in (obj_a, obj_b, reference)]
    centres = [c for c in centres if c is not None]
    if len(centres) < 3:
        return []
    centroid = np.mean([c[:2] for c in centres], axis=0)
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        yaw = rng.uniform(0.0, 360.0)
        pos = _ring_sample(centroid, d, yaw, DEFAULT_CAMERA_HEIGHT)
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, centroid, DEFAULT_CAMERA_HEIGHT))
    return out


def _sample_positions_in_poly(
    scene_ctx: SceneContext, poly: np.ndarray, n: int, rs: np.random.RandomState
) -> List[np.ndarray]:
    """Sample valid camera positions specifically inside *poly* (XY polygon)."""
    from ..bench_generation.batch_utils import point_in_poly
    xs = poly[:, 0]; ys = poly[:, 1]
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    positions: List[np.ndarray] = []
    attempts = 0
    while len(positions) < n and attempts < n * 80:
        attempts += 1
        x = rs.uniform(xmin, xmax)
        y = rs.uniform(ymin, ymax)
        if not point_in_poly(x, y, poly):
            continue
        pos = np.array([x, y, DEFAULT_CAMERA_HEIGHT], dtype=float)
        if not _check_valid(scene_ctx, pos):
            continue
        positions.append(pos)
    return positions


def inside_room(scene_ctx: SceneContext, rng: random.Random, *,
                room: Optional[str] = None,
                dist_from_centroid_max: Optional[float] = None,
                n: int = 32) -> List[Sample]:
    """
    Generic 'agent must be inside this room' sampler. Looks toward the
    centroid by default.

    When *room* is specified the sampler draws directly from the room's own
    polygon rather than filtering a global sample (which is dominated by the
    first / largest room in multi-room scenes).
    """
    centroid = _room_centroid(scene_ctx, room)
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))

    # Try room-specific polygon sampling first (avoids multi-room bias).
    positions: List[np.ndarray] = []
    if room is not None:
        scene_ctx._ensure_room_index()
        poly = getattr(scene_ctx, "_room_polygons_by_id", {}).get(room)
        if poly is not None and len(poly) >= 3:
            positions = _sample_positions_in_poly(scene_ctx, poly, n * 2, rs)

    # Fallback: global sample + filter (for room=None or unknown polygon id).
    if not positions:
        positions = scene_ctx.sample_positions_in_room(num_points=n * 3, rng=rs)
        if room is not None:
            positions = [p for p in positions
                         if scene_ctx.is_position_in_room_id(p, room)]

    out: List[Sample] = []
    for pos in positions:
        if dist_from_centroid_max is not None and centroid is not None:
            if float(np.linalg.norm(pos[:2] - centroid)) > dist_from_centroid_max:
                continue
        # Look toward centroid (or random yaw if centroid is the position)
        if centroid is not None and float(np.linalg.norm(pos[:2] - centroid)) > 0.1:
            target_pt = np.array([centroid[0], centroid[1], DEFAULT_CAMERA_HEIGHT])
        else:
            yaw = rng.uniform(0.0, 2.0 * math.pi)
            target_pt = pos + np.array([math.cos(yaw), math.sin(yaw), 0.0])
        out.append((pos, target_pt))
        if len(out) >= n:
            break
    return out


def near_portal_in_room_a(scene_ctx: SceneContext, rng: random.Random, *,
                          portal: str, room_a: Optional[str] = None,
                          dist_min: float = 0.5, dist_max: float = 3.0,
                          n: int = 32) -> List[Sample]:
    portal_pos = scene_ctx.get_portal_position(portal)
    if portal_pos is None:
        return []
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        yaw = rng.uniform(0.0, 360.0)
        pos = _ring_sample(portal_pos[:2], d, yaw, DEFAULT_CAMERA_HEIGHT)
        if not _check_valid(scene_ctx, pos):
            continue
        if room_a is not None and not scene_ctx.is_position_in_room_id(pos, room_a):
            continue
        target_pt = np.array([portal_pos[0], portal_pos[1], DEFAULT_CAMERA_HEIGHT])
        out.append((pos, target_pt))
    return out


def near_portal_facing(scene_ctx: SceneContext, rng: random.Random, *,
                       portal: str, dist_min: float = 0.5, dist_max: float = 2.5,
                       n: int = 32) -> List[Sample]:
    return near_portal_in_room_a(scene_ctx, rng, portal=portal, room_a=None,
                                 dist_min=dist_min, dist_max=dist_max, n=n)


def around_occluder_to_expose(scene_ctx: SceneContext, rng: random.Random, *,
                              occluder: str,
                              target: Optional[str] = None,
                              target_label: Optional[str] = None,
                              dist_min: float = 0.8, dist_max: float = 3.0,
                              n: int = 32) -> List[Sample]:
    return around_occluder_back(scene_ctx, rng, occluder=occluder,
                                target=target,
                                dist_min=dist_min, dist_max=dist_max, n=n)


def orthogonal_to_segment(scene_ctx: SceneContext, rng: random.Random, *,
                          obj_a: str, obj_b: str,
                          min_lateral_angle_deg: float = 60.0,
                          dist_min: float = 0.5, dist_max: float = 3.0,
                          n: int = 32) -> List[Sample]:
    ca = scene_ctx.get_object_centre(obj_a)
    cb = scene_ctx.get_object_centre(obj_b)
    if ca is None or cb is None:
        return []
    seg = cb[:2] - ca[:2]
    sn = float(np.linalg.norm(seg))
    if sn < 1e-6:
        return []
    seg = seg / sn
    perp = np.array([-seg[1], seg[0]])
    midpoint = 0.5 * (ca[:2] + cb[:2])
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        ang = math.radians(rng.uniform(min_lateral_angle_deg, 90.0))
        sign = rng.choice([1.0, -1.0])
        d = rng.uniform(dist_min, dist_max)
        direction = math.cos(ang) * perp * sign + math.sin(ang) * seg
        pos_xy = midpoint + d * direction
        pos = np.array([pos_xy[0], pos_xy[1], DEFAULT_CAMERA_HEIGHT])
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, midpoint, DEFAULT_CAMERA_HEIGHT))
    return out


def behind_object_face(scene_ctx: SceneContext, rng: random.Random, *,
                       target: str, face: str = "back",
                       side_theta_deg: float = 45.0,
                       dist_min: float = 0.5, dist_max: float = 2.5,
                       n: int = 32) -> List[Sample]:
    centre = scene_ctx.get_object_centre(target)
    if centre is None:
        return []
    n_face = scene_ctx._object_face_normal(target, face)
    if n_face is None:
        return []
    # Compose 2D direction for sampling
    face_xy = n_face[:2]
    nf = float(np.linalg.norm(face_xy))
    if nf < 1e-6:
        face_xy = np.array([1.0, 0.0])
    else:
        face_xy = face_xy / nf
    out: List[Sample] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        ang = math.radians(rng.uniform(-side_theta_deg, side_theta_deg))
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        d = rng.uniform(dist_min, dist_max)
        direction = np.array([
            face_xy[0] * cos_a - face_xy[1] * sin_a,
            face_xy[0] * sin_a + face_xy[1] * cos_a,
        ])
        pos_xy = centre[:2] + direction * d
        pos = np.array([pos_xy[0], pos_xy[1], DEFAULT_CAMERA_HEIGHT])
        if not _check_valid(scene_ctx, pos):
            continue
        out.append(_pos_facing(pos, centre[:2], DEFAULT_CAMERA_HEIGHT))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGION_REGISTRY = {
    "around_object":              around_object,
    "around_pair":                around_pair,
    "equidistant_to_pair":        equidistant_to_pair,
    "orthogonal_to_portal":       orthogonal_to_portal,
    "view_breaking_projection":   view_breaking_projection,
    "view_with_separation":       view_with_separation,
    "behind_front_object":        behind_front_object,
    "around_occluder_back":       around_occluder_back,
    "around_occluder_to_expose":  around_occluder_to_expose,
    "bearing_sector_in_room":     bearing_sector_in_room,
    "view_of_triple":             view_of_triple,
    "inside_room":                inside_room,
    "near_portal_in_room_a":      near_portal_in_room_a,
    "near_portal_facing":         near_portal_facing,
    "orthogonal_to_segment":      orthogonal_to_segment,
    "behind_object_face":         behind_object_face,
}


def sample_region(name: str, scene_ctx: SceneContext, rng: random.Random,
                  **kwargs) -> List[Sample]:
    fn = REGION_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown region generator: {name}")
    return fn(scene_ctx, rng, **kwargs)
