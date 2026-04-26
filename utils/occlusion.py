#!/usr/bin/env python3
"""
Occlusion and visibility utilities based on labels.json AABBs.

Features:
- Load scene AABBs from labels.json (bounding_box with 8 vertices)
- Ray vs AABB intersection (slab method)
- Camera frustum checks using intrinsics K and camtoworld
- Ensure visibility by small lateral/vertical camera adjustments while keeping target fixed

Assumptions:
- Bounding boxes are axis-aligned in world coordinates (min/max over x,y,z of 8 vertices)
- Right-handed world, Z is up
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import math
import numpy as np


@dataclass
class AABB:
    id: str
    label: str
    bmin: np.ndarray  # shape (3,)
    bmax: np.ndarray  # shape (3,)


def _safe_array(a: Any) -> np.ndarray:
    return np.array(a, dtype=float)


def load_scene_aabbs(scene_path: str, exclude_labels: Optional[List[str]] = None) -> List[AABB]:
    sp = Path(scene_path)
    labels_file = sp / 'labels.json'
    aabbs: List[AABB] = []
    if not labels_file.exists():
        return aabbs
    if exclude_labels is None:
        exclude_labels = ['wall', 'floor', 'ceiling', 'room']
    with open(labels_file, 'r') as f:
        data = json.load(f)
    for obj in data:
        if not isinstance(obj, dict):
            continue
        label = str(obj.get('label', '')).lower()
        if label in exclude_labels:
            continue
        bbox = obj.get('bounding_box')
        ins_id = str(obj.get('ins_id', ''))
        if not bbox or len(bbox) < 8:
            continue
        xs = [float(p.get('x', 0.0)) for p in bbox]
        ys = [float(p.get('y', 0.0)) for p in bbox]
        zs = [float(p.get('z', 0.0)) for p in bbox]
        bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
        bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
        aabbs.append(AABB(id=ins_id, label=label, bmin=bmin, bmax=bmax))
    return aabbs


def load_scene_wall_aabbs(scene_path: str) -> List[AABB]:
    """Load wall segments from structure.json and convert to AABBs.

    Each wall has: {thickness: float (m), height: float (m), location: [[x1,y1], [x2,y2]]}
    We model the wall as a vertical rectangular prism: extrude the segment by thickness/2 to both sides in XY,
    with Z spanning [0, height]. Returns list of AABB boxes labeled 'wall'.
    """
    sp = Path(scene_path)
    struct_file = sp / 'structure.json'
    walls: List[AABB] = []
    if not struct_file.exists():
        return walls
    try:
        with open(struct_file, 'r') as f:
            data = json.load(f)
        wall_items = data.get('walls', [])
        idx = 0
        for w in wall_items:
            try:
                loc = w.get('location', None)
                th = float(w.get('thickness', 0.2) or 0.2)
                h = float(w.get('height', 2.8) or 2.8)
                if not loc or len(loc) != 2:
                    continue
                x1, y1 = float(loc[0][0]), float(loc[0][1])
                x2, y2 = float(loc[1][0]), float(loc[1][1])
                p1 = np.array([x1, y1], dtype=float)
                p2 = np.array([x2, y2], dtype=float)
                seg = p2 - p1
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-6:
                    # degenerate: use a square of thickness around point
                    half = th * 0.5
                    xs = [x1 - half, x1 + half]
                    ys = [y1 - half, y1 + half]
                else:
                    dir_xy = seg / (seg_len + 1e-12)
                    # perpendicular in XY plane
                    n = np.array([-dir_xy[1], dir_xy[0]], dtype=float)
                    half = th * 0.5
                    # rectangle corners in XY
                    q1 = p1 + n * half
                    q2 = p1 - n * half
                    q3 = p2 + n * half
                    q4 = p2 - n * half
                    xs = [q1[0], q2[0], q3[0], q4[0]]
                    ys = [q1[1], q2[1], q3[1], q4[1]]
                bmin = np.array([min(xs), min(ys), 0.0], dtype=float)
                bmax = np.array([max(xs), max(ys), h], dtype=float)
                walls.append(AABB(id=f"wall_{idx}", label='wall', bmin=bmin, bmax=bmax))
                idx += 1
            except Exception:
                continue
    except Exception:
        return walls
    return walls


def camtoworld_from_pos_target(pos: np.ndarray, tgt: np.ndarray, up_vec: Optional[np.ndarray] = None) -> np.ndarray:
    pos = _safe_array(pos)
    tgt = _safe_array(tgt)
    f = tgt - pos
    f = f / (np.linalg.norm(f) + 1e-8)
    if up_vec is None:
        up_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up_vec = _safe_array(up_vec)
    r = np.cross(f, up_vec)
    if np.linalg.norm(r) > 1e-6:
        r = r / np.linalg.norm(r)
    else:
        r = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(r, f)
    c2w = np.eye(4, dtype=float)
    c2w[:3, 0] = -r
    c2w[:3, 1] = u
    c2w[:3, 2] = f
    c2w[:3, 3] = pos
    return c2w


def world_to_camera(viewmat: np.ndarray, pw: np.ndarray) -> np.ndarray:
    """viewmat = inverse(camtoworld)."""
    pw_h = np.concatenate([pw, np.ones(1, dtype=float)])
    pc_h = viewmat @ pw_h
    return pc_h[:3]


def project_point(K: np.ndarray, pc: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(pc[0]), float(pc[1]), float(pc[2])
    if z <= 1e-6:
        return float('inf'), float('inf'), z
    u = K[0, 0] * (x / z) + K[0, 2]
    v = K[1, 1] * (y / z) + K[1, 2]
    return u, v, z


def aabb_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    xs = [bmin[0], bmax[0]]
    ys = [bmin[1], bmax[1]]
    zs = [bmin[2], bmax[2]]
    corners = []
    for xi in xs:
        for yi in ys:
            for zi in zs:
                corners.append([xi, yi, zi])
    return np.array(corners, dtype=float)


def point_in_image(u: float, v: float, width: int, height: int, border: int = 0) -> bool:
    return (border <= u <= (width - 1 - border)) and (border <= v <= (height - 1 - border))


def intersects_ray_aabb(ray_o: np.ndarray, ray_d: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (t_enter, t_exit) if ray intersects AABB, else None.

    Robust to zero components in ray direction without producing NaNs.
    """
    tmin_overall = -float('inf')
    tmax_overall = +float('inf')
    for i in range(3):
        di = float(ray_d[i])
        oi = float(ray_o[i])
        bmin_i = float(bmin[i])
        bmax_i = float(bmax[i])
        if abs(di) < 1e-12:
            # Ray parallel to slab. If origin not within slab, no hit
            if oi < bmin_i or oi > bmax_i:
                return None
            # Otherwise, this axis imposes no constraint (t in [-inf, +inf])
            t1_i, t2_i = -float('inf'), +float('inf')
        else:
            invd = 1.0 / di
            t1_i = (bmin_i - oi) * invd
            t2_i = (bmax_i - oi) * invd
            if t1_i > t2_i:
                t1_i, t2_i = t2_i, t1_i
        tmin_overall = max(tmin_overall, t1_i)
        tmax_overall = min(tmax_overall, t2_i)
        if tmax_overall < tmin_overall:
            return None
    t_enter = tmin_overall
    t_exit = tmax_overall
    if t_exit >= max(t_enter, 0.0):
        return float(t_enter), float(t_exit)
    return None


def is_target_in_fov(scene_k: np.ndarray, camtoworld: np.ndarray, target_bmin: np.ndarray, target_bmax: np.ndarray, width: int, height: int, require_center: bool = True) -> bool:
    view = np.linalg.inv(camtoworld)
    corners = aabb_corners(target_bmin, target_bmax)
    pcs = np.array([world_to_camera(view, c) for c in corners])
    # If all corners are behind camera, return False
    if np.all(pcs[:, 2] <= 1e-6):
        return False
    uv = np.array([project_point(scene_k, pc) for pc in pcs])
    uvs = uv[:, :2]
    zs = uv[:, 2]
    # visible if any corner in front and projects within image bounds
    in_front = zs > 1e-6
    any_in_image = any(point_in_image(u, v, width, height, border=2) for (u, v), z in zip(uvs, zs) if z > 1e-6)
    if require_center:
        center = 0.5 * (target_bmin + target_bmax)
        pc = world_to_camera(view, center)
        u, v, z = project_point(scene_k, pc)
        if not (z > 1e-6 and point_in_image(u, v, width, height, border=2)):
            return False
    return bool(any_in_image and np.any(in_front))


def is_occluded_by_any(ray_o: np.ndarray, ray_tgt: np.ndarray, aabbs: List[AABB], target_id: Optional[str] = None, eps: float = 1e-3) -> bool:
    """Check if target point is occluded by any AABB.
    
    For backward compatibility when ray_tgt is a single point (center).
    Use is_box_occluded_by_any for full box occlusion checking.
    """
    ray_d = ray_tgt - ray_o
    dist = float(np.linalg.norm(ray_d))
    if dist < 1e-6:
        return False
    ray_d = ray_d / dist
    for box in aabbs:
        if target_id and (box.id == target_id):
            continue
        hit = intersects_ray_aabb(ray_o, ray_d, box.bmin, box.bmax)
        if hit is None:
            continue
        t_enter, _ = hit
        # if intersects before reaching target center, it's an occluder
        if 0.0 <= t_enter <= (dist - eps):
            return True
    return False


def is_box_occluded_by_any(ray_o: np.ndarray, target_bmin: np.ndarray, target_bmax: np.ndarray, aabbs: List[AABB], target_id: Optional[str] = None, eps: float = 1e-3) -> bool:
    """Check if entire target box is occluded by any AABB.
    
    Samples 9 points: 8 corners + center.
    Returns True if ANY of these points is occluded (strict: entire box must be unoccluded).
    
    Args:
        ray_o: Camera position
        target_bmin: Target box minimum corner
        target_bmax: Target box maximum corner
        aabbs: List of potential occluder AABBs
        target_id: ID of target box to exclude from occlusion test
        eps: Small epsilon for numerical stability
    
    Returns:
        True if any sampled point is occluded, False if all points are clear
    """
    # Sample 8 corners + center (9 points total)
    corners = aabb_corners(target_bmin, target_bmax)
    center = 0.5 * (target_bmin + target_bmax)
    sample_points = np.vstack([corners, center.reshape(1, 3)])
    
    # Check each sample point
    for pt in sample_points:
        ray_d = pt - ray_o
        dist = float(np.linalg.norm(ray_d))
        if dist < 1e-6:
            continue  # Skip if point coincides with camera
        ray_d = ray_d / dist
        
        # Check against all occluders
        for box in aabbs:
            if target_id and (box.id == target_id):
                continue
            hit = intersects_ray_aabb(ray_o, ray_d, box.bmin, box.bmax)
            if hit is None:
                continue
            t_enter, _ = hit
            # If ray hits occluder before reaching sample point, this point is occluded
            if 0.0 <= t_enter <= (dist - eps):
                return True  # At least one point is occluded -> box is occluded
    
    return False  # All points are clear -> box is not occluded


def is_point_inside_any(p: np.ndarray, aabbs: List[AABB]) -> bool:
    for b in aabbs:
        if np.all(p >= b.bmin) and np.all(p <= b.bmax):
            return True
    return False


def ensure_visibility(scene_path: str,
                      camera_pos: np.ndarray,
                      target_pos: np.ndarray,
                      K: Optional[np.ndarray] = None,
                      width: int = 400,
                      height: int = 400,
                      target_id: Optional[str] = None,
                      max_iters: int = 24,
                      step: float = 0.2) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Try small lateral/vertical adjustments to make target visible and unoccluded.
    Returns (new_camera_pos, info)
    info: {visible: bool, in_fov: bool, occluded: bool, attempts: int, adjusted: bool}
    """
    # Load objects and walls
    aabbs_objs = load_scene_aabbs(scene_path)
    print('Loaded {} objects', len(aabbs_objs))
    aabbs_walls = load_scene_wall_aabbs(scene_path)
    aabbs = aabbs_objs + aabbs_walls
    pos = _safe_array(camera_pos)
    tgt = _safe_array(target_pos)
    info = {'visible': False, 'in_fov': False, 'occluded': True, 'attempts': 0, 'adjusted': False}

    # load occupancy bounds if available
    occ_min = None
    occ_max = None
    rooms = None
    try:
        occ_path = Path(scene_path) / 'occupancy.json'
        if occ_path.exists():
            with open(occ_path, 'r') as f:
                occ = json.load(f)
            occ_min = np.array(occ.get('min', occ.get('lower', [None, None, None])), dtype=float)
            occ_max = np.array(occ.get('max', occ.get('upper', [None, None, None])), dtype=float)
    except Exception:
        occ_min = None
        occ_max = None

    # load room polygons from structure.json for 2D containment
    try:
        struct_path = Path(scene_path) / 'structure.json'
        if struct_path.exists():
            with open(struct_path, 'r') as f:
                struct = json.load(f)
            rooms = [np.array(r.get('profile', []), dtype=float) for r in struct.get('rooms', []) if r.get('profile')]
    except Exception:
        rooms = None

    if K is None:
        focal = float(width * 0.4)
        K = np.array([[focal, 0.0, width/2.0], [0.0, focal, height/2.0], [0.0, 0.0, 1.0]], dtype=float)
    else:
        K = _safe_array(K)

    def check(p_cam: np.ndarray) -> Tuple[bool, bool, bool]:
        c2w = camtoworld_from_pos_target(p_cam, tgt)
        # find target bbox if id known
        if target_id:
            tbox = next((b for b in aabbs if b.id == target_id), None)
        else:
            # approximate: choose the box whose center is nearest to target_pos
            if len(aabbs) == 0:
                tbox = None
            else:
                centers = np.array([(b.bmin + b.bmax) * 0.5 for b in aabbs])
                idx = int(np.argmin(np.linalg.norm(centers - tgt[None, :], axis=1)))
                tbox = aabbs[idx]
        if tbox is None:
            # without boxes, consider only center point in FOV
            in_fov = is_target_in_fov(K, c2w, tgt - 1e-3, tgt + 1e-3, width, height, require_center=True)
            occluded = False
            return in_fov and not occluded, in_fov, occluded
        in_fov = is_target_in_fov(K, c2w, tbox.bmin, tbox.bmax, width, height, require_center=True)
        occluded = is_box_occluded_by_any(p_cam, tbox.bmin, tbox.bmax, aabbs, target_id=tbox.id)
        return (in_fov and not occluded), in_fov, occluded

    # helper: point-in-polygon for room containment (ray casting)
    def point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[(i + 1) % n]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
        return inside

    # helper: point-in-polygon for room containment (ray casting)
    def point_in_polygon(pt: Tuple[float, float], poly: np.ndarray) -> bool:
        x, y = pt
        inside = False
        n = len(poly)
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[(i + 1) % n]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
        return inside

    # verify occupancy and room constraints
    def within_occupancy(p: np.ndarray) -> bool:
        if occ_min is None or occ_max is None:
            return True
        return np.all(p >= occ_min - 1e-6) and np.all(p <= occ_max + 1e-6)

    def clamp_occupancy(p: np.ndarray) -> np.ndarray:
        if occ_min is None or occ_max is None:
            return p
        return np.minimum(np.maximum(p, occ_min), occ_max)

    def point_in_any_room(p: np.ndarray) -> bool:
        if rooms is None or len(rooms) == 0:
            return True
        x, y = float(p[0]), float(p[1])
        for poly in rooms:
            try:
                if point_in_polygon((x, y), poly):
                    return True
            except Exception:
                continue
        return False

    def snap_to_nearest_room_centroid(p: np.ndarray) -> np.ndarray:
        if rooms is None or len(rooms) == 0:
            return p
        x, y = float(p[0]), float(p[1])
        best = None
        best_d = float('inf')
        for poly in rooms:
            if len(poly) == 0:
                continue
            cx, cy = float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
            d = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            if d < best_d:
                best_d = d
                best = (cx, cy)
        if best is None:
            return p
        q = p.copy()
        q[0], q[1] = best[0], best[1]
        return q

    # Clamp/snap initial position before checks
    pos = clamp_occupancy(pos)
    if not point_in_any_room(pos):
        pos = clamp_occupancy(snap_to_nearest_room_centroid(pos))

    # First try original (after clamp)
    ok, in_fov, occ = check(pos)
    info.update({'visible': ok, 'in_fov': in_fov, 'occluded': occ, 'attempts': 0, 'adjusted': False})

    if ok and within_occupancy(pos) and point_in_any_room(pos):
        return pos, info

    # generate search offsets in a spiral on right/up plane
    forward = tgt - pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)

    # offsets pattern: (dx, dy) in units of step; cover a diamond/spiral
    offsets = [(0, 0)]
    radius = 1
    while len(offsets) < max_iters:
        # diamond ring at current radius
        for dx in range(-radius, radius + 1):
            dy = radius - abs(dx)
            for sgn in (-1, 1):
                off = (dx, sgn * dy)
                if off not in offsets:
                    offsets.append(off)
                if len(offsets) >= max_iters:
                    break
            if len(offsets) >= max_iters:
                break
        radius += 1

    tried = 0
    for (dx, dy) in offsets[1:]:  # skip (0,0) already checked
        tried += 1
        candidate = pos + right * (dx * step) + up * (dy * step)
        # clamp to occupancy and snap to nearest room centroid if outside
        candidate = clamp_occupancy(candidate)
        if not point_in_any_room(candidate):
            candidate = clamp_occupancy(snap_to_nearest_room_centroid(candidate))
        # avoid placing camera inside any object
        if is_point_inside_any(candidate, aabbs):
            continue
        # final guards
        if not within_occupancy(candidate):
            continue
        if not point_in_any_room(candidate):
            continue
        ok, in_fov, occ = check(candidate)
        if ok:
            info.update({'visible': True, 'in_fov': in_fov, 'occluded': occ, 'attempts': tried, 'adjusted': True})
            return candidate, info

    # give up, return original with info
    info.update({'attempts': tried, 'adjusted': False})
    return pos, info


# -------------- NEW: 2D occlusion area computation -----------------

from typing import Iterable

def _poly_area(points: List[Tuple[float, float]]) -> float:
    """Signed area (abs gives area). points: [(x,y), ...] assumed in order."""
    if not points or len(points) < 3:
        return 0.0
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def _monotone_chain_convex_hull(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Convex hull (Monotone chain)."""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _suth_hodg_clip(subject: List[Tuple[float, float]],
                    clip:    List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sutherland–Hodgman polygon clipping: intersection(subject ∩ clip)."""
    def inside(p, a, b):
        # keep left side of edge a->b
        return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= 0
    def intersect(a1,a2,b1,b2):
        # segment a1a2 with b1b2
        x1,y1 = a1; x2,y2 = a2; x3,y3 = b1; x4,y4 = b2
        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(den) < 1e-12:
            return a2  # parallel/collinear fallback
        px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / den
        py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / den
        return (px, py)
    output = subject[:]
    for i in range(len(clip)):
        input_list = output[:]
        output = []
        A = clip[i]
        B = clip[(i+1) % len(clip)]
        if not input_list:
            break
        S = input_list[-1]
        for E in input_list:
            if inside(E, A, B):
                if not inside(S, A, B):
                    output.append(intersect(S, E, A, B))
                output.append(E)
            elif inside(S, A, B):
                output.append(intersect(S, E, A, B))
            S = E
    return output

def _rect_polygon(width: int, height: int) -> List[Tuple[float, float]]:
    return [(0,0), (width-1,0), (width-1,height-1), (0,height-1)]

def _project_box_polygon(K: np.ndarray, camtoworld: np.ndarray,
                         bmin: np.ndarray, bmax: np.ndarray,
                         width: int, height: int) -> Tuple[List[Tuple[float,float]], List[float]]:
    """Project AABB to image; return convex hull polygon (clipped to image) + per-corner depths (z of visible corners)."""
    view = np.linalg.inv(camtoworld)
    corners3d = aabb_corners(bmin, bmax)
    uv_pts: List[Tuple[float,float]] = []
    zs: List[float] = []
    for i in range(corners3d.shape[0]):
        pc = world_to_camera(view, corners3d[i])
        u, v, z = project_point(K, pc)
        if z > 1e-6:  # in front of camera
            uv_pts.append((u, v))
            zs.append(z)
    if len(uv_pts) < 3:
        return [], []
    # convex hull then clip to image rect
    hull = _monotone_chain_convex_hull(uv_pts)
    if len(hull) < 3:
        return [], []
    rect = _rect_polygon(width, height)
    clipped = _suth_hodg_clip(hull, rect)
    if len(clipped) < 3:
        return [], []
    return clipped, zs

from matplotlib.path import Path as _MplPath  # 放到文件顶部的 imports 里
import cv2  # 确保在文件顶部已经导入

import cv2

def occluded_area_on_image(
    ray_o: np.ndarray,
    target_bmin: np.ndarray,
    target_bmax: np.ndarray,
    aabbs: List[AABB],
    K: np.ndarray,
    camtoworld: np.ndarray,
    width: int,
    height: int,
    target_id: Optional[str] = None,
    depth_mode: str = "mean",   # "mean" or "min"
    return_per_occluder: bool = True,
) -> Dict[str, Any]:

    # ---------- 1) 目标投影 ----------
    tgt_poly, tgt_zs = _project_box_polygon(K, camtoworld, target_bmin, target_bmax, width, height)
    if len(tgt_poly) < 3 or not tgt_zs:
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 0.0,
            'occlusion_ratio_image': 0.0,
            'per_occluder': [] if return_per_occluder else None,
        }

    tgt_depth = (min(tgt_zs) if depth_mode == "min" else float(np.mean(tgt_zs)))
    tgt_poly_np = np.array(tgt_poly, dtype=float)

    # 目标的包围盒（做 ROI）
    x0 = int(max(0,     np.floor(np.min(tgt_poly_np[:, 0]))))
    x1 = int(min(width-1,  np.ceil(np.max(tgt_poly_np[:, 0]))))
    y0 = int(max(0,     np.floor(np.min(tgt_poly_np[:, 1]))))
    y1 = int(min(height-1, np.ceil(np.max(tgt_poly_np[:, 1]))))

    if x1 < x0 or y1 < y0:
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 0.0,
            'occlusion_ratio_image': 0.0,
            'per_occluder': [] if return_per_occluder else None,
        }

    # 可选 margin（略微扩大 ROI，防止边界数值波动）
    MARGIN = 2
    rx0 = max(0, x0 - MARGIN)
    ry0 = max(0, y0 - MARGIN)
    rx1 = min(width-1,  x1 + MARGIN)
    ry1 = min(height-1, y1 + MARGIN)

    # 在 ROI 内构造目标掩膜
    tgt_poly_roi = np.round(tgt_poly_np - np.array([rx0, ry0], dtype=float)).astype(np.int32)
    h_roi = ry1 - ry0 + 1
    w_roi = rx1 - rx0 + 1

    tgt_mask = np.zeros((h_roi, w_roi), np.uint8)
    cv2.fillPoly(tgt_mask, [tgt_poly_roi], 1)
    target_px = int(np.count_nonzero(tgt_mask))
    if target_px == 0:
        return {
            'target_area_px': 0.0,
            'occluded_area_px': 0.0,
            'visible_area_px': 0.0,
            'occlusion_ratio_target': 0.0,
            'occlusion_ratio_image': 0.0,
            'per_occluder': [] if return_per_occluder else None,
        }

    # ---------- 2) 构造更近遮挡者的并集掩膜（只在 ROI 内） ----------
    union_mask = np.zeros_like(tgt_mask, np.uint8)
    per_occ: List[Dict[str, Any]] = []

    for box in aabbs:
        if target_id and (box.id == target_id):
            continue

        occ_poly, occ_zs = _project_box_polygon(K, camtoworld, box.bmin, box.bmax, width, height)
        if len(occ_poly) < 3 or not occ_zs:
            continue

        occ_depth = (min(occ_zs) if depth_mode == "min" else float(np.mean(occ_zs)))
        # 必须比目标更近
        if occ_depth + 1e-6 >= tgt_depth:
            continue

        occ_poly_np = np.array(occ_poly, dtype=float)

        # 与 ROI 的 bbox 粗裁剪：若多边形 bbox 与 ROI 无交，则跳过
        ox0 = np.floor(np.min(occ_poly_np[:, 0])); oy0 = np.floor(np.min(occ_poly_np[:, 1]))
        ox1 = np.ceil(np.max(occ_poly_np[:, 0]));  oy1 = np.ceil(np.max(occ_poly_np[:, 1]))
        if (ox1 < rx0) or (ox0 > rx1) or (oy1 < ry0) or (oy0 > ry1):
            continue

        # 投影到 ROI 并填充
        occ_poly_roi = np.round(occ_poly_np - np.array([rx0, ry0], dtype=float)).astype(np.int32)
        occ_mask = np.zeros_like(tgt_mask, np.uint8)
        cv2.fillPoly(occ_mask, [occ_poly_roi], 1)

        # 只关心覆盖目标的区域
        overlap = (occ_mask == 1) & (tgt_mask == 1)
        if not np.any(overlap):
            continue

        # 更新并集（布尔或）
        union_mask |= occ_mask

        if return_per_occluder:
            px = int(np.count_nonzero(overlap))
            if px > 0:
                per_occ.append({'id': box.id, 'label': box.label, 'pixels': px})

        # 早停：如果并集已经覆盖了目标全部像素
        if np.count_nonzero(union_mask & tgt_mask) == target_px:
            break

    # ---------- 3) 计算遮挡像素 ----------
    occluded_mask = (union_mask == 1) & (tgt_mask == 1)
    occluded_px = int(np.count_nonzero(occluded_mask))
    visible_px = target_px - occluded_px

    result = {
        'target_area_px': float(target_px),
        'occluded_area_px': float(occluded_px),
        'visible_area_px': float(visible_px),
        'occlusion_ratio_target': (occluded_px / target_px) if target_px > 0 else 0.0,
        'occlusion_ratio_image': (occluded_px / float(width * height)),
    }
    if return_per_occluder:
        result['per_occluder'] = per_occ
    return result
