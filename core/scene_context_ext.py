"""
core/scene_context_ext.py
SceneContext extension methods required by the evaluation module.

These methods are attached to the existing SceneContext class via monkey-patch
on import, so existing code continues to work and new code (predicates,
region_generators, …) can call ctx.method(...) directly.

Conventions
-----------
* Camera state is always (cam_pos: np.ndarray(3,), cam_target: np.ndarray(3,)).
* Distances are metres, angles are degrees.
* Object ids are strings (AABB.id from labels.json).
* All methods are pure: no side-effects on the SceneContext.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .scene_context import SceneContext, DEFAULT_CAMERA_HEIGHT
from ..utils.occlusion import (
    AABB,
    aabb_corners,
    camtoworld_from_pos_target,
    intersects_ray_aabb,
    is_box_occluded_by_any,
    project_point,
    world_to_camera,
)


def _intrinsics_K(ctx: SceneContext) -> np.ndarray:
    """Return intrinsics K as a 3x3 ndarray (handles list/dict/ndarray inputs)."""
    intr = ctx.intrinsics
    if isinstance(intr, dict):
        K = intr.get("K")
    else:
        K = intr
    return np.asarray(K, dtype=float)


def _intrinsics_wh(ctx: SceneContext) -> Tuple[int, int]:
    intr = ctx.intrinsics
    if isinstance(intr, dict):
        return int(intr.get("width", 400)), int(intr.get("height", 400))
    return 400, 400


# ---------------------------------------------------------------------------
# AABB / object lookup
# ---------------------------------------------------------------------------

def _get_aabb(self: SceneContext, obj_id: str) -> Optional[AABB]:
    """Return the AABB object by id, or None."""
    return self.get_object_by_id(obj_id)


def _get_aabb_dict(self: SceneContext, obj_id: str) -> Optional[dict]:
    """Return a {min, max} dict for the AABB; None if not found."""
    box = self.get_object_by_id(obj_id)
    if box is None:
        return None
    return {"min": box.bmin.tolist(), "max": box.bmax.tolist()}


def _get_object_centre(self: SceneContext, obj_id: str) -> Optional[np.ndarray]:
    box = self.get_object_by_id(obj_id)
    if box is None:
        return None
    return 0.5 * (box.bmin + box.bmax)


def _get_annotation(self: SceneContext, obj_id: str, key: str):
    """
    Return raw annotation field from labels.json for obj_id (e.g. front_normal).
    Returns None if missing.
    """
    for entry in self._labels_raw:
        if str(entry.get("ins_id", "")) == str(obj_id) and key in entry:
            return entry[key]
    return None


# ---------------------------------------------------------------------------
# Visibility / occlusion (corner-based, dense-AABB robust)
# ---------------------------------------------------------------------------

# Visibility tuning.  These two constants set the policy for how strictly a
# "corner" is judged occluded.  See the docstring of ``_visible_corner_mask``.
_VIS_CORNER_INSET = 0.05          # fraction of half-extent to pull corners inward
_VIS_CORNER_INSET_MAX = 0.05      # never inset more than 5 cm
_VIS_REL_DEPTH_EPS = 0.02         # blocker must be ≥ 2% closer than target


def _filter_blockers_for_target(target_box, all_blockers):
    """Exclude blockers that overlap or touch the target AABB.

    Rationale: in dense AABB scenes (floor/walls flush with object bottoms,
    adjacent objects sharing a face) a touching blocker would otherwise be
    counted as an occluder of every ray reaching the target's surface.
    Anything that physically intersects/contacts the target cannot occlude it
    from any external viewpoint, so we drop those.
    """
    tmin = np.asarray(target_box.bmin, dtype=float)
    tmax = np.asarray(target_box.bmax, dtype=float)
    out = []
    for b in all_blockers:
        if b.id == target_box.id:
            continue
        bmin = np.asarray(b.bmin, dtype=float)
        bmax = np.asarray(b.bmax, dtype=float)
        # AABB overlap test with a small slack so touching boxes are excluded.
        slack = 1e-3
        if np.all(bmin <= tmax + slack) and np.all(bmax >= tmin - slack):
            continue
        out.append(b)
    return out


def _inset_corners(bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """Return 8 AABB corners pulled inward by a small inset.

    The inset is ``_VIS_CORNER_INSET`` of each half-extent capped at
    ``_VIS_CORNER_INSET_MAX`` (metres).  This is the single line that makes the
    corner-based visibility test robust to neighbours that share a face with
    the target: shrunk corners are guaranteed to be strictly *inside* the
    target volume, so a tangent neighbour's AABB does not contain them.
    """
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)
    half = 0.5 * (bmax - bmin)
    inset = np.minimum(half * _VIS_CORNER_INSET, _VIS_CORNER_INSET_MAX)
    lo = bmin + inset
    hi = bmax - inset
    return np.array([
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [lo[0], hi[1], hi[2]],
        [hi[0], hi[1], hi[2]],
    ], dtype=float)


def _corner_occluded(cam_pos: np.ndarray, corner: np.ndarray,
                     blockers) -> bool:
    """True iff some blocker AABB intersects the cam→corner ray strictly
    closer than the corner, by a relative margin of ``_VIS_REL_DEPTH_EPS``.

    Single ray, no 9-point sampling — the caller is already sampling 8
    corners, so per-corner cost dominates the visibility budget.
    """
    ray = corner - cam_pos
    dist = float(np.linalg.norm(ray))
    if dist < 1e-6:
        return False
    ray /= dist
    eps = max(1e-3, dist * _VIS_REL_DEPTH_EPS)
    for b in blockers:
        hit = intersects_ray_aabb(cam_pos, ray, b.bmin, b.bmax)
        if hit is None:
            continue
        t_enter, t_exit = hit
        if t_exit <= 1e-3:
            continue  # blocker entirely behind the camera
        if 0.0 <= t_enter <= (dist - eps):
            return True
    return False


def _visible_corner_mask(
    self: SceneContext,
    obj_id: str,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
) -> np.ndarray:
    """
    Returns a length-8 bool array indicating which AABB corners are
    (in front of camera) AND (project inside image) AND (not occluded).
    """
    box = self.get_object_by_id(obj_id)
    if box is None:
        return np.zeros(8, dtype=bool)

    c2w = camtoworld_from_pos_target(np.asarray(cam_pos), np.asarray(cam_target))
    view = np.linalg.inv(c2w)
    K = _intrinsics_K(self)
    width, height = _intrinsics_wh(self)

    corners = _inset_corners(box.bmin, box.bmax)
    blockers = _filter_blockers_for_target(box, self.all_blockers)

    out = np.zeros(8, dtype=bool)
    for i, c in enumerate(corners):
        pc = world_to_camera(view, c)
        if pc[2] <= 1e-6:
            continue
        u, v, _ = project_point(K, pc)
        if not (0 <= u < width and 0 <= v < height):
            continue
        if _corner_occluded(np.asarray(cam_pos), c, blockers):
            continue
        out[i] = True
    return out


def _occlusion_fraction(
    self: SceneContext,
    obj_id: str,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
) -> float:
    """
    Approximate fraction of AABB corners occluded by other geometry.
    Returns a float in [0, 1]. Uses corner-trace approximation (cheap; ok for AABB-only scenes).

    Definition:
        occlusion_fraction = (corners_in_front_in_image_BUT_blocked) /
                             max(1, corners_in_front_in_image)
    Corners that are simply behind the camera or out of frame don't count
    as "occluded" — they're "out of frame".
    """
    box = self.get_object_by_id(obj_id)
    if box is None:
        return 1.0

    c2w = camtoworld_from_pos_target(np.asarray(cam_pos), np.asarray(cam_target))
    view = np.linalg.inv(c2w)
    K = _intrinsics_K(self)
    width, height = _intrinsics_wh(self)

    corners = _inset_corners(box.bmin, box.bmax)
    blockers = _filter_blockers_for_target(box, self.all_blockers)

    in_frame = 0
    blocked = 0
    for c in corners:
        pc = world_to_camera(view, c)
        if pc[2] <= 1e-6:
            continue
        u, v, _ = project_point(K, pc)
        if not (0 <= u < width and 0 <= v < height):
            continue
        in_frame += 1
        if _corner_occluded(np.asarray(cam_pos), c, blockers):
            blocked += 1
    if in_frame == 0:
        return 1.0
    return blocked / in_frame


def _primary_occluder(
    self: SceneContext,
    obj_id: str,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
) -> Optional[str]:
    """
    Return the id of the object/wall that occludes the most corners of `obj_id`,
    or None if no occlusion.
    """
    box = self.get_object_by_id(obj_id)
    if box is None:
        return None
    blockers = [b for b in self.all_blockers if b.id != box.id]
    if not blockers:
        return None

    corners = aabb_corners(box.bmin, box.bmax)
    counts: dict[str, int] = {}
    for c in corners:
        # Find first blocker along the ray cam_pos -> c
        ray_o = np.asarray(cam_pos)
        ray_d = c - ray_o
        depth = float(np.linalg.norm(ray_d))
        if depth < 1e-6:
            continue
        ray_d = ray_d / depth
        best_t = depth - 1e-3
        best_id = None
        for b in blockers:
            hit = intersects_ray_aabb(ray_o, ray_d, b.bmin, b.bmax)
            if hit is None:
                continue
            t_enter, _ = hit
            if 1e-3 < t_enter < best_t:
                best_t = t_enter
                best_id = b.id
        if best_id is not None:
            counts[best_id] = counts.get(best_id, 0) + 1

    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Image-space helpers (for ScaleBand / AspectExposed / SeparationOnImage / Centered)
# ---------------------------------------------------------------------------

def _project_aabb_corners(
    self: SceneContext,
    obj_id: str,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (uv: (8,2) image-pixel coords, depth: (8,) z, in_frame_mask: (8,) bool).
    """
    box = self.get_object_by_id(obj_id)
    if box is None:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)
    c2w = camtoworld_from_pos_target(np.asarray(cam_pos), np.asarray(cam_target))
    view = np.linalg.inv(c2w)
    K = _intrinsics_K(self)
    width, height = _intrinsics_wh(self)

    corners = aabb_corners(box.bmin, box.bmax)
    uv = np.full((8, 2), np.nan, dtype=float)
    in_frame = np.zeros(8, dtype=bool)
    for i, c in enumerate(corners):
        pc = world_to_camera(view, c)
        if pc[2] <= 1e-6:
            continue
        u, v, _ = project_point(K, pc)
        uv[i] = (u, v)
        if 0 <= u < width and 0 <= v < height:
            in_frame[i] = True
    return uv, in_frame


def _project_point(
    self: SceneContext,
    pos: np.ndarray,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Project a single world point to image pixels; None if behind camera."""
    c2w = camtoworld_from_pos_target(np.asarray(cam_pos), np.asarray(cam_target))
    view = np.linalg.inv(c2w)
    K = _intrinsics_K(self)
    pc = world_to_camera(view, np.asarray(pos))
    if pc[2] <= 1e-6:
        return None
    u, v, _ = project_point(K, pc)
    return (float(u), float(v))


# ---------------------------------------------------------------------------
# Camera intrinsics helpers
# ---------------------------------------------------------------------------

def _image_width(self: SceneContext) -> int:
    return _intrinsics_wh(self)[0]


def _image_height(self: SceneContext) -> int:
    return _intrinsics_wh(self)[1]


def _default_hfov_deg(self: SceneContext) -> float:
    """Compute horizontal field-of-view in degrees from intrinsics K."""
    K = _intrinsics_K(self)
    width = self._image_width()
    fx = float(K[0, 0])
    return 2.0 * math.degrees(math.atan(0.5 * width / fx))


# ---------------------------------------------------------------------------
# Room helpers (per-id polygons)
# ---------------------------------------------------------------------------

def _ensure_room_index(self: SceneContext) -> None:
    """
    Build a room-index dict mapping room_id (str) → polygon (np.ndarray (N,2)).
    Reads structure.json if present, falls back to ordinal ids r0, r1, ... .
    Idempotent.
    """
    if hasattr(self, "_room_polygons_by_id") and self._room_polygons_by_id is not None:
        return
    self._room_polygons_by_id = {}
    sf = self.scene_path / "structure.json"
    if sf.exists():
        try:
            import json
            with open(sf, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            rooms = data.get("rooms", [])
            for i, r in enumerate(rooms):
                rid = str(r.get("id", f"r{i}"))
                poly = r.get("profile") or r.get("polygon") or r.get("vertices") or r.get("contour")
                if poly is not None and len(poly) >= 3:
                    arr = np.asarray(poly, dtype=float)
                    if arr.shape[1] > 2:
                        arr = arr[:, :2]
                    self._room_polygons_by_id[rid] = arr
        except Exception:
            pass
    # Fallback: ordinal ids matching SceneContext.room_polygons
    if not self._room_polygons_by_id:
        for i, poly in enumerate(self.room_polygons):
            self._room_polygons_by_id[f"r{i}"] = np.asarray(poly, dtype=float)


def _room_ids(self: SceneContext) -> List[str]:
    self._ensure_room_index()
    return list(self._room_polygons_by_id.keys())


def _is_position_in_room(self: SceneContext, pos, room_id: Optional[str] = None) -> bool:
    """
    Check whether a 3D position is inside a room polygon (in XY).
    If room_id is None, returns True if pos is inside any room.
    """
    from ..bench_generation.batch_utils import point_in_poly
    pos_arr = np.asarray(pos)
    x, y = float(pos_arr[0]), float(pos_arr[1])

    if room_id is None:
        for poly in self.room_polygons:
            if point_in_poly(x, y, poly):
                return True
        return False

    self._ensure_room_index()
    poly = self._room_polygons_by_id.get(room_id)
    if poly is None:
        return False
    return point_in_poly(x, y, poly)


def _room_id_at(self: SceneContext, pos) -> Optional[str]:
    """Return the room_id containing pos, or None."""
    from ..bench_generation.batch_utils import point_in_poly
    self._ensure_room_index()
    pos_arr = np.asarray(pos)
    x, y = float(pos_arr[0]), float(pos_arr[1])
    for rid, poly in self._room_polygons_by_id.items():
        if point_in_poly(x, y, poly):
            return rid
    return None


def _room_id_for_object(self: SceneContext, obj_id: str) -> Optional[str]:
    centre = self._get_object_centre(obj_id)
    if centre is None:
        return None
    return self._room_id_at(centre)


# ---------------------------------------------------------------------------
# Scene-bounds helper (for action lattice)
# ---------------------------------------------------------------------------

def _get_scene_bounds(self: SceneContext) -> dict:
    """
    Compute axis-aligned XY bounds covering all room polygons.
    Returns {x_min, x_max, y_min, y_max, agent_height}.
    """
    if self.room_polygons:
        all_xy = np.concatenate([np.asarray(p, dtype=float) for p in self.room_polygons], axis=0)
    else:
        # Fall back to object AABB extents
        if not self.objects:
            return {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0,
                    "agent_height": DEFAULT_CAMERA_HEIGHT}
        bmins = np.array([o.bmin for o in self.objects])
        bmaxs = np.array([o.bmax for o in self.objects])
        all_xy = np.concatenate([bmins[:, :2], bmaxs[:, :2]], axis=0)
    return {
        "x_min": float(all_xy[:, 0].min()),
        "x_max": float(all_xy[:, 0].max()),
        "y_min": float(all_xy[:, 1].min()),
        "y_max": float(all_xy[:, 1].max()),
        "agent_height": float(DEFAULT_CAMERA_HEIGHT),
    }


# ---------------------------------------------------------------------------
# Plane normals (for OrthogonalToPlane)
# ---------------------------------------------------------------------------

def _get_plane_normal(self: SceneContext, plane_key: str) -> Optional[np.ndarray]:
    """
    Look up a plane normal by key. Supports:
      • Wall ids (e.g., 'wall_3') — normal = horizontal perp of wall segment
      • Portal ids (e.g., 'portal_0') — same as the portal's plane normal
      • Object face keys formatted '<obj_id>:<side>' where side ∈ {front,back,left,right,top,bottom}
      • A pair-axis key '<obj_a>:<obj_b>' → vertical plane containing the A–B segment

    Returns a unit np.ndarray(3,) or None.
    """
    if not isinstance(plane_key, str):
        return None

    # Generic "vertical" / "horizontal" placeholders — the instantiator could
    # not bind a concrete wall/portal, so accept any forward direction as
    # satisfying OrthogonalToPlane.
    if plane_key in ("vertical", "horizontal", "any", ""):
        # Return None so the predicate hits its permissive fallback branch.
        return None

    # Pair vertical-plane: 'obj_a:obj_b' (no side keyword)
    if ":" in plane_key:
        a, b = plane_key.split(":", 1)
        side_kw = {"front", "back", "left", "right", "top", "bottom"}
        if b in side_kw:
            return self._object_face_normal(a, b)
        return self._pair_vertical_plane_normal(a, b)

    # Wall or portal
    if plane_key.startswith("wall_"):
        for w in self.wall_aabbs:
            if w.id == plane_key:
                # Normal = whichever XY axis is the shorter span
                dx = w.bmax[0] - w.bmin[0]
                dy = w.bmax[1] - w.bmin[1]
                if dx <= dy:
                    return np.array([1.0, 0.0, 0.0])
                return np.array([0.0, 1.0, 0.0])
    if plane_key.startswith("portal_"):
        portals = self._portals()
        for p in portals:
            if p["id"] == plane_key:
                return np.asarray(p["normal"], dtype=float)
    return None


def _object_face_normal(self: SceneContext, obj_id: str, side: str) -> Optional[np.ndarray]:
    """
    Return the outward normal of the named face of an object.
    Uses front_normal annotation if present; otherwise falls back to AABB axes.
    """
    front = self._get_annotation(obj_id, "front_normal")
    if front is not None:
        fn = np.asarray(front, dtype=float)
        fn /= np.linalg.norm(fn) + 1e-9
        if side == "front": return fn
        if side == "back":  return -fn
        # Left/right derived from cross product with up
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fn, up)
        right /= np.linalg.norm(right) + 1e-9
        if side == "right": return right
        if side == "left":  return -right
        if side == "top":   return up
        if side == "bottom": return -up
    # Fallback: AABB axes
    AXES = {
        "front": np.array([0.0, 1.0, 0.0]),
        "back":  np.array([0.0, -1.0, 0.0]),
        "left":  np.array([-1.0, 0.0, 0.0]),
        "right": np.array([1.0, 0.0, 0.0]),
        "top":   np.array([0.0, 0.0, 1.0]),
        "bottom": np.array([0.0, 0.0, -1.0]),
    }
    return AXES.get(side)


def _pair_vertical_plane_normal(
    self: SceneContext, obj_a_id: str, obj_b_id: str
) -> Optional[np.ndarray]:
    """Normal of the vertical plane containing the A–B centre segment (in XY)."""
    ca = self._get_object_centre(obj_a_id)
    cb = self._get_object_centre(obj_b_id)
    if ca is None or cb is None:
        return None
    seg = (cb - ca)[:2]
    if np.linalg.norm(seg) < 1e-6:
        return None
    seg /= np.linalg.norm(seg)
    # Normal in XY perpendicular to segment
    return np.array([-seg[1], seg[0], 0.0])


# ---------------------------------------------------------------------------
# Portals (heuristic, derived from structure.json or wall gaps)
# ---------------------------------------------------------------------------

def _portals(self: SceneContext) -> List[dict]:
    """
    Return cached list of portal dicts:
        {id, position, normal, width, height, room_a, room_b, type}

    Supports two schemas in structure.json:
      • Modern: ``doors``/``openings``/``portals`` lists (one dict per door).
      • Spatial-Training-Room dataset: a ``holes`` list whose entries each have
        a ``type`` (e.g. "DOOR", "OPENING", "WINDOW") and a ``profile`` of 4
        3D corners describing the rectangular opening cut in the wall.

    For the ``holes`` schema we:
      • take the centroid of the 4 corners as the portal position,
      • derive the in-plane horizontal width and vertical height,
      • compute the plane normal from two non-parallel edges,
      • auto-assign ``room_a``/``room_b`` by polygon containment of points
        offset by ``thickness + 0.05 m`` on either side of the door plane.
    """
    if hasattr(self, "_portal_cache") and self._portal_cache is not None:
        return self._portal_cache
    self._portal_cache = []
    sf = self.scene_path / "structure.json"
    if not sf.exists():
        return self._portal_cache
    try:
        import json
        with open(sf, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return self._portal_cache

    # -- modern schema --
    legacy_items = data.get("doors") or data.get("openings") or data.get("portals") or []
    for i, d in enumerate(legacy_items):
        loc = d.get("location") or d.get("center") or d.get("position")
        if loc is None:
            continue
        if isinstance(loc, list) and len(loc) > 0 and isinstance(loc[0], list):
            p1 = np.asarray(loc[0], dtype=float)
            p2 = np.asarray(loc[1], dtype=float)
            mid = 0.5 * (p1 + p2)
            seg = p2 - p1
            seg2 = seg[:2] / max(1e-9, np.linalg.norm(seg[:2]))
            normal = np.array([-seg2[1], seg2[0], 0.0])
            pos = np.array([mid[0], mid[1], DEFAULT_CAMERA_HEIGHT])
            width = float(np.linalg.norm(seg[:2]))
            height = 2.2
        else:
            pos_arr = np.asarray(loc, dtype=float)
            if pos_arr.shape[0] == 2:
                pos = np.array([pos_arr[0], pos_arr[1], DEFAULT_CAMERA_HEIGHT])
            else:
                pos = pos_arr.astype(float)
            normal_raw = d.get("normal", [0.0, 1.0, 0.0])
            normal = np.asarray(normal_raw, dtype=float)
            if normal.shape[0] == 2:
                normal = np.array([normal[0], normal[1], 0.0])
            width = float(d.get("width", 0.9))
            height = float(d.get("height", 2.2))
        self._portal_cache.append({
            "id": str(d.get("id", f"portal_{i}")),
            "type": str(d.get("type", "DOOR")).upper(),
            "position": pos,
            "normal": normal / (np.linalg.norm(normal) + 1e-9),
            "width": width,
            "height": height,
            "thickness": float(d.get("thickness", 0.12)),
            "room_a": str(d.get("room_a", "")) or None,
            "room_b": str(d.get("room_b", "")) or None,
        })

    # -- holes schema (this dataset) --
    holes = data.get("holes", []) or []
    for j, h in enumerate(holes):
        prof = h.get("profile")
        if not prof or len(prof) < 4:
            continue
        corners = np.asarray(prof, dtype=float)[:, :3]
        centre = corners.mean(axis=0)
        # Profiles may be 4-corner rectangles OR many-point arches.  Treat the
        # profile as a planar polygon and derive width/height from the bbox of
        # the profile after collapsing the plane-thin axis.  Plane normal =
        # direction of zero variance in the profile.
        bmin = corners.min(axis=0)
        bmax = corners.max(axis=0)
        spans = bmax - bmin
        # Plane axis = axis with smallest span (≤ a couple of cm for a flat
        # rectangle cut into a wall).
        plane_axis = int(np.argmin(spans))
        in_plane = [k for k in (0, 1, 2) if k != plane_axis]
        # Normal points along the plane axis; sign is arbitrary — refine below.
        normal = np.zeros(3, dtype=float)
        normal[plane_axis] = 1.0
        # Height = span along Z; width = span along the in-plane non-Z axis
        # (almost always one of X or Y, since walls are vertical).
        if plane_axis == 2:
            # Horizontal opening lying flat — unusual for a door; use XY bbox
            width = float(max(spans[0], spans[1]))
            height = float(spans[2])
        else:
            xy_axis = 1 if plane_axis == 0 else 0
            width = float(spans[xy_axis])
            height = float(spans[2])
        if width < 0.10 or height < 1.0:
            # Sliver / decorative cutout — not a real portal
            continue

        thickness = float(h.get("thickness", 0.12))
        # Sample points offset by (thickness/2 + 0.10) along ±normal at floor height
        offset = thickness * 0.5 + 0.10
        probe_xy = np.array([centre[0], centre[1], 0.0])
        side_pos = probe_xy + normal * offset
        side_neg = probe_xy - normal * offset
        room_a = self._room_id_at(side_pos) if hasattr(self, "_room_id_at") else None
        room_b = self._room_id_at(side_neg) if hasattr(self, "_room_id_at") else None

        pos = np.array([centre[0], centre[1], DEFAULT_CAMERA_HEIGHT])
        self._portal_cache.append({
            "id": f"portal_{len(self._portal_cache)}",
            "type": str(h.get("type", "DOOR")).upper(),
            "position": pos,
            "normal": normal,
            "width": width,
            "height": height,
            "thickness": thickness,
            "room_a": room_a,
            "room_b": room_b,
        })
    return self._portal_cache


def _get_portal_position(self: SceneContext, portal_id: str) -> Optional[np.ndarray]:
    for p in self._portals():
        if p["id"] == portal_id:
            return p["position"]
    return None


def _portal_dict(self: SceneContext, portal_id: str) -> Optional[dict]:
    for p in self._portals():
        if p["id"] == portal_id:
            return p
    return None


# ---------------------------------------------------------------------------
# Region samplers (used by region_generators.py orthogonal-to-portal etc.)
# ---------------------------------------------------------------------------

def _sample_orthogonal_to_portal(
    self: SceneContext,
    portal_id: str,
    dist_min: float = 0.4,
    dist_max: float = 1.5,
    ortho_max_deg: float = 15.0,
    n: int = 32,
    rng: Optional[random.Random] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Sample (cam_pos, cam_target) pairs facing a portal near-orthogonally.
    Camera looks along the portal normal toward the portal centre.
    """
    rng = rng or random.Random()
    portal = self._portal_dict(portal_id)
    if portal is None:
        return []
    pos = portal["position"]
    n_vec = portal["normal"]
    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    attempts = 0
    while len(samples) < n and attempts < n * 10:
        attempts += 1
        d = rng.uniform(dist_min, dist_max)
        # Random offset within ±ortho_max_deg of -normal direction
        ang = math.radians(rng.uniform(-ortho_max_deg, ortho_max_deg))
        # Sample camera position on either side
        side = rng.choice([1.0, -1.0])
        # Rotate -side*n_vec by ang in XY
        base_dir = -side * n_vec
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        rot = np.array([
            base_dir[0] * cos_a - base_dir[1] * sin_a,
            base_dir[0] * sin_a + base_dir[1] * cos_a,
            base_dir[2],
        ])
        cam_pos = pos + rot * d
        cam_pos[2] = DEFAULT_CAMERA_HEIGHT
        if not self.is_position_valid(cam_pos):
            continue
        cam_target = pos.copy()
        cam_target[2] = DEFAULT_CAMERA_HEIGHT
        samples.append((cam_pos, cam_target))
    return samples


def _sample_positions_anywhere(
    self: SceneContext, n: int = 32, rng: Optional[random.Random] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Sample (pos, target) anywhere in the scene; targets face the room centroid."""
    rng = rng or random.Random()
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    positions = self.sample_positions_in_room(num_points=n, rng=rs)
    out = []
    for pos in positions:
        # Default target: 1m forward in a random yaw
        yaw = rng.uniform(0, 2 * math.pi)
        tgt = pos + np.array([math.sin(yaw), math.cos(yaw), 0.0])
        out.append((pos, tgt))
    return out


# ---------------------------------------------------------------------------
# Apply patches
# ---------------------------------------------------------------------------

_BINDINGS = {
    "get_aabb": _get_aabb,
    "get_aabb_dict": _get_aabb_dict,
    "get_object_centre": _get_object_centre,
    "get_annotation": _get_annotation,
    "visible_corner_mask": _visible_corner_mask,
    "occlusion_fraction": _occlusion_fraction,
    "primary_occluder": _primary_occluder,
    "project_aabb_corners": _project_aabb_corners,
    "project_world_point": _project_point,
    "_image_width": _image_width,
    "_image_height": _image_height,
    "default_hfov_deg": _default_hfov_deg,
    "_ensure_room_index": _ensure_room_index,
    "room_ids": _room_ids,
    "is_position_in_room_id": _is_position_in_room,
    "room_id_at": _room_id_at,
    "room_id_for_object": _room_id_for_object,
    "get_scene_bounds": _get_scene_bounds,
    "get_plane_normal": _get_plane_normal,
    "_object_face_normal": _object_face_normal,
    "_pair_vertical_plane_normal": _pair_vertical_plane_normal,
    "_portals": _portals,
    "get_portal_position": _get_portal_position,
    "_portal_dict": _portal_dict,
    "sample_orthogonal_to_portal": _sample_orthogonal_to_portal,
    "sample_positions_anywhere": _sample_positions_anywhere,
}

for _name, _fn in _BINDINGS.items():
    if not hasattr(SceneContext, _name):
        setattr(SceneContext, _name, _fn)
    # Also expose under the original module-level name (with leading underscore)
    # because several extension methods call e.g. ``self._get_object_centre`` /
    # ``self._get_annotation`` directly.
    _alias = f"_{_name}" if not _name.startswith("_") else _name
    if not hasattr(SceneContext, _alias):
        setattr(SceneContext, _alias, _fn)
