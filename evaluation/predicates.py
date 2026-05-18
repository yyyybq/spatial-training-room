"""
evaluation/predicates.py

Tier-1 predicate primitives (12 functions).

All predicates share the signature:

    Predicate(cam_pos, cam_target, hfov_deg, scene_ctx, *, **kwargs) -> bool

Where:
    cam_pos    : np.ndarray (3,)   — world-frame camera position
    cam_target : np.ndarray (3,)   — world-frame look-at target
    hfov_deg   : float             — horizontal field-of-view in degrees
    scene_ctx  : SceneContext (with scene_context_ext patches loaded)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

# Touch the extension module so SceneContext gets monkey-patched
from ..core import scene_context_ext  # noqa: F401
from ..core.scene_context import SceneContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward(cam_pos: np.ndarray, cam_target: np.ndarray) -> np.ndarray:
    f = np.asarray(cam_target) - np.asarray(cam_pos)
    n = np.linalg.norm(f)
    return f / (n + 1e-9)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b) + 1e-9)
    c = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _signed_yaw_xy(forward: np.ndarray, dir_to: np.ndarray) -> float:
    """+ = right of forward; − = left."""
    f = np.array([forward[0], forward[1]], dtype=float)
    d = np.array([dir_to[0], dir_to[1]], dtype=float)
    f /= np.linalg.norm(f) + 1e-9
    d /= np.linalg.norm(d) + 1e-9
    cross = f[0] * d[1] - f[1] * d[0]
    dot = float(np.clip(np.dot(f, d), -1.0, 1.0))
    angle = math.degrees(math.acos(dot))
    return -angle if cross < 0 else angle


# ---------------------------------------------------------------------------
# Tier-1 primitives
# ---------------------------------------------------------------------------

def Visible(cam_pos, cam_target, hfov_deg, scene_ctx,
            *, obj: str, min_corners: int = 4, max_occ: float = 0.30) -> bool:
    if scene_ctx.get_aabb(obj) is None:
        return False
    mask = scene_ctx.visible_corner_mask(obj, cam_pos, cam_target)
    if int(mask.sum()) < min_corners:
        return False
    occ = scene_ctx.occlusion_fraction(obj, cam_pos, cam_target)
    return occ <= max_occ


def DistanceBand(cam_pos, cam_target, hfov_deg, scene_ctx,
                 *, obj: str, d_min: float, d_max: float) -> bool:
    centre = scene_ctx.get_object_centre(obj)
    if centre is None:
        return False
    d = float(np.linalg.norm(np.asarray(cam_pos) - centre))
    return d_min <= d <= d_max


def AspectExposed(cam_pos, cam_target, hfov_deg, scene_ctx,
                  *, obj: str, min_ratio: float = 0.30) -> bool:
    uv, in_frame = scene_ctx.project_aabb_corners(obj, cam_pos, cam_target)
    if uv.size == 0:
        return False
    pts = uv[in_frame]
    if pts.shape[0] < 4:
        return False
    du = float(pts[:, 0].max() - pts[:, 0].min())
    dv = float(pts[:, 1].max() - pts[:, 1].min())
    if du < 1e-3 or dv < 1e-3:
        return False
    return (min(du, dv) / max(du, dv)) >= min_ratio


def Centered(cam_pos, cam_target, hfov_deg, scene_ctx,
             *, obj: str, theta_deg: float = 25.0) -> bool:
    centre = scene_ctx.get_object_centre(obj)
    if centre is None:
        return False
    f = _forward(cam_pos, cam_target)
    d = centre - np.asarray(cam_pos)
    if np.linalg.norm(d) < 1e-6:
        return True
    return _angle_between(f, d) <= theta_deg


def ScaleBand(cam_pos, cam_target, hfov_deg, scene_ctx,
              *, obj: str, s_min: float = 0.05, s_max: float = 0.70) -> bool:
    uv, in_frame = scene_ctx.project_aabb_corners(obj, cam_pos, cam_target)
    if uv.size == 0 or in_frame.sum() < 2:
        return False
    pts = uv[in_frame]
    du = float(pts[:, 0].max() - pts[:, 0].min())
    dv = float(pts[:, 1].max() - pts[:, 1].min())
    img_area = float(scene_ctx._image_width() * scene_ctx._image_height())
    if img_area <= 0:
        return False
    s = (du * dv) / img_area
    return s_min <= s <= s_max


def BearingWithin(cam_pos, cam_target, hfov_deg, scene_ctx,
                  *, obj: Optional[str] = None, point: Optional[list] = None,
                  bearing_min_deg: float = -180.0,
                  bearing_max_deg: float = 180.0,
                  theta_deg: Optional[float] = None) -> bool:
    # YAML often uses `theta_deg: N` → ±N symmetric window
    if theta_deg is not None:
        bearing_min_deg = -float(theta_deg)
        bearing_max_deg =  float(theta_deg)
    if obj is not None:
        target = scene_ctx.get_object_centre(obj)
    elif point is not None:
        target = np.asarray(point, dtype=float)
    else:
        return False
    if target is None:
        return False
    f = _forward(cam_pos, cam_target)
    d = target - np.asarray(cam_pos)
    yaw = _signed_yaw_xy(f, d)
    return bearing_min_deg <= yaw <= bearing_max_deg


def InRoom(cam_pos, cam_target, hfov_deg, scene_ctx,
           *, room_id: Optional[str] = None,
           room: Optional[str] = None) -> bool:
    rid = room_id if room_id is not None else room
    return scene_ctx.is_position_in_room_id(np.asarray(cam_pos), rid)


def NotBlockedBy(cam_pos, cam_target, hfov_deg, scene_ctx,
                 *, obj: str, blocker: Optional[str] = None) -> bool:
    occ = scene_ctx.occlusion_fraction(obj, cam_pos, cam_target)
    if blocker is None:
        return occ < 0.5
    primary = scene_ctx.primary_occluder(obj, cam_pos, cam_target)
    return primary != blocker


def OrthogonalToPlane(cam_pos, cam_target, hfov_deg, scene_ctx,
                      *, plane: str, max_offset_deg: float = 15.0,
                      theta_max_deg: Optional[float] = None) -> bool:
    if theta_max_deg is not None:
        max_offset_deg = float(theta_max_deg)
    n = scene_ctx.get_plane_normal(plane)
    if n is None:
        # Permissive fallback: unspecified plane → don't fail the slot just
        # because the instantiator did not bind a concrete plane id.
        return True
    f = _forward(cam_pos, cam_target)
    angle = _angle_between(f, n)
    angle = min(angle, 180.0 - angle)
    # Camera forward should be ALONG the normal (i.e., perpendicular to plane)
    # → small angle between forward and normal.
    return angle <= max_offset_deg


def PairVisible(cam_pos, cam_target, hfov_deg, scene_ctx,
                *, obj_a: str, obj_b: str,
                min_corners: int = 3, max_occ: float = 0.40) -> bool:
    return (
        Visible(cam_pos, cam_target, hfov_deg, scene_ctx,
                obj=obj_a, min_corners=min_corners, max_occ=max_occ)
        and
        Visible(cam_pos, cam_target, hfov_deg, scene_ctx,
                obj=obj_b, min_corners=min_corners, max_occ=max_occ)
    )


def SeparationOnImage(cam_pos, cam_target, hfov_deg, scene_ctx,
                      *, obj_a: str, obj_b: str,
                      min_separation_px: float = 40.0,
                      s_min_frac: Optional[float] = None) -> bool:
    if s_min_frac is not None:
        min_separation_px = float(s_min_frac) * float(scene_ctx._image_width())
    if obj_a == obj_b:
        # Same proxy id on both sides → caller is using one object as a stand-in
        # for both sides of a portal. Skip rather than fail the slot.
        return True
    uva, fa = scene_ctx.project_aabb_corners(obj_a, cam_pos, cam_target)
    uvb, fb = scene_ctx.project_aabb_corners(obj_b, cam_pos, cam_target)
    if fa.sum() == 0 or fb.sum() == 0:
        return False
    ca = uva[fa].mean(axis=0)
    cb = uvb[fb].mean(axis=0)
    return float(np.linalg.norm(ca - cb)) >= min_separation_px


def SideView(cam_pos, cam_target, hfov_deg, scene_ctx,
             *, obj: str, side: str = "front",
             min_angle_deg: float = 30.0, max_angle_deg: float = 80.0) -> bool:
    centre = scene_ctx.get_object_centre(obj)
    n = scene_ctx._object_face_normal(obj, side)
    if centre is None or n is None:
        return False
    d = np.asarray(cam_pos) - centre
    if np.linalg.norm(d) < 1e-6:
        return False
    angle = _angle_between(d, n)
    return min_angle_deg <= angle <= max_angle_deg


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PREDICATE_REGISTRY = {
    "Visible":           Visible,
    "DistanceBand":      DistanceBand,
    "AspectExposed":     AspectExposed,
    "Centered":          Centered,
    "ScaleBand":         ScaleBand,
    "BearingWithin":     BearingWithin,
    "InRoom":            InRoom,
    "NotBlockedBy":      NotBlockedBy,
    "OrthogonalToPlane": OrthogonalToPlane,
    "PairVisible":       PairVisible,
    "SeparationOnImage": SeparationOnImage,
    "SideView":          SideView,
}


def evaluate_predicate(name: str, cam_pos, cam_target, hfov_deg,
                       scene_ctx, **kwargs) -> bool:
    fn = PREDICATE_REGISTRY.get(name)
    if fn is None:
        raise KeyError(f"Unknown predicate: {name}")
    return bool(fn(cam_pos, cam_target, hfov_deg, scene_ctx, **kwargs))
