"""
SceneContext: loads and caches all scene data needed by task generators.

Wraps the existing utils (occlusion, batch_utils) to provide a single
interface for querying scene geometry, room polygons, and object AABBs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reuse existing scene-loading utilities
from ..utils.occlusion import (
    AABB,
    camtoworld_from_pos_target,
    is_occluded_by_any,
    is_box_occluded_by_any,
    is_target_in_fov,
    occluded_area_on_image,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    world_to_camera,
    aabb_corners,
    project_point,
    point_in_image,
)
from ..bench_generation.batch_utils import (
    load_room_polys,
    load_structure_height_bounds,
    point_in_poly,
    create_intrinsics,
    count_visible_corners_for_box,
    is_aabb_occluded,
    BLACKLIST,
)

# Camera defaults (matching existing generators)
DEFAULT_WIDTH = 400
DEFAULT_HEIGHT = 400
DEFAULT_CAMERA_HEIGHT = 0.8     # metres above floor


@dataclass
class SceneContext:
    """
    Loaded representation of a 3D scene.

    Usage::

        ctx = SceneContext.load("/path/to/scene/0013_840910")
        for obj in ctx.objects:
            print(obj.label, obj.center)
    """

    scene_path: Path
    scene_name: str

    # Geometry
    objects: List[AABB] = field(default_factory=list)       # furniture / objects (no walls)
    wall_aabbs: List[AABB] = field(default_factory=list)    # wall geometry
    room_polygons: List[np.ndarray] = field(default_factory=list)   # XY room footprints

    # Vertical bounds
    floor_z: float = 0.0
    ceiling_z: float = 3.0

    # Intrinsics (default)
    intrinsics: Dict[str, Any] = field(default_factory=lambda: create_intrinsics())

    # Labels raw (for metadata)
    _labels_raw: List[Dict] = field(default_factory=list, repr=False)

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    @classmethod
    def load(cls, scene_path: str | Path) -> "SceneContext":
        """Load and return a fully initialised SceneContext."""
        p = Path(scene_path)
        ctx = cls(scene_path=p, scene_name=p.name)
        ctx._load_all()
        return ctx

    def _load_all(self) -> None:
        """Load all scene data from disk."""
        sp = str(self.scene_path)

        # Objects
        self.objects = load_scene_aabbs(sp)
        # Back-compat alias: legacy code (and a couple of tests) read
        # ``scene_ctx.aabbs``.  We keep both names pointing at the same list
        # so future renames can deprecate ``aabbs`` without breaking callers.
        self.aabbs = self.objects
        self.wall_aabbs = load_scene_wall_aabbs(sp)
        self.room_polygons = load_room_polys(sp)

        # Height bounds
        min_h, max_h = load_structure_height_bounds(sp)
        if min_h is not None:
            self.floor_z = float(min_h)
        if max_h is not None:
            self.ceiling_z = float(max_h)

        # Raw labels
        lf = self.scene_path / "labels.json"
        if lf.exists():
            with open(lf, "r", encoding="utf-8") as f:
                self._labels_raw = json.load(f)

    # -----------------------------------------------------------------------
    # Convenience queries
    # -----------------------------------------------------------------------

    @property
    def all_blockers(self) -> List[AABB]:
        """All geometry that can occlude (objects + walls)."""
        return self.objects + self.wall_aabbs

    def get_object_by_id(self, ins_id: str) -> Optional[AABB]:
        for obj in self.objects:
            if obj.id == ins_id:
                return obj
        # Allow portal ids ("portal_N") to be used as targets for predicates
        # like Visible/DistanceBand by returning a synthetic thin-slab AABB
        # straddling the portal opening.
        if isinstance(ins_id, str) and ins_id.startswith("portal_"):
            for p in self._portals():
                if p["id"] != ins_id:
                    continue
                pos = np.asarray(p["position"], dtype=float)
                w = 0.5 * float(p["width"])
                h = 0.5 * float(p["height"])
                t = 0.5 * float(p.get("thickness", 0.12))
                # Build axis-aligned box: half-width along the in-plane XY axis,
                # half-thickness along the normal axis, half-height along Z.
                normal = np.asarray(p["normal"], dtype=float)
                bmin = pos.copy(); bmax = pos.copy()
                if abs(normal[0]) > abs(normal[1]):
                    bmin[0] -= t; bmax[0] += t
                    bmin[1] -= w; bmax[1] += w
                else:
                    bmin[1] -= t; bmax[1] += t
                    bmin[0] -= w; bmax[0] += w
                bmin[2] = float(pos[2]) - h
                bmax[2] = float(pos[2]) + h
                return AABB(id=ins_id, label=str(p.get("type", "PORTAL")).lower(),
                            bmin=bmin, bmax=bmax)
        return None

    def get_objects_by_label(self, label: str) -> List[AABB]:
        label_lower = label.lower()
        return [o for o in self.objects if o.label.lower() == label_lower]

    @property
    def valid_objects(self) -> List[AABB]:
        """Objects that are not on the blacklist."""
        return [o for o in self.objects if o.label.lower() not in BLACKLIST]

    def is_position_in_room(self, pos: np.ndarray) -> bool:
        """Return True if (x, y) is inside any room polygon."""
        x, y = float(pos[0]), float(pos[1])
        for poly in self.room_polygons:
            if point_in_poly(x, y, poly):
                return True
        return False

    def is_position_valid(
        self,
        pos: np.ndarray,
        min_wall_dist: float = 0.15,
        min_object_dist: float = 0.2,
        camera_height: Optional[float] = None,
    ) -> bool:
        """
        Return True if `pos` is a legal camera placement:
        - Inside a room polygon
        - Not inside any object AABB
        - Not closer than min_object_dist to any object centre
        - Height within bounds
        """
        if not self.is_position_in_room(pos):
            return False

        ch = camera_height if camera_height is not None else DEFAULT_CAMERA_HEIGHT
        # NOTE: floor_z / ceiling_z come from `occupancy.json`, which in many
        # exporters is a 2D occupancy *slab* (often z in [0.1, 1.0]) rather
        # than the room ceiling.  We therefore only enforce the z bound when
        # the slab is plausibly a real room volume (>= 2 m tall).
        if (self.ceiling_z - self.floor_z) >= 2.0:
            if not (self.floor_z <= pos[2] <= self.ceiling_z):
                return False

        for obj in self.objects:
            center = 0.5 * (obj.bmin + obj.bmax)
            if np.linalg.norm(pos - center) < min_object_dist:
                return False
            # inside AABB check
            if np.all(pos >= obj.bmin) and np.all(pos <= obj.bmax):
                return False

        return True

    # -----------------------------------------------------------------------
    # Visibility helpers
    # -----------------------------------------------------------------------

    def is_object_visible(
        self,
        obj: AABB,
        cam_pos: np.ndarray,
        cam_target: np.ndarray,
        min_visible_corners: int = 2,
        max_occ_ratio: float = 0.85,
    ) -> bool:
        """
        Return True if `obj` is sufficiently visible from the given camera pose.
        """
        c2w = camtoworld_from_pos_target(cam_pos, cam_target)
        intrinsics = self.intrinsics

        # Check that at least N corners project inside the image
        corners = aabb_corners(obj)
        visible_count = count_visible_corners_for_box(
            corners, c2w, intrinsics, self.all_blockers, obj
        )
        if visible_count < min_visible_corners:
            return False

        # Check occlusion ratio
        occ_ratio, _ = occluded_area_on_image(
            obj, c2w, intrinsics,
            [b for b in self.all_blockers if b.id != obj.id],
        )
        return occ_ratio <= max_occ_ratio

    def get_visible_objects(
        self,
        cam_pos: np.ndarray,
        cam_target: np.ndarray,
        min_visible_corners: int = 2,
        max_occ_ratio: float = 0.85,
    ) -> List[AABB]:
        """Return all valid objects that are visible from the given view."""
        return [
            obj for obj in self.valid_objects
            if self.is_object_visible(obj, cam_pos, cam_target,
                                       min_visible_corners, max_occ_ratio)
        ]

    # -----------------------------------------------------------------------
    # Geometry helpers
    # -----------------------------------------------------------------------

    def distance_to_object(self, cam_pos: np.ndarray, obj: AABB) -> float:
        """Euclidean distance from camera to object centre (XY plane only)."""
        center = 0.5 * (obj.bmin + obj.bmax)
        delta = cam_pos[:2] - center[:2]
        return float(np.linalg.norm(delta))

    def bearing_to_object(self, cam_pos: np.ndarray, cam_target: np.ndarray, obj: AABB) -> float:
        """
        Signed bearing angle (degrees) from camera forward to object centre.
        Positive = object is to the right.
        """
        center = 0.5 * (obj.bmin + obj.bmax)
        forward = cam_target - cam_pos
        to_obj = center - cam_pos
        # Project to XY
        fwd_xy = np.array([forward[0], forward[1]], dtype=float)
        obj_xy = np.array([to_obj[0], to_obj[1]], dtype=float)

        fwd_len = np.linalg.norm(fwd_xy)
        obj_len = np.linalg.norm(obj_xy)
        if fwd_len < 1e-6 or obj_len < 1e-6:
            return 0.0

        fwd_xy /= fwd_len
        obj_xy /= obj_len

        cos_a = np.clip(np.dot(fwd_xy, obj_xy), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_a))

        # Signed: cross product z-component gives sign
        cross_z = fwd_xy[0] * obj_xy[1] - fwd_xy[1] * obj_xy[0]
        if cross_z < 0:
            angle = -angle
        return angle

    def sample_positions_in_room(
        self,
        num_points: int = 20,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        min_wall_dist: float = 0.15,
        min_object_dist: float = 0.2,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[np.ndarray]:
        """
        Uniformly sample valid camera positions across all room polygons.
        Returns list of [x, y, z] positions.
        """
        if rng is None:
            rng = np.random.RandomState()

        positions: List[np.ndarray] = []

        for poly in self.room_polygons:
            xs = poly[:, 0]
            ys = poly[:, 1]
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()

            attempts = 0
            per_poly = max(1, num_points // max(len(self.room_polygons), 1))

            while len(positions) < num_points and attempts < num_points * 50:
                attempts += 1
                x = rng.uniform(xmin, xmax)
                y = rng.uniform(ymin, ymax)
                pos = np.array([x, y, camera_height], dtype=float)

                if self.is_position_valid(pos, min_wall_dist, min_object_dist, camera_height):
                    positions.append(pos)

        return positions[:num_points]

    def sample_positions_around_object(
        self,
        obj: AABB,
        target_distance: float,
        num_points: int = 8,
        camera_height: float = DEFAULT_CAMERA_HEIGHT,
        distance_tolerance: float = 0.25,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[np.ndarray]:
        """
        Sample camera positions at approximately `target_distance` from `obj`.
        Positions are sampled on a ring around the object centre.
        """
        if rng is None:
            rng = np.random.RandomState()

        center = 0.5 * (obj.bmin + obj.bmax)
        positions: List[np.ndarray] = []

        angles = np.linspace(0, 2 * math.pi, num_points + 1)[:-1]
        rng.shuffle(angles)

        for angle in angles:
            # Vary distance slightly
            dist = target_distance + rng.uniform(-distance_tolerance, distance_tolerance)
            dist = max(0.3, dist)
            x = center[0] + dist * math.cos(angle)
            y = center[1] + dist * math.sin(angle)
            pos = np.array([x, y, camera_height], dtype=float)

            if self.is_position_valid(pos):
                positions.append(pos)

        return positions
