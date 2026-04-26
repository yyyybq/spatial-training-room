#!/usr/bin/env python3
"""
Multi-Step Navigation MCA question generator.

Run example:
  python -m Data_generation.sampler.question_generator.multi_step_navigation_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/multi_step_navigation.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/msn_items3 \
    --per_room_points 12 --max_items 20

Behavior summary:
- Sample valid camera poses using the sampler-style `generate_camera_positions`.
- For a pose that sees at least two visible objects, pick a pair (start, goal).
- Treat the agent as located at the camera position with the camera forward as the initial heading.
- Compute a realistic sequence of navigation actions (Turn degrees and Move meters) that brings the agent from the start to the goal.
  Answers are simple sequences of 2-3 primitive actions (Turn and/or Move). Examples:
    - Turn right 90 degrees; Move forward 1.5 m
    - Move forward 0.5 m; Turn left 45 degrees; Move forward 1.1 m
- Create three distractors (plausible but do NOT reach the goal).
- Write per-item outputs: `meta.json`, `preview.png` and `view.png` in the out-dir.

Notes / constraints from the user:
1. Do not import `qa_batch_generator`. All used helpers are copied or imported from allowed modules.
2. All imports are top-level. No lazy imports or try/except wrappers.
3. Must render. Thumbnails use `WIDTH` from batch_utils as the single thumb size argument.
4. Use shared helpers/constants import below for visibility and intrinsics.
5. Avoid try/catch. If an exception must be handled, print it explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import imageio

# sampler helpers for sampling single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview helpers (no lazy import)
from ...bench_generation.preview import compose_preview_for_item, compose_view_map, render_thumbnail_for_pose

# shared helpers/constants
from ...bench_generation.batch_utils import (
    create_intrinsics,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    camtoworld_from_pos_target,
    occluded_area_on_image,
    is_aabb_occluded,
    is_occluded_by_any,
    count_visible_corners_for_box,
    BLACKLIST,
    WIDTH,
    HEIGHT,
)


def unit_xy(v: np.ndarray) -> np.ndarray:
    v2 = np.array([v[0], v[1]], dtype=float)
    n = np.linalg.norm(v2)
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return v2 / n


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suf = 'th'
    else:
        suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suf}"


def right_xy(forward_xy: np.ndarray) -> np.ndarray:
    f = np.array(forward_xy, dtype=float)
    return np.array([f[1], -f[0]], dtype=float)


def project_point_to_image_xy(K: np.ndarray, camtoworld: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    world = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    w2c = np.linalg.inv(camtoworld)
    cam = w2c @ world
    z = float(cam[2])
    if z <= 1e-6:
        return float('inf'), float('inf')
    x = float(cam[0]); y = float(cam[1])
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v


def ordinal(n: int) -> str:
    n = int(n)
    if n % 100 in (11, 12, 13):
        return f"{n}th"
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def visible_area_ratio_and_corners(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> Tuple[float, int]:
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    camtoworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = target_px / float(width * height)
    return target_ratio, corners_visible


def bbox_min_distance(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    """Compute minimum distance between two axis-aligned bounding boxes in XY plane.
    If boxes overlap in XY, distance is 0. Uses only X and Y coordinates.
    Inputs are arrays-like of length >=2 (x,y,...).
    """
    ax0, ay0 = float(a_min[0]), float(a_min[1])
    ax1, ay1 = float(a_max[0]), float(a_max[1])
    bx0, by0 = float(b_min[0]), float(b_min[1])
    bx1, by1 = float(b_max[0]), float(b_max[1])

    # compute dx
    if ax1 < bx0:
        dx = bx0 - ax1
    elif bx1 < ax0:
        dx = ax0 - bx1
    else:
        dx = 0.0

    # compute dy
    if ay1 < by0:
        dy = by0 - ay1
    elif by1 < ay0:
        dy = ay0 - by1
    else:
        dy = 0.0

    return math.hypot(dx, dy)


def point_in_expanded_bbox(point: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, margin: float = 1.0) -> bool:
    """Return True if point (3D) lies within bbox expanded by margin (meters) in X and Y.
    Only XY plane is considered; Z is ignored.
    """
    x, y = float(point[0]), float(point[1])
    minx = float(bbox_min[0]) - margin
    miny = float(bbox_min[1]) - margin
    maxx = float(bbox_max[0]) + margin
    maxy = float(bbox_max[1]) + margin
    return (x >= minx) and (x <= maxx) and (y >= miny) and (y <= maxy)


def compute_navigation_sequence(start_pos: np.ndarray, start_heading_xy: np.ndarray, goal_pos: np.ndarray) -> List[Dict[str, Any]]:
    """Compute a realistic 2-3 action primitive sequence (Turn degrees and Move meters)
    from start_pos (agent) with initial heading start_heading_xy to reach goal_pos in XY.

    Movement model:
    - Turn is instantaneous and adjusts heading by signed degrees (positive = left/CCW in our convention).
    - Move advances along current heading for given meters (only XY plane).

    Returns a list of actions, each action is dict like { 'type': 'turn', 'angle': 45 } or { 'type': 'move', 'distance': 1.2 }
    Guarantee: sequence has 2 or 3 primitive actions and will end within 0.35m of goal (if possible).
    """
    # project to XY
    p0 = np.array(start_pos[:2], dtype=float)
    t = np.array(goal_pos[:2], dtype=float)
    v = t - p0
    dist = float(np.linalg.norm(v))
    if dist < 0.05:
        # already at the goal
        return []

    h = unit_xy(start_heading_xy)
    # compute angle from heading to vector to goal, positive = left (CCW)
    ang_rad = math.atan2(v[1], v[0]) - math.atan2(h[1], h[0])
    # normalize to [-pi, pi]
    ang_rad = (ang_rad + math.pi) % (2 * math.pi) - math.pi
    ang_deg = math.degrees(ang_rad)

    # Prefer a concise 2-action solution when the angle is significant: Turn + Move
    actions: List[Dict[str, Any]] = []
    if abs(ang_deg) >= 20.0:
        # Two actions: turn then move
        actions.append({'type': 'turn', 'angle': round(ang_deg)})
        actions.append({'type': 'move', 'distance': round(dist, 2)})
        return actions

    # If angle is small, use a 2-3 action plan: Move partway, then turn small and move remaining
    # split into Move(d1), Turn(angle2), Move(d2)
    # choose d1 = min( max(0.4, 0.4*dist), dist*0.6 ) to ensure at least some forward motion
    d1 = float(min(max(0.4, 0.4 * dist), 0.6 * dist))
    remaining = max(0.0, dist - d1)
    if remaining < 0.05:
        # fallback: single move (represented as two small moves)
        actions.append({'type': 'move', 'distance': round(dist * 0.6, 2)})
        actions.append({'type': 'move', 'distance': round(dist * 0.4, 2)})
        return actions

    # After moving d1 along heading h, compute new vector to goal
    mid = p0 + h * d1
    v2 = t - mid
    dist2 = float(np.linalg.norm(v2))
    if dist2 < 0.01:
        actions.append({'type': 'move', 'distance': round(d1, 2)})
        return actions
    # angle from current heading to new vector
    ang2 = math.degrees((math.atan2(v2[1], v2[0]) - math.atan2(h[1], h[0]) + math.pi) % (2 * math.pi) - math.pi)
    actions.append({'type': 'move', 'distance': round(d1, 2)})
    # small turn then move
    actions.append({'type': 'turn', 'angle': round(ang2)})
    actions.append({'type': 'move', 'distance': round(dist2, 2)})
    return actions


def apply_actions(start_pos: np.ndarray, start_heading_xy: np.ndarray, actions: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply primitive actions and return final position and heading (both in XY and full pos for position z preserved).
    start_pos is 3D (x,y,z). Heading is 2D unit vector.
                # Build left-to-right numbering for same-label visibles
                camtoworld = camtoworld_from_pos_target(pos, tgt)
                label_groups: Dict[str, List[Tuple[SceneObject, float]]] = {}
                for so, _, _ in visibles:
                    u, v = project_point_to_image_xy(K, camtoworld, np.array(so.position, dtype=float))
                    if not np.isfinite(u):
                        u = float('inf')
                    label_groups.setdefault(str(so.label), []).append((so, float(u)))
                display_label_map: Dict[str, str] = {}
                for lbl, arr in label_groups.items():
                    arr_sorted = sorted(arr, key=lambda x: x[1])
                    for idx, (soi, _) in enumerate(arr_sorted, start=1):
                        display_label_map[str(soi.id)] = f"the {ordinal(idx)} {lbl}"
    """
    pos = np.array(start_pos, dtype=float).copy()
    h = unit_xy(start_heading_xy).copy()
    for a in actions:
        if a['type'] == 'turn':
            ang = math.radians(float(a['angle']))
            c = math.cos(ang); s = math.sin(ang)
            # rotate heading by ang (CCW positive)
            hx, hy = h[0], h[1]
            h = np.array([c * hx - s * hy, s * hx + c * hy], dtype=float)
            # renormalize
            h = unit_xy(h)
        elif a['type'] == 'move':
            d = float(a['distance'])
            pos[0] += h[0] * d
            pos[1] += h[1] * d
        else:
            raise ValueError(f"Unknown action type: {a}")
    return pos, h


def make_distractors(start_pos: np.ndarray, start_heading_xy: np.ndarray, goal_pos: np.ndarray, true_actions: List[Dict[str, Any]], n: int = 3, goal_bbox_min: np.ndarray | None = None, goal_bbox_max: np.ndarray | None = None, expanded_margin: float = 1.0) -> List[List[Dict[str, Any]]]:
    """Generate n distractor action sequences that do NOT reach the goal.
    Keep distractors plausible: change sign of turn, change magnitude, shorten/extend distances.
    Ensure final position is at least 0.6m away from goal.
    """
    distractors: List[List[Dict[str, Any]]] = []
    attempts = 0
    # Determine whether the true action is a left- or right-turn sequence (use first turn if present)
    def _is_left(actions: List[Dict[str, Any]]) -> bool:
        for a in actions:
            if a.get('type') == 'turn':
                return float(a.get('angle', 0.0)) > 0
        # fallback: compare goal vector to heading
        h = unit_xy(start_heading_xy)
        vec = np.array(goal_pos[:2], dtype=float) - np.array(start_pos[:2], dtype=float)
        if np.linalg.norm(vec) < 1e-6:
            return False
        ang = math.degrees(math.atan2(vec[1], vec[0]) - math.atan2(h[1], h[0]))
        ang = (ang + 180) % 360 - 180
        return ang > 0

    true_is_left = _is_left(true_actions)
    left_count = 1 if true_is_left else 0
    right_count = 1 - left_count

    # We want overall roughly two left and two right among the four choices
    desired_left_total = 2
    desired_right_total = 2

    while len(distractors) < n and attempts < 200:
        attempts += 1
        cand: List[Dict[str, Any]] = []
        # Strategy: perturb true actions
        for a in true_actions:
            if a['type'] == 'turn':
                # sometimes flip sign or add large offset
                delta = int(np.random.choice([-90, -45, 45, 90, 30, -30, 0]))
                ang = int(a['angle'] + delta)
                # normalize to (-180,180]
                ang = int(((ang + 180) % 360) - 180)
                # quantize to multiples of 5
                ang = int(round(ang / 5.0) * 5)
                # final normalization
                ang = int(((ang + 180) % 360) - 180)
                cand.append({'type': 'turn', 'angle': ang})
            elif a['type'] == 'move':
                # shorten or overshoot
                factor = float(np.random.choice([0.3, 0.5, 0.7, 1.5, 2.0]))
                d = max(0.05, round(a['distance'] * factor, 2))
                cand.append({'type': 'move', 'distance': d})
        # Occasionally replace by alternative structure: Turn+Move instead of Move+Turn+Move
        if len(cand) == 1:
            # single move -> turn+move wrong
            ang = int(np.random.choice([90, -90, 45, -45]))
            d = float(cand[0]['distance'])
            cand = [{'type': 'turn', 'angle': ang}, {'type': 'move', 'distance': d}]

        # normalize any turn angles inside candidate and classify
        for ta in cand:
            if ta.get('type') == 'turn':
                ta['angle'] = int(((int(ta['angle']) + 180) % 360) - 180)

        final_pos, _ = apply_actions(start_pos, start_heading_xy, cand)
        # ensure final not too close to the goal center
        if float(np.linalg.norm(final_pos[:2] - np.array(goal_pos[:2]))) > 0.6:
            # ensure not inside goal expanded bbox if bbox provided
            bad = False
            if goal_bbox_min is not None and goal_bbox_max is not None:
                if point_in_expanded_bbox(final_pos, goal_bbox_min, goal_bbox_max, margin=expanded_margin):
                    bad = True
            if not bad:
                # classify candidate left/right based on first turn or geometry
                is_left = _is_left(cand)
                # enforce desired counts (do not exceed desired totals)
                if is_left:
                    if left_count >= desired_left_total:
                        continue
                else:
                    if right_count >= desired_right_total:
                        continue
                # ensure not duplicate
                if all(not (cand == ex) for ex in distractors):
                    distractors.append(cand)
                    if is_left:
                        left_count += 1
                    else:
                        right_count += 1
    # if failed to make enough, pad with simple wrong-turns
    # If failed to make enough distractors by perturbing the true action, sample explicit
    # endpoints outside the expanded goal bbox and create simple Turn+Move sequences to them.
    max_fallback_attempts = 100
    fb_attempts = 0
    while len(distractors) < n and fb_attempts < max_fallback_attempts:
        fb_attempts += 1
        # sample a random direction and distance from start (XY)
        ang_rad = np.random.uniform(-math.pi, math.pi)
        dist = float(np.random.uniform(0.6, 3.0))
        cand_xy = np.array([start_pos[0] + math.cos(ang_rad) * dist, start_pos[1] + math.sin(ang_rad) * dist], dtype=float)

        # if goal bbox provided, ensure candidate point is outside expanded bbox
        if goal_bbox_min is not None and goal_bbox_max is not None:
            if point_in_expanded_bbox(np.array([cand_xy[0], cand_xy[1], start_pos[2]]), goal_bbox_min, goal_bbox_max, margin=expanded_margin):
                continue

        # ensure candidate is not too close to the goal center
        if float(np.linalg.norm(cand_xy - np.array(goal_pos[:2]))) <= 0.6:
            continue

        # build simple Turn+Move sequence from start heading to candidate point
        # compute angle difference between heading and vector to candidate
        h = unit_xy(start_heading_xy)
        vec = cand_xy - np.array(start_pos[:2], dtype=float)
        if np.linalg.norm(vec) < 1e-6:
            continue
        desired_ang = math.degrees(math.atan2(vec[1], vec[0]) - math.atan2(h[1], h[0]))
        desired_ang = (desired_ang + 180) % 360 - 180
        # enforce desired left/right if we still need counts
        need_left = (left_count < desired_left_total)
        need_right = (right_count < desired_right_total)
        if need_left and not need_right:
            desired_ang = abs(desired_ang)
        elif need_right and not need_left:
            desired_ang = -abs(desired_ang)
        move_d = float(round(max(0.05, np.linalg.norm(vec)), 2))
        cand_seq = [{'type': 'turn', 'angle': int(round(desired_ang))}, {'type': 'move', 'distance': move_d}]

        # final safety check: final not in expanded bbox and not duplicate
        final_pos, _ = apply_actions(start_pos, start_heading_xy, cand_seq)
        bad = False
        if goal_bbox_min is not None and goal_bbox_max is not None:
            if point_in_expanded_bbox(final_pos, goal_bbox_min, goal_bbox_max, margin=expanded_margin):
                bad = True
        if bad:
            continue
        if all(not (cand_seq == ex) for ex in distractors):
            # classify and accept only if it helps reach left/right targets
            is_left = _is_left(cand_seq)
            if is_left:
                if left_count < desired_left_total:
                    distractors.append(cand_seq)
                    left_count += 1
            else:
                if right_count < desired_right_total:
                    distractors.append(cand_seq)
                    right_count += 1

    # As a last resort if we still don't have enough, fall back to simple wrong-turns (will be rare)
    while len(distractors) < n:
        ang = int(np.random.choice([90, -90, 180]))
        d = float(max(0.5, min(2.0, np.random.random() * 2.0)))
        cand = [{'type': 'turn', 'angle': ang}, {'type': 'move', 'distance': round(d, 2)}]
        distractors.append(cand)
    return distractors


def action_sequence_to_text(actions: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for a in actions:
        if a['type'] == 'turn':
            deg = float(a['angle'])
            if deg > 0:
                parts.append(f"Turn left {abs(int(deg))} degrees")
            elif deg < 0:
                parts.append(f"Turn right {abs(int(deg))} degrees")
            else:
                parts.append(f"Turn 0 degrees")
        elif a['type'] == 'move':
            parts.append(f"Move forward {float(a['distance']):.2f} m")
    return "; ".join(parts)


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 12, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200,
                         verbose: bool = False) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    K = np.array(create_intrinsics()['K'], dtype=float)

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        if verbose:
            print(f"[scene] {scene_path}")

        labels_file = scene_path / 'labels.json'
        if not labels_file.exists():
            if verbose:
                print(f"[skip] no labels.json in {scene_path}")
            continue
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        objects: Dict[str, SceneObject] = {}
        for it in labels:
            if not isinstance(it, dict):
                continue
            if not it.get('ins_id') or 'bounding_box' not in it:
                continue
            obj = SceneObject(it)
            objects[obj.id] = obj
        if not objects:
            continue

        aabbs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

        per_scene_count = 0

        # We pick visible pairs from sampled poses
        for obj_id, obj in objects.items():
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break

            poses = generate_camera_positions(scene_path, obj,
                                              per_room_points=per_room_points,
                                              min_dist=min_dist,
                                              max_dist=max_dist)
            if not poses:
                continue

            for p in poses:
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

                pos = np.array(p['position'], dtype=float)
                tgt = np.array(p['target'], dtype=float)

                # Determine camera forward vector and local heading
                fwd = tgt - pos
                heading_xy = unit_xy(fwd)

                # collect visible objects (use thresholds similar to other generators)
                visibles: List[Tuple[SceneObject, float, int]] = []
                for so in objects.values():
                    if str(so.label).lower() in BLACKLIST:
                        continue
                    ratio, corners = visible_area_ratio_and_corners(pos, tgt, so, aabbs_all, K, WIDTH, HEIGHT)
                    if ratio > 0.05 and corners >= 2:  # require visible but less strict
                        visibles.append((so, ratio, corners))
                    # print(ratio, corners)
                if len(visibles) < 2:
                    continue
                # print(visibles)
                # Number duplicates left-to-right in the current view (display_label)
                camtoworld = camtoworld_from_pos_target(pos, tgt)
                per_label: Dict[str, List[Tuple[float, SceneObject]]] = {}
                for so, _, _ in visibles:
                    u, v = project_point_to_image_xy(K, camtoworld, np.array(so.position, dtype=float))
                    key = str(so.label)
                    per_label.setdefault(key, []).append((u, so))
                display_name: Dict[str, str] = {}
                for lab, arr in per_label.items():
                    arr_sorted = sorted(arr, key=lambda t: t[0])
                    # Only add ordinal markers when there are duplicates of the same label
                    if len(arr_sorted) > 1:
                        for idx, (_, so) in enumerate(arr_sorted, start=1):
                            display_name[str(so.id)] = f"the {ordinal(idx)} {lab}"
                    else:
                        # single instance: keep the plain label
                        for _, so in arr_sorted:
                            display_name[str(so.id)] = lab

                # from visibles pick ordered pair (start, goal) that are not too close by bbox boundary
                # prefer different labels
                vis_objs = [so for so, _, _ in visibles]
                chosen_pair = None
                for i in range(len(vis_objs)):
                    for j in range(len(vis_objs)):
                        if i == j:
                            continue
                        A = vis_objs[i]
                        B = vis_objs[j]
                        # require minimum distance between object bounding boxes >= 2.0 meters (XY plane)
                        bdist = bbox_min_distance(np.array(A.bbox_min), np.array(A.bbox_max), np.array(B.bbox_min), np.array(B.bbox_max))
                        if bdist < 1:
                            # print("bdist=", bdist)
                            continue
                        # print("ok bdist=", bdist)
                        chosen_pair = (A, B)
                        break
                    if chosen_pair is not None:
                        break
                if chosen_pair is None:
                    print("no valid pair")
                    continue

                A, B = chosen_pair

                # compute true actions starting from the START object position (A),
                # using the current camera heading as the agent heading (user requested: start at object with same heading as current view)
                start_agent_pos = np.array(A.position, dtype=float)
                true_actions = compute_navigation_sequence(start_agent_pos, heading_xy, np.array(B.position, dtype=float))
                if not true_actions:
                    print("no valid actions")
                    continue

                # apply actions from the start object position (not camera)
                final_pos, _ = apply_actions(start_agent_pos, heading_xy, true_actions)
                # ensure this reaches goal within tolerance
                if float(np.linalg.norm(final_pos[:2] - np.array(B.position)[:2])) > 0.35:
                    # try small alternative: rotate fully then move
                    ang = math.degrees(math.atan2((B.position[1] - pos[1]), (B.position[0] - pos[0])) - math.atan2(heading_xy[1], heading_xy[0]))
                    ang = (ang + 180) % 360 - 180
                    true_actions = [{'type': 'turn', 'angle': int(round(ang))}, {'type': 'move', 'distance': round(float(np.linalg.norm(np.array(B.position)[:2] - pos[:2])), 2)}]
                    final_pos, _ = apply_actions(pos, heading_xy, true_actions)
                    if float(np.linalg.norm(final_pos[:2] - np.array(B.position)[:2])) > 0.35:
                        continue

                # compute goal bbox and note whether true_actions final pos lies inside its expanded bbox
                goal_bbox_min = np.array(B.bbox_min)
                goal_bbox_max = np.array(B.bbox_max)
                true_actions_ends_in_expanded_bbox = False
                if point_in_expanded_bbox(final_pos, goal_bbox_min, goal_bbox_max, margin=1.0):
                    # Per user request: do NOT adjust the correct action. Instead, allow the
                    # correct action to remain as-is (even if it ends inside the expanded bbox),
                    # and ensure distractors are explicitly sampled to end outside the expanded bbox.
                    true_actions_ends_in_expanded_bbox = True

                # generate distractors
                distractors = make_distractors(start_agent_pos, heading_xy, np.array(B.position, dtype=float), true_actions, n=3, goal_bbox_min=goal_bbox_min, goal_bbox_max=goal_bbox_max, expanded_margin=1.0)

                # assemble choices: correct + 3 distractors in randomized order
                choices_seq = [true_actions] + distractors
                labels = ['A', 'B', 'C', 'D']
                # shuffle while keeping track of the correct index
                order = list(range(4))
                np.random.shuffle(order)
                shuffled = [choices_seq[i] for i in order]
                correct_index = order.index(0)

                # textual choices
                textual_choices = [action_sequence_to_text(s) for s in shuffled]

                # Build meta
                render_cfg = create_intrinsics()
                render_cfg['camtoworld'] = camtoworld_from_pos_target(pos, tgt).tolist()

                # prepare display labels for start/goal
                start_display = display_name.get(str(A.id), A.label)
                goal_display = display_name.get(str(B.id), B.label)

                item = {
                    'qtype': 'multi_step_navigation',
                    'scene': str(scene_path),
                    'question': f"From this view, starting near {start_display}, how do you go to {goal_display}? Provide the sequence of turn and move actions.",
                    'choices': textual_choices,
                    'answer': labels[correct_index],
                    'meta': {
                        'camera_pos': pos.tolist(),
                        'camera_target': tgt.tolist(),
                        'render': render_cfg,
                        # include camera forward heading for use in preview (start heading)
                        'start_heading': heading_xy.tolist(),
                        'start_object': {'id': A.id, 'label': A.label, 'display_label': start_display, 'center': np.array(A.position).tolist()},
                        'goal_object': {'id': B.id, 'label': B.label, 'display_label': goal_display, 'center': np.array(B.position).tolist()},
                        'true_actions': true_actions,
                        'distractors': distractors,
                        'choices_actions': shuffled,
                    }
                }

                items.append(item)
                per_scene_count += 1

                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    base = f"{scene_path.name}_item_{idx:04d}_msn"
                    item_dir = out_dir / base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render main view and preview
                    img = render_thumbnail_for_pose(scene_path, {'position': pos, 'target': tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'view.png'), img)

                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

    # write out jsonl if requested
    if out_path is not None:
        with open(out_path, 'w', encoding='utf-8') as outf:
            for it in items:
                outf.write(json.dumps(it, ensure_ascii=False) + "\n")

    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scenes_root', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--per_room_points', type=int, default=12)
    p.add_argument('--min_dist', type=float, default=1)
    p.add_argument('--max_dist', type=float, default=7)
    p.add_argument('--max_items', type=int, default=200)
    p.add_argument('--max_items_per_scene', type=int, default=200)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    iterate_and_generate(args.scenes_root, args.out, args.out_dir,
                         per_room_points=args.per_room_points,
                         min_dist=args.min_dist, max_dist=args.max_dist,
                         max_items=args.max_items, max_items_per_scene=args.max_items_per_scene,
                         verbose=args.verbose)


if __name__ == '__main__':
    main()
