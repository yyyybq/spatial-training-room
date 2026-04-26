#!/usr/bin/env python3
"""Find camera poses that see all objects in given pairs and render previews.

This script automatically calculates the optimal camera distance based on the
combined size of the object pair and the camera's Field of View (FOV).

Usage example:
python -m Data_generation.active_data.find_cam_pos \
     --scene /data/liubinglin/jijiatong/ViewSuite/data/InteriorGS/0267_840790 \
     --pairs /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/result/0267_2objects.json \
     --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/result --per-angle 36 --render

python -m Data_generation.active_data.find_cam_pos \
     --scene /data/liubinglin/jijiatong/ViewSuite/data/InteriorGS/0267_840790 \
     --pairs /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/test/test.json \
     --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/active_data/test --per-angle 36 --render
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import imageio

from Data_generation.bench_generation.batch_utils import (
    create_intrinsics,
    camtoworld_from_pos_target,
    occluded_area_on_image,
    count_visible_corners_for_box,
    WIDTH,
    HEIGHT,
    load_room_polys,
    point_in_poly,
)
from Data_generation.bench_generation.batch_utils import is_pos_inside_any_room

from Data_generation.bench_generation.preview import render_thumbnail_for_pose, compose_preview_for_item


def load_labels(scene_dir: Path) -> List[Dict[str, Any]]:
    p = scene_dir / 'labels.json'
    if not p.exists():
        raise FileNotFoundError(f"labels.json not found in {scene_dir}")
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_obj_in_labels(labels: List[Dict[str, Any]], ins_id: str) -> Dict[str, Any] | None:
    for it in labels:
        if not isinstance(it, dict):
            continue
        iid = str(it.get('ins_id') or it.get('id') or '')
        if iid == str(ins_id):
            return it
    return None


def bbox_center_xyz(bbox_points: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    xs = [float(p['x']) for p in bbox_points]
    ys = [float(p['y']) for p in bbox_points]
    zs = [float(p['z']) for p in bbox_points]
    return (float(np.mean(xs)), float(np.mean(ys)), float(np.mean(zs)))


def get_bbox_minmax(bbox_points: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = [float(p['x']) for p in bbox_points]
    ys = [float(p['y']) for p in bbox_points]
    zs = [float(p['z']) for p in bbox_points]
    return np.array([min(xs), min(ys), min(zs)], dtype=float), np.array([max(xs), max(ys), max(zs)], dtype=float)


def point_to_aabb_distance(p: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    """Euclidean distance from point p to AABB [bmin,bmax]. Returns 0.0 if inside."""
    p = np.array(p, dtype=float)
    bmin = np.array(bmin, dtype=float)
    bmax = np.array(bmax, dtype=float)
    closest = np.minimum(np.maximum(p, bmin), bmax)
    return float(np.linalg.norm(p - closest))


def calculate_auto_radii(bbox_list_a: List[Dict], bbox_list_b: List[Dict], K: np.ndarray, img_width: int) -> Tuple[List[float], float]:
    """
    Calculates candidate radii based on the combined size of the objects and camera FOV.
    Formula: distance = object_size_world * (focal_length / image_width)
    """
    # 1. Compute Union Bounding Box
    min_a, max_a = get_bbox_minmax(bbox_list_a)
    min_b, max_b = get_bbox_minmax(bbox_list_b)
    
    union_min = np.minimum(min_a, min_b)
    union_max = np.maximum(max_a, max_b)
    
    # 2. Compute World Diagonal (Conservative Size)
    # Using the diagonal ensures the object fits regardless of rotation
    world_size = float(np.linalg.norm(union_max - union_min))
    
    # 3. Get Focal Length from K
    focal_length = K[0, 0]
    
    # 4. Calculate Ideal Distance
    # geometry: object_screen_size / img_width = object_world_size * focal / (distance * img_width)
    # we want object_screen_size ~= img_width * margin (e.g. 0.8)
    # So: distance = (object_world_size * focal) / (img_width * margin)
    # But simplifying to the user's logic: dist = size * fov_factor
    # where fov_factor = focal / width. 
    fov_factor = focal_length / img_width
    
    # Use a safety margin so objects don't touch the edges (1.1x to 1.2x bigger view)
    safety_margin = 1.2 
    base_dist = world_size * fov_factor * safety_margin
    
    # Ensure a minimum distance to avoid clipping near plane
    base_dist = max(base_dist, 0.5) 
    
    # Generate a range of distances: Ideal, slightly further, much further
    # We don't go closer than ideal because it would likely clip the object
    radii = [base_dist * 0.8, base_dist, base_dist * 1.3, base_dist * 1.6, base_dist * 2.0]
    
    return radii, base_dist


def visible_enough(scene_path: Path, cam_pos: np.ndarray, cam_tgt: np.ndarray, 
                   bbox_min: np.ndarray, bbox_max: np.ndarray, 
                   K: np.ndarray, aabbs_all) -> Tuple[bool, str]:
    """
    Checks visibility and returns (success, failure_reason_code).
    Codes: 'ok', 'low_vis', 'no_corners', 'occluded'
    """
    # 1. Basic Projection & Corners

    corners = count_visible_corners_for_box(
        cam_pos, cam_tgt, K, bbox_min, bbox_max, WIDTH, HEIGHT, corner_threshold=1, aabbs_all=aabbs_all
    )


    # compute visible pixel area ratio via occlusion helper (if available)
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_tgt)
    occ = occluded_area_on_image(
        cam_pos, bbox_min, bbox_max, aabbs_all, K, camtoworld, WIDTH, HEIGHT, target_id=None, depth_mode='min', return_per_occluder=False
    )
    target_px = float(occ.get('target_area_px', 0.0)) if isinstance(occ, dict) else 0.0
    ratio = target_px / float(WIDTH * HEIGHT)


    # debug
    # print(f'ratio: {ratio}, corners: {corners}')

    if ratio <= 0.05:
        return False, 'low_vis'
    if corners < 1:
        return False, 'no_corners'

    # 2. Occlusion Check
    camtoworld = camtoworld_from_pos_target(cam_pos, cam_tgt)

    occ = occluded_area_on_image(
        cam_pos, bbox_min, bbox_max, aabbs_all, K, camtoworld, 
        WIDTH, HEIGHT, target_id=None, depth_mode='min', return_per_occluder=False
    )
    occl_ratio = float(occ.get('occlusion_ratio_target', 0.0)) if isinstance(occ, dict) else 0.0

    print(f'ratio: {ratio}, occl_ratio: {occl_ratio}, corners: {corners}')

    if occl_ratio >= 0.7:
        return False, 'occluded'

    return True, 'ok'


def sample_camera_for_pair(scene_path: Path, objA: Dict[str, Any], objB: Dict[str, Any], 
                           max_tries: int = 500, per_angle: int = 36, preview_dir: Path | None = None, pair_idx: int | None = None) -> Tuple[Dict[str, Any] | None, str]:
    labels = load_labels(scene_path)
    la = find_obj_in_labels(labels, objA['id'])
    lb = find_obj_in_labels(labels, objB['id'])
    if la is None or lb is None:
        return None, "object_id_not_found"

    # Pre-calc bounding geometry
    a_min, a_max = get_bbox_minmax(la['bounding_box'])
    b_min, b_max = get_bbox_minmax(lb['bounding_box'])
    a_center = np.array(bbox_center_xyz(la['bounding_box']), dtype=float)
    b_center = np.array(bbox_center_xyz(lb['bounding_box']), dtype=float)

    mid = (a_center + b_center) / 2.0

    # Load intrinsics and calculating radii
    K = np.array(create_intrinsics()['K'], dtype=float)
    
    # --- NEW: Auto-calculate radii ---
    radii, ideal_dist = calculate_auto_radii(la['bounding_box'], lb['bounding_box'], K, WIDTH)
    
    heights = [1.0, 1.2, 1.35, 1.5, 1.65, 1.8]

    # Load AABBs
    from Data_generation.bench_generation.batch_utils import load_scene_aabbs, load_scene_wall_aabbs
    aabbs = load_scene_aabbs(str(scene_path))
    aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

    # Load room polygons and determine which room contains both objects (X/Y only)
    room_polys = load_room_polys(str(scene_path))
    if not room_polys:
        return None, "no_room_polys"

    def _find_room_idx(pt: np.ndarray) -> int | None:
        x, y = float(pt[0]), float(pt[1])
        for i, poly in enumerate(room_polys):
            try:
                if point_in_poly(x, y, poly):
                    return i
            except Exception:
                continue
        return None

    room_a = _find_room_idx(a_center)
    room_b = _find_room_idx(b_center)
    if room_a is None or room_b is None or room_a != room_b:
        return None, "objects_not_in_same_room"
    target_room_idx = room_a


    tries = 0
    failure_counter = Counter()

    for r in radii:
        for z in heights:
            for yaw in np.linspace(0, 2 * math.pi, per_angle, endpoint=False):
                if tries >= max_tries:
                    break
                tries += 1
                
                cam_pos = np.array([mid[0] + r * math.cos(yaw), mid[1] + r * math.sin(yaw), z], dtype=float)
                # Look slightly down or at the center height of objects
                target_z = (a_center[2] + b_center[2]) / 2.0
                cam_tgt = np.array([mid[0], mid[1], target_z], dtype=float)

                # Ensure camera is placed inside the same room as the two objects
                try:
                    x, y = float(cam_pos[0]), float(cam_pos[1])
                    if not point_in_poly(x, y, room_polys[target_room_idx]):
                        failure_counter['camera_outside_room'] += 1
                        continue
                except Exception:
                    failure_counter['camera_outside_room'] += 1
                    continue

                # Reject cameras that are inside any object AABB
                try:
                    inside_obj = False
                    for obj_aabb in aabbs:
                        try:
                            if (cam_pos[0] >= float(obj_aabb.bmin[0]) - 1e-9 and cam_pos[0] <= float(obj_aabb.bmax[0]) + 1e-9 and
                                cam_pos[1] >= float(obj_aabb.bmin[1]) - 1e-9 and cam_pos[1] <= float(obj_aabb.bmax[1]) + 1e-9 and
                                cam_pos[2] >= float(obj_aabb.bmin[2]) - 1e-9 and cam_pos[2] <= float(obj_aabb.bmax[2]) + 1e-9):
                                inside_obj = True
                                break
                        except Exception:
                            continue
                    if inside_obj:
                        failure_counter['camera_inside_object'] += 1
                        continue
                except Exception:
                    failure_counter['camera_inside_object'] += 1
                    continue

                # Reject cameras too close to any wall (within 0.05 m)
                try:
                    from Data_generation.bench_generation.batch_utils import load_scene_wall_aabbs
                    walls = load_scene_wall_aabbs(str(scene_path))
                    too_close = False
                    for w in walls:
                        try:
                            d = point_to_aabb_distance(cam_pos, w.bmin, w.bmax)
                            if d < 0.05:
                                too_close = True
                                break
                        except Exception:
                            continue
                    if too_close:
                        failure_counter['camera_too_close_wall'] += 1
                        continue
                except Exception:
                    # if wall checks fail, count and skip
                    failure_counter['camera_wall_check_error'] += 1
                    continue

                # Check Object A
                okA, reasonA = visible_enough(scene_path, cam_pos, cam_tgt, a_min, a_max, K, aabbs_all)
                if not okA:
                    failure_counter[f"ObjA_{reasonA}"] += 1
                    continue
                
                # Check Object B
                okB, reasonB = visible_enough(scene_path, cam_pos, cam_tgt, b_min, b_max, K, aabbs_all)
                if not okB:
                    failure_counter[f"ObjB_{reasonB}"] += 1
                    continue

                # Success
                found = {
                    'camera_pos': cam_pos.tolist(),
                    'camera_target': cam_tgt.tolist(),
                    'radius': float(r),
                    'yaw': float(yaw),
                    'calculated_ideal_dist': float(ideal_dist)
                }

                # If requested, write a composed preview for this candidate
                if preview_dir is not None:
                    try:
                        preview_dir.mkdir(parents=True, exist_ok=True)
                        item = {
                            'qtype': 'multi_step_navigation_mca',
                            'scene': str(scene_path),
                            'question': 'candidate view',
                            'choices': [],
                            'answer': '',
                            'meta': {
                                'camera_pos': cam_pos.tolist(),
                                'camera_target': cam_tgt.tolist(),
                                'start_object': {'center': a_center.tolist(), 'display_label': la.get('label', '')},
                                'goal_object': {'center': b_center.tolist(), 'display_label': lb.get('label', '')},
                                'start_heading': (cam_tgt - cam_pos).tolist(),
                                'choices_actions': [],
                            }
                        }
                        name_idx = pair_idx if pair_idx is not None else 0
                        fname = preview_dir / f'pair_{name_idx:04d}_try_{tries:04d}.png'
                        compose_preview_for_item(item, scene_path, fname, thumb_size=WIDTH)
                    except Exception:
                        pass

                return found, "success"

    # If we reach here, we failed
    if tries == 0:
        return None, "setup_error"
    
    # Construct a summary failure reason
    most_common = failure_counter.most_common(1)
    if most_common:
        top_reason, count = most_common[0]
        return None, f"Max tries reached. Top reason: {top_reason} ({count}/{tries})"
    else:
        return None, "Max tries reached (unknown reason)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, type=Path)
    p.add_argument('--pairs', required=True, type=Path, help='JSON file with pairs')
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--per-angle', type=int, default=36)
    p.add_argument('--max-tries', type=int, default=500)
    p.add_argument('--render', action='store_true', help='Render thumbnail for found poses')
    p.add_argument('--preview-trials', action='store_true', help='Save preview images for each successful candidate during sampling')
    args = p.parse_args()

    scene = args.scene
    pairs = json.load(open(args.pairs, 'r', encoding='utf-8'))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, pair in enumerate(pairs):
        # pair parsing logic
        if isinstance(pair, list) and len(pair) >= 2:
            a, b = pair[0], pair[1]
        elif isinstance(pair, dict) and 'a' in pair and 'b' in pair:
            a, b = pair['a'], pair['b']
        else:
            try:
                a, b = pair[0], pair[1]
            except Exception:
                print(f"Skipping unsupported pair format at index {idx}: {pair}")
                continue

        # Call sampling with new return signature
        preview_dir = out_dir if args.preview_trials else None
        found, message = sample_camera_for_pair(scene, a, b, max_tries=args.max_tries, per_angle=args.per_angle, preview_dir=preview_dir, pair_idx=idx)
        
        out = {
            'pair': (a, b), 
            'found': found, 
            'status': 'success' if found else 'failed',
            'message': message
        }

        if found:
            print(f"Pair {idx}: Found pose at dist {found['radius']:.2f}m (Ideal: {found['calculated_ideal_dist']:.2f}m)")
            if args.render:
                try:
                    img = render_thumbnail_for_pose(scene, {
                        'position': np.array(found['camera_pos'], dtype=float), 
                        'target': np.array(found['camera_target'], dtype=float)
                    }, thumb_size=WIDTH)
                    
                    fp = out_dir / f'pair_{idx:04d}_view.png'
                    imageio.imwrite(str(fp), img)
                    out['view_path'] = str(fp)
                except Exception as e:
                    out['render_error'] = str(e)
                    print(f"  Render error: {e}")
        else:
            print(f"Pair {idx}: Failed. Reason: {message}")

        results.append(out)
        
        # write per-pair meta
        with open(out_dir / f'pair_{idx:04d}.json', 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    # summary
    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()