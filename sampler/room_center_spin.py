#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a short spinning video from each room's center so you can visually inspect
the most common object class in that room.
python -m Data_generation.sampler.room_center_spin --scene /data/liubinglin/jijiatong/ViewSuite/data/InteriorGS/0267_840790 --out /data/liubinglin/jijiatong/ViewSuite/Data_generation/video --thumb 400 --frames 120 --fps 24
Behavior:
- Parse `labels.json` and `structure.json` to find room polygons and objects.
- For each room, pick the object class (label) with the largest count among objects
  whose centers fall inside that room.
- Compute room centroid as preferred camera position; if centroid is illegal (inside
  object or colliding with walls), sample nearby legal points and pick the closest.
- Keep camera position fixed and rotate yaw 0..360°, rendering frames using
  `Data_generation.bench_generation.preview.render_thumbnail_for_pose`.
- Save per-room MP4 videos to output folder.

This script re-uses the project's helpers in `bench_generation` and `sampler`.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import imageio
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from pathlib import Path as _Path
from tqdm import tqdm
import time
import torch


def _render_helper(args):
    """Top-level helper so it can be pickled for ProcessPoolExecutor.

    args: tuple(scene_path_str, pose_dict, thumb, room_idx, gpu_id)
    """
    scene_path_str, pose_dict, thumb, room_idx, gpu_id = args
    import torch
    torch.cuda.set_device(gpu_id)  # Set the GPU for this process
    img = render_thumbnail_for_pose(_Path(scene_path_str), pose_dict, thumb_size=thumb)
    return img

# project helpers
from Data_generation.bench_generation.batch_utils import (
    load_room_polys,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    ensure_position_legal,
    load_structure_height_bounds,
    point_in_poly,
    count_visible_corners_for_box,
)
from Data_generation.bench_generation.preview import render_thumbnail_for_pose
from Data_generation.utils.occlusion import occluded_area_on_image, camtoworld_from_pos_target

# try to import batch rendering helpers from preview
from Data_generation.bench_generation.preview import (
    prepare_gaussian_data,
    create_camera_intrinsics,
    setup_camera,
    render_gaussians,
    PLYGaussianLoader,
)
_HAVE_BATCH_RENDER = True

# reuse sampling helper from sampler.generate_view
from Data_generation.sampler.generate_view import sample_points_in_room


def find_objects_by_room(scene_path: Path):
    labels_p = scene_path / 'labels.json'
    if not labels_p.exists():
        return {}
    with open(labels_p, 'r', encoding='utf-8') as f:
        labels = json.load(f)

    polys = load_room_polys(str(scene_path))
    room_objs = defaultdict(list)

    for item in labels:
        if not isinstance(item, dict) or not item.get('ins_id'):
            continue
        if 'bounding_box' not in item or len(item['bounding_box']) < 1:
            continue
        xs = [p['x'] for p in item['bounding_box']]
        ys = [p['y'] for p in item['bounding_box']]
        zs = [p['z'] for p in item['bounding_box']]
        center = np.array([(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0], dtype=float)
        # find first room that contains center
        placed = False
        for ridx, poly in enumerate(polys):
                if point_in_poly(float(center[0]), float(center[1]), np.array(poly)):
                    room_objs[ridx].append({'ins_id': item.get('ins_id'), 'label': item.get('label'), 'center': center, 'bbox': item.get('bounding_box')})
                    placed = True
                    break
        # if not placed, ignore for room-based videos
    return room_objs, polys


def choose_camera_position_for_room(scene_path: Path, poly: np.ndarray, aabbs, min_dist: float = 0.4, min_h=None, max_h=None):
    # try centroid first
    centroid_xy = np.mean(np.array(poly, dtype=float), axis=0)
    lower, upper = load_structure_height_bounds(str(scene_path))
    if min_h is None and lower is not None:
        min_h = lower
    if max_h is None and upper is not None:
        max_h = upper
    # choose a reasonable height in [min_h,max_h] or default 1.0
    if min_h is None or max_h is None:
        z = 1.0
    else:
        z = float(0.5 * (min_h + max_h))

    cand = np.array([float(centroid_xy[0]), float(centroid_xy[1]), z], dtype=float)
    # check not inside any object and not too close to object centers
    def _too_close(pt):
        for b in aabbs:
            center = 0.5 * (b.bmin + b.bmax)
            if np.linalg.norm(pt - center) < float(min_dist):
                return True
        return False

    if ensure_position_legal(str(scene_path), cand, min_h=min_h, max_h=max_h, tol=0.5) and (not _too_close(cand)):
        return cand

    # fallback: sample candidates inside room and pick nearest legal
    pts = sample_points_in_room(poly, count=60, min_height=(min_h or 0.6), max_height=(max_h or 1.2))
    best = None
    best_dist = float('inf')
    for p in pts:
        if ensure_position_legal(str(scene_path), p, min_h=min_h, max_h=max_h, tol=0.5) and (not _too_close(p)):
            d = np.linalg.norm(p[:2] - centroid_xy[:2])
            if d < best_dist:
                best_dist = d
                best = p
    return best


def sample_camera_positions_for_room(scene_path: Path, poly: np.ndarray, aabbs, trials: int = 3, min_dist: float = 0.4, min_h=None, max_h=None):
    """Return up to `trials` legal camera 3D positions inside `poly`.

    Uses `sample_points_in_room` to propose candidates and filters by
    `ensure_position_legal` and distance-to-object rules.
    """
    pts = sample_points_in_room(poly, count=max(20, trials * 10), min_height=(min_h or 0.6), max_height=(max_h or 1.2))
    chosen = []

    def _too_close(pt):
        for b in aabbs:
            center = 0.5 * (b.bmin + b.bmax)
            if np.linalg.norm(pt - center) < float(min_dist):
                return True
        return False

    lower, upper = load_structure_height_bounds(str(scene_path))
    if min_h is None and lower is not None:
        min_h = lower
    if max_h is None and upper is not None:
        max_h = upper

    for p in pts:
        if len(chosen) >= trials:
            break
        if ensure_position_legal(str(scene_path), p, min_h=min_h, max_h=max_h, tol=0.5) and (not _too_close(p)):
            chosen.append(p)

    # If we didn't find enough, try centroid as a fallback
    if len(chosen) < trials:
        centroid_xy = np.mean(np.array(poly, dtype=float), axis=0)
        if min_h is None or max_h is None:
            z = 1.0
        else:
            z = float(0.5 * (min_h + max_h))
        cand = np.array([float(centroid_xy[0]), float(centroid_xy[1]), z], dtype=float)
        if ensure_position_legal(str(scene_path), cand, min_h=min_h, max_h=max_h, tol=0.5) and (not _too_close(cand)):
            chosen.append(cand)

    return chosen


def render_room_spin(scene_path: Path, room_idx: int, poly: np.ndarray, out_dir: Path, thumb=400, frames=120, fps=24, workers: int = 4, attempt_idx: int = 0, pos: np.ndarray = None):
    aabbs = load_scene_aabbs(str(scene_path))
    if pos is None:
        pos = choose_camera_position_for_room(scene_path, poly, aabbs)
    if pos is None:
        print(f"[warn] room {room_idx} attempt{attempt_idx}: no legal camera position found, skip")
        return None

    # render frames rotating around yaw
    angles = list(np.linspace(0.0, 2.0 * math.pi, num=frames, endpoint=False))
    poses = []
    for i, ang in enumerate(angles):
        forward = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=float)
        target = pos + forward
        p = {'position': pos.copy(), 'target': target.copy(), 'index': i}
        poses.append(p)
    # -----------------------------
    # Pre-render visibility check across all poses (avoid rendering if nothing meets criteria)
    # Criteria per-frame per-object: image area ratio >= 0.01 and visible corner count > 1
    # We only keep objects that appear in >1 distinct frames.
    # -----------------------------
    width = thumb; height = thumb
    focal = float(width * 0.4)
    K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=float)
    # include wall aabbs as occluders and filter to those inside the same room polygon
    aabbs_all_raw = load_scene_aabbs(str(scene_path)) + load_scene_wall_aabbs(str(scene_path))
    # filter to AABBs whose center lies inside current room polygon
    aabbs_all = []
    for b in aabbs_all_raw:
        center = 0.5 * (b.bmin + b.bmax)
        if point_in_poly(float(center[0]), float(center[1]), np.array(poly)):
            aabbs_all.append(b)
    visible_frames = {}

    def _check_frame_visibility(args):
        fi, p = args
        pos_np = np.array(p['position'], dtype=float)
        tgt_np = np.array(p['target'], dtype=float)
        camtoworld = camtoworld_from_pos_target(pos_np, tgt_np)
        found = []
        for b in aabbs_all:
            occ_res = occluded_area_on_image(pos_np, np.array(b.bmin), np.array(b.bmax), aabbs_all, K, camtoworld, width, height, target_id=getattr(b, 'id', None), depth_mode='min', return_per_occluder=False)
            target_px = float(occ_res.get('target_area_px', 0.0))
            target_ratio = target_px / float(width * height)
            if target_ratio < 0.01:
                continue
            cnt = count_visible_corners_for_box(pos_np, tgt_np, np.array(K), np.array(b.bmin), np.array(b.bmax), width, height, corner_threshold=2, aabbs_all=aabbs_all, target_id=getattr(b, 'id', None), require_unoccluded=True)
            if cnt <= 1:
                continue
            oid = getattr(b, 'id', None)
            if oid is None:
                continue
            found.append((oid, fi))
        return found

    start_check = time.time()
    # parallel pre-check across frames (CPU-bound checks)
    with ThreadPoolExecutor(max_workers=min(8, max(1, int(workers)))) as exe:
        for res in tqdm(exe.map(_check_frame_visibility, enumerate(poses)), total=len(poses), desc=f"room{room_idx} precheck"):
            for oid, fi in res:
                visible_frames.setdefault(oid, set()).add(fi)
    dur_check = time.time() - start_check
    print(f"[info] room{room_idx} precheck done in {dur_check:.2f}s, found {len(visible_frames)} candidate ids")

    # aggregate visible ids (unique across frames)
    visible_ids = list(visible_frames.keys())
    if not visible_ids:
        print(f"[info] room {room_idx} attempt{attempt_idx}: no visible objects in any frame, skip rendering")
        return None

    report = {
        'scene': str(scene_path.name),
        'room_idx': int(room_idx),
        'attempt_idx': int(attempt_idx),
        'unique_visible_ids': visible_ids,
    }
    # map visible ids to labels and count per-label occurrences
    id_to_label = {getattr(b, 'id', None): getattr(b, 'label', 'UNKNOWN') for b in aabbs_all}
    from collections import Counter as _Counter
    label_counter = _Counter()
    for oid in visible_ids:
        label_counter[id_to_label.get(oid, 'UNKNOWN')] += 1
    report['label_counts'] = dict(label_counter)
    report['labels'] = list(label_counter.keys())
    report['num_labels'] = int(len(report['labels']))
    import json as _json
    rpt_path = out_dir / f"{scene_path.name}_room{room_idx}_attempt{attempt_idx}_visibility.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(rpt_path, 'w', encoding='utf-8') as _f:
        _json.dump(report, _f, indent=2, ensure_ascii=False)
    print(f"[info] wrote visibility report {rpt_path}")

    # proceed to rendering since we have visible objects

    # Proceed to rendering
    images = []

    # Render frames in parallel when workers > 1, preserving order
    if workers and int(workers) > 1:
        num_gpus = torch.cuda.device_count()
        args_list = [
            (str(scene_path), p, thumb, int(room_idx), i % num_gpus)  # Assign GPU in round-robin
            for i, p in enumerate(poses)
        ]
        try:
            # Prefer process-based parallelism for CPU-bound rendering
            with ProcessPoolExecutor(max_workers=int(workers)) as exe:
                for img in tqdm(exe.map(_render_helper, args_list), total=len(args_list), desc=f"room{room_idx} render"):
                    if img is not None:
                        images.append(img)
        except Exception as e:
            print(f"[warn] room {room_idx}: process pool failed ({e}); falling back to threads")
            # Thread-based fallback if processes fail (e.g., pickling, GPU context)
            from functools import partial
            render_fn = partial(render_thumbnail_for_pose, scene_path, thumb_size=thumb)
            with ThreadPoolExecutor(max_workers=int(workers)) as exe:
                for img in tqdm(exe.map(render_fn, poses), total=len(poses), desc=f"room{room_idx} render-t"):
                    if img is not None:
                        images.append(img)
    else:
        # Sequential rendering when workers <= 1
        for p in poses:
            img = render_thumbnail_for_pose(scene_path, p, thumb_size=thumb)
            if img is not None:
                images.append(img)

    if not images:
        print(f"[warn] room {room_idx}: no frames rendered")
        return None

    # (pre-check already computed and report saved before rendering)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_path.name}_room{room_idx}_attempt{attempt_idx}_spin.mp4"
    # write mp4 using imageio-ffmpeg
    writer = imageio.get_writer(str(out_path), fps=fps, codec='libx264', macro_block_size=None)
    for im in tqdm(images, desc=f"room{room_idx} write", total=len(images)):
        writer.append_data(im)
    writer.close()
    print(f"[ok] wrote {out_path}")
    return out_path


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Path to a scene folder (contains labels.json and 3dgs_compressed.ply)')
    parser.add_argument('--out', required=True, help='Output folder for generated videos')
    parser.add_argument('--thumb', type=int, default=400)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for parallel rendering')
    parser.add_argument('--trials', type=int, default=6, help='Number of camera position attempts per room')
    args = parser.parse_args()

    scene = Path(args.scene)
    out = Path(args.out)
    if not scene.exists():
        raise FileNotFoundError(f"scene not found: {scene}")

    room_objs, polys = find_objects_by_room(scene)
    if not polys:
        print(f"[error] no rooms found for scene {scene}")
        return
    # sample multiple camera positions per room and render each attempt
    aabbs = load_scene_aabbs(str(scene))
    for ridx, poly in enumerate(polys):
        candidates = sample_camera_positions_for_room(scene, poly, aabbs, trials=args.trials)
        if not candidates:
            print(f"[info] room {ridx}: no legal camera positions found, skip")
            continue
        for attempt_idx, pos in enumerate(candidates):
            print(f"[info] scene={scene.name} room={ridx} attempt={attempt_idx}: rendering")
            render_room_spin(scene, ridx, poly, out, thumb=args.thumb, frames=args.frames, fps=args.fps, workers=args.workers, attempt_idx=attempt_idx, pos=pos)


if __name__ == '__main__':
    main_cli()
