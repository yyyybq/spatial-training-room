#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch generate per-object views for all scenes under a parent folder.

For each scene (subfolder) the script will:
 - read labels.json to discover objects
 - for each object, call generate_camera_positions to produce candidate poses
 - for each accepted pose, save a per-pose meta JSON and a thumbnail PNG

Filename format (per pose):
  {scene_name}_{objid}_{sanitized_label}_room{room_idx}_view{view_idx}.json
  {scene_name}_{objid}_{sanitized_label}_room{room_idx}_view{view_idx}.png

Usage:
    python -m sampler.batch_generate_views --scenes_root /path/to/data --out ./out_views --per_room_points 12

"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import List

import imageio
import numpy as np

from sampler.generate_view import (
    SceneObject,
    generate_camera_positions,
    render_thumbnail_for_pose,
)
from bench_generation.qa_batch_generator import load_room_polys, point_in_poly, load_scene_aabbs
from utils.occlusion import occluded_area_on_image, camtoworld_from_pos_target


def sanitize_label(label: str) -> str:
    # simple sanitize for filenames
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in label)[:60]


def find_room_index_for_object(scene_root: Path, obj_center_xy: List[float]) -> int:
    """Return the index of the first room polygon that contains the object center, or -1 if none."""
    polys = load_room_polys(str(scene_root))
    if not polys:
        return -1
    for i, poly in enumerate(polys):
        try:
            if point_in_poly(float(obj_center_xy[0]), float(obj_center_xy[1]), np.array(poly)):
                return i
        except Exception:
            continue
    return -1


def process_scene(scene_path: Path, out_dir: Path, per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5, exclude_ids=None):
    labels_path = scene_path / 'labels.json'
    if not labels_path.exists():
        print(f"[warn] labels.json not found in {scene_path}, skip")
        return

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    # build objects list using the SceneObject helper
    objects = {}
    for item in labels:
        if isinstance(item, dict) and item.get('ins_id') and 'bounding_box' in item and len(item.get('bounding_box', [])) >= 4:
            obj = SceneObject(item)
            objects[obj.id] = obj

    if not objects:
        print(f"[info] no valid objects in {scene_path}")
        return

    scene_name = scene_path.name
    scene_out = out_dir / scene_name
    scene_out.mkdir(parents=True, exist_ok=True)

    # load scene aabbs for occlusion computations
    aabbs = load_scene_aabbs(str(scene_path))

    for obj_id, obj in objects.items():
        label_safe = sanitize_label(obj.label)
        # exclude by provided blank list (object ids)
        if exclude_ids and obj_id in exclude_ids:
            print(f"[info] scene={scene_name} obj={obj_id} ({obj.label}): in exclude list, skip")
            continue
        room_idx = find_room_index_for_object(scene_path, obj.position[:2])
        if room_idx == -1:
            print(f"[info] scene={scene_name} obj={obj_id} ({obj.label}): object center not inside any room, skip")
            continue
        poses = generate_camera_positions(scene_path, obj, per_room_points=per_room_points, min_dist=min_dist, max_dist=max_dist)
        if not poses:
            print(f"[info] scene={scene_name} obj={obj_id} ({obj.label}): no valid poses")
            continue
        # save each pose separately
        for i, p in enumerate(poses):
            fname_base = f"{scene_name}_{obj_id}_{label_safe}_room{room_idx}_view{i:03d}"

            # compute image-space target area ratio using occlusion helper
            try:
                width = 400
                height = 400
                focal = float(width * 0.4)
                K = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=float)
                pos_np = np.array(p['position'], dtype=float)
                tgt_np = np.array(p['target'], dtype=float)
                camtoworld = camtoworld_from_pos_target(pos_np, tgt_np)
                occ_res = occluded_area_on_image(pos_np, np.array(obj.bbox_min), np.array(obj.bbox_max), aabbs, K, camtoworld, width, height, target_id=obj_id, depth_mode='min', return_per_occluder=False)
                target_px = float(occ_res.get('target_area_px', 0.0))
                target_image_ratio = float(target_px) / float(width * height)
            except Exception:
                target_image_ratio = 0.0

            meta = {
                'scene': scene_name,
                'object_id': obj_id,
                'label': obj.label,
                'bbox_min': [float(x) for x in obj.bbox_min.tolist()],
                'bbox_max': [float(x) for x in obj.bbox_max.tolist()],
                'pose_index': int(i),
                'position': [float(x) for x in p['position'].tolist()],
                'target': [float(x) for x in p['target'].tolist()],
                'forward': [float(x) for x in p.get('forward', np.array([0.0, 0.0, 0.0])).tolist()],
                'occlusion_ratio': float(p.get('occlusion_ratio', 1.0)),
                'target_image_ratio': float(target_image_ratio),
                # relative angle/direction removed by user request
            }
            meta_path = scene_out / (fname_base + '.json')
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
            # render thumbnail (try/except to avoid hard crash on render errors)
            try:
                img = render_thumbnail_for_pose(scene_path, p, thumb_size=256)
                if img is not None:
                    img_path = scene_out / (fname_base + '.png')
                    imageio.imwrite(str(img_path), img)
            except Exception as e:
                print(f"[warn] render failed for {fname_base}: {e}")

        print(f"[info] scene={scene_name} obj={obj_id} ({obj.label}): saved {len(poses)} poses to {scene_out}")


def iter_scene_folders(root: Path) -> List[Path]:
    # list directories directly under root
    kids = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return kids


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_root', required=True, help='Parent folder that contains scene subfolders')
    parser.add_argument('--out', required=True, help='Output folder to store per-pose meta and thumbnails')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--exclude_file', type=str, default=None, help='Path to a text file listing object ids to exclude, one per line')
    parser.add_argument('--exclude_ids', type=str, default=None, help='Comma-separated object ids to exclude')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = iter_scene_folders(scenes_root)
    if not scenes:
        print(f"[error] no scene subfolders found in {scenes_root}")
        return

    # build exclude set
    exclude_set = set()
    if getattr(args, 'exclude_file', None):
        p = Path(args.exclude_file)
        if p.exists():
            with open(p, 'r') as ef:
                for ln in ef:
                    s = ln.strip()
                    if s:
                        exclude_set.add(s)
        else:
            print(f"[warn] exclude_file not found: {p}")
    if getattr(args, 'exclude_ids', None):
        for s in args.exclude_ids.split(','):
            s2 = s.strip()
            if s2:
                exclude_set.add(s2)

    for sc in scenes:
        process_scene(sc, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, exclude_ids=exclude_set)


if __name__ == '__main__':
    main_cli()
