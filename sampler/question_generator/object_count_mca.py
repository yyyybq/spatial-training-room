#!/usr/bin/env python3
"""
Generate `object_count_mca` items using sampler-style view sampling.

This generator samples camera poses per object using `generate_camera_positions`,
then for each sampled pose selects a category (label) and asks: "How many {label}s
are there in the picture?" The count is computed as the number of visible instances
with that label (visibility determined by corner visibility + occlusion area).

The generator attempts to produce a balanced distribution of counts 0..5 by
enforcing per-count quotas close to 1/6 of `--max_items`.

Per-item outputs (when --out-dir provided):
  - meta.json (contains camera pose, asked label, visible instances info and answer)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)

Run example:
  python -m Data_generation.sampler.question_generator.object_count_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/object_count.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/object_count_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer (no lazy import)
from ...bench_generation.preview import compose_preview_for_item, compose_view_map

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


def is_instance_visible(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    """Return True if object `so` is visible from camera (pos->tgt).

    Visibility criteria:
      - at least 4 box corners visible (using count_visible_corners_for_box)
      - occluded target area px ratio > 0.01
    """
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, WIDTH, HEIGHT, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 2:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(WIDTH * HEIGHT)
    return target_ratio > 0.03


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(202312)
    items: List[Dict[str, Any]] = []

    # quotas for counts 0..5 (try to make them ~equal -> total ~max_items)
    base = max_items // 6
    rem = max_items - base * 6
    quotas = [base + (1 if i < rem else 0) for i in range(6)]
    generated_counts = [0] * 6

    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")

        labels_file = scene_path / 'labels.json'
        if not labels_file.exists():
            print(f"[warn] no labels.json in {scene_path}, skip")
            continue
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)

        objects: Dict[str, SceneObject] = {}
        # build SceneObject map as other generators do
        for it in labels:
            if not isinstance(it, dict):
                continue
            if not it.get('ins_id') or 'bounding_box' not in it:
                continue
            obj = SceneObject(it)
            objects[obj.id] = obj

        if not objects:
            print(f"[info] no valid objects in {scene_path}")
            continue

        aabbs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

        per_scene_count = 0

        # Precompute label -> list of object ids for sampling labels
        label_to_objs: Dict[str, List[SceneObject]] = {}
        for so in objects.values():
            label = str(so.label)
            label_to_objs.setdefault(label, []).append(so)

        labels_pool = list(label_to_objs.keys())
        if not labels_pool:
            continue

        for obj_id, obj in objects.items():
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break

            poses = generate_camera_positions(scene_path, obj, per_room_points=per_room_points, min_dist=min_dist, max_dist=max_dist)
            if not poses:
                if verbose:
                    print(f"[info] no poses for scene={scene_path.name} obj={obj_id} ({obj.label})")
                continue

            for pi, p in enumerate(poses):
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

                pos = np.array(p['position'], dtype=float)
                tgt = np.array(p['target'], dtype=float)

                # build visible list using occlusion + corner visibility checks
                visible: List[SceneObject] = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                for oid, so in objects.items():
                    if so.label in BLACKLIST:
                        continue
                    if is_instance_visible(pos, tgt, so, aabbs_all, K, WIDTH, HEIGHT):
                        visible.append(so)

                if not visible:
                    if verbose:
                        print(f"[skip] no visible objects after filtering for scene={scene_path.name} pose_index={pi}")
                    continue

                # Determine labels present in view and their counts
                labels_in_view = list({str(so.label) for so in visible})
                if not labels_in_view:
                    continue
                counts_map = {}
                for lab in labels_in_view:
                    counts_map[lab] = sum(1 for so in visible if str(so.label) == lab)

                # Strategy: prefer the label with maximum count in view (ties random).
                # If its quota is full, try other labels in descending count order.
                sorted_labels = sorted(counts_map.items(), key=lambda x: (-x[1], x[0]))
                # collect candidates in order
                candidate = None
                for lab, lab_cnt in sorted_labels:
                    if lab_cnt <= 5 and generated_counts[lab_cnt] < quotas[lab_cnt]:
                        candidate = (lab, lab_cnt)
                        break

                # If no in-view label satisfies quotas, attempt to create a 0-count example by
                # picking a label that exists in the scene but is NOT visible in this view.
                if candidate is None:
                    scene_labels = list(label_to_objs.keys())
                    labels_not_in_view = [l for l in scene_labels if l not in labels_in_view and l not in BLACKLIST]
                    if labels_not_in_view and generated_counts[0] < quotas[0]:
                        # produce a 0-count item
                        lab0 = rng.choice(labels_not_in_view)
                        candidate = (lab0, 0)

                # As a last resort, accept the highest-count label even if its quota is full
                if candidate is None:
                    lab, lab_cnt = sorted_labels[0]
                    candidate = (lab, lab_cnt)

                label_choice, cnt = candidate

                # create item
                camtworld = camtoworld_from_pos_target(pos, tgt)
                render = create_intrinsics()
                render['camtoworld'] = camtworld.tolist()

                # prepare choices: select distractors near the correct count to avoid
                # widely separated options. Prefer neighbors cnt+-1, cnt+-2, etc.
                neighbors = []
                for offset in (1, -1, 2, -2, 3):
                    v = cnt + offset
                    if 0 <= v <= 5 and v not in neighbors:
                        neighbors.append(v)
                    if len(neighbors) >= 3:
                        break
                # if not enough neighbors, fill with remaining values
                if len(neighbors) < 3:
                    for v in range(0,6):
                        if v == cnt or v in neighbors:
                            continue
                        neighbors.append(v)
                        if len(neighbors) >= 3:
                            break

                opts = [cnt] + neighbors[:3]
                # shuffle options but keep them reasonably local to correct
                rng.shuffle(opts)
                choices = [str(x) for x in opts]
                labels_ABC = ['A','B','C','D']
                answer = labels_ABC[choices.index(str(cnt))]

                # prepare visible_instances listing centers for the asked label only (for preview markers)
                vis_instances = []
                for so in visible:
                    if str(so.label) == label_choice:
                        center = 0.5 * (so.bbox_min + so.bbox_max)
                        vis_instances.append({'id': so.id, 'label': str(so.label), 'center': center.tolist()})

                item = {
                    'qtype': 'object_count',
                    'scene': str(scene_path),
                    'question': f"How many {label_choice}s are there in the picture?",
                    'choices': choices,
                    'answer': answer,
                    'meta': {
                        'category': label_choice,
                        'camera_pos': pos.tolist(),
                        'camera_target': tgt.tolist(),
                        'render': render,
                        'visible_instances': vis_instances,
                    }
                }

                # accept and record
                items.append(item)
                generated_counts[cnt] += 1
                per_scene_count += 1

                # save per-item outputs if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    fname_base = f"{scene_path.name}_item_{idx:04d}_object_count_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)
                    # render preview and view
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                if len(items) >= max_items:
                    break

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    print(f"Generated counts distribution: {generated_counts}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate object_count_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=3.5)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    parser.add_argument('--verbose', action='store_true', help='Print debug/skip reasons during generation')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scenes_root, out_path, out_dir, per_room_points=args.per_room_points, min_dist=args.min_dist, max_dist=args.max_dist, max_items=args.max_items, max_items_per_scene=args.max_items_per_scene, render=args.render, verbose=args.verbose)
    print(f"Wrote {len(items)} object_count_mca items to {out_path}")


if __name__ == '__main__':
    main()
