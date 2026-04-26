#!/usr/bin/env python3
"""
Generate `chain_position_reasoning_mca` items using sampler-style view sampling.

Problem type (Chain Reasoning over relative positions):
Example: "If the bed is north of the table and the table is west of the door, where is the bed relative to the door?"

Sampling policy:
- Use the sampler method (generate_camera_positions) to get valid camera poses.
- For a given pose, find objects visible in the image (target area ratio > 0.1, and at least 1 visible box corner).
- If there exist 3 objects with pairwise center distances >= 0.5m (in XY) and no one is vertically "above" another (max |z_diff| within the triple < 0.4m),
  then select a triple and construct a chain question using a chosen "North" direction.

North direction:
- Define North as the camera forward vector projected to the XY plane at the selected pose, normalized.
- East is the right vector in XY (per-camera), normalized. These define a local frame for relations.

Answer and choices:
- Compute the final relation between A and C (8-way: N, S, E, W, NE, NW, SE, SW), output choices as
  ["North","South","East","West","NorthEast","NorthWest","SouthEast","SouthWest"].
- The correct answer is one of these, recorded as letter 'A'..'H'.

Rendering:
- Must render: per-item outputs include meta.json, preview.png, and view.png.
- Thumbnails use WIDTH from batch_utils as the size.

Run example:
  python -m Data_generation.sampler.question_generator.chain_position_reasoning_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/chain_position.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/chain_items3 \
    --per_room_points 12 --max_items 200
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
        return np.array([0.5, 0.0], dtype=float)
    return v2 / n


def right_xy(forward_xy: np.ndarray) -> np.ndarray:
    # 2D right perpendicular (rotate forward by -90 deg)
    f = np.array(forward_xy, dtype=float)
    return np.array([f[1], -f[0]], dtype=float)


def visible_area_ratio_and_corners(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> Tuple[float, int]:
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    camtoworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = target_px / float(width * height)
    return target_ratio, corners_visible


def compute_visibility_metrics(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> Tuple[float, float]:
    """Compute two visibility percentages for an object at a given pose:
    - visible_view_percent: visible (non-occluded) object pixels divided by full image pixels (W*H).
    - visible_object_percent: visible (non-occluded) object pixels divided by the object's in-frame projected pixels if unoccluded
      (i.e., visible_px / (visible_px + occluded_px)).

    If occluded_area_on_image doesn't return occluded_area_px explicitly, we derive it using occlusion_ratio_target when available.
    """
    camtoworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    visible_px = float(occ.get('target_area_px', 0.0))  # non-occluded target pixels
    # prefer direct occluded px if provided
    occluded_px = float(occ.get('occluded_area_px', 0.0)) if 'occluded_area_px' in occ else 0.0
    if occluded_px <= 0.0:
        occ_ratio = float(occ.get('occlusion_ratio_target', 0.0))
        if 0.0 <= occ_ratio < 1.0:
            # visible = (1 - r) * total -> total = visible / (1 - r)
            total_px_est = visible_px / max(1e-6, (1.0 - occ_ratio))
            occluded_px = max(0.0, total_px_est - visible_px)
    total_px = max(0.0, visible_px + occluded_px)
    visible_view_percent = visible_px / float(width * height)
    visible_object_percent = (visible_px / total_px) if total_px > 0 else 0.0
    return float(visible_view_percent), float(visible_object_percent)


def categorize_relation(vec_xy: np.ndarray, north_xy: np.ndarray, east_xy: np.ndarray) -> str:
    # Quantize angle to 8 sectors centered at N, NE, E, SE, S, SW, W, NW
    # Compute angle in local frame
    x = float(np.dot(vec_xy, east_xy))
    y = float(np.dot(vec_xy, north_xy))
    ang = math.atan2(y, x)  # angle from East toward North
    # Map to 8 bins: E(0), NE(45), N(90), NW(135), W(180/-180), SW(-135), S(-90), SE(-45)
    deg = math.degrees(ang)
    # shift to [0,360)
    if deg < 0:
        deg += 360.0
    sectors = [
        (0, "East"),
        (45, "NorthEast"),
        (90, "North"),
        (135, "NorthWest"),
        (180, "West"),
        (225, "SouthWest"),
        (270, "South"),
        (315, "SouthEast"),
        (360, "East"),  # wrap
    ]
    # Find nearest sector center among [0,45,90,...,315]
    centers = [s[0] for s in sectors[:-1]]
    best_label = "East"
    best_d = 1e9
    for c, lab in zip(centers, [s[1] for s in sectors[:-1]]):
        d = abs((deg - c + 180) % 360 - 180)  # circular distance
        if d < best_d:
            best_d = d
            best_label = lab
    return best_label


def categorize_relation_4(vec_xy: np.ndarray, north_xy: np.ndarray, east_xy: np.ndarray) -> str:
    """Quantize relation to 4 principal directions: North/South/East/West.
    Uses the dominant axis (larger absolute component in local east/north frame).
    """
    x = float(np.dot(vec_xy, east_xy))
    y = float(np.dot(vec_xy, north_xy))
    if abs(x) >= abs(y):
        return "East" if x >= 0 else "West"
    else:
        return "North" if y >= 0 else "South"


def project_point_to_image_xy(K: np.ndarray, camtoworld: np.ndarray, xyz: np.ndarray) -> Tuple[float, float]:
    """Project a world point xyz onto the image using K and camtoworld.
    Returns (u,v). If depth<=0, returns (+inf,+inf) to push it to the right/bottom.
    """
    world = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    w2c = np.linalg.inv(camtoworld)
    cam = w2c @ world
    z = float(cam[2])
    if z <= 1e-6:
        return float('inf'), float('inf')
    x = float(cam[0])
    y = float(cam[1])
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return u, v


def ordinal(n: int) -> str:
    n = int(n)
    if 10 <= (n % 100) <= 20:
        suf = 'th'
    else:
        suf = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suf}"


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200,
                         verbose: bool = False) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

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

        for obj_id, obj in objects.items():
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break

            poses = generate_camera_positions(scene_path, obj,
                                              per_room_points=per_room_points,
                                              min_dist=min_dist,
                                              max_dist=max_dist)
            if not poses:
                continue

            K = np.array(create_intrinsics()['K'], dtype=float)

            for pi, p in enumerate(poses):
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

                pos = np.array(p['position'], dtype=float)
                tgt = np.array(p['target'], dtype=float)

                # Determine local North/East in XY from camera forward
                fwd = tgt - pos
                fwd_xy = unit_xy(fwd)
                east_xy = unit_xy(right_xy(fwd_xy))
                north_xy = unit_xy(fwd_xy)

                # Build visible set with required constraints
                visibles: List[Tuple[SceneObject, float, int]] = []  # (obj, area_ratio, visible_corners)
                for so in objects.values():
                    # Skip target object if blacklisted label
                    if str(so.label).lower() in BLACKLIST:
                        continue
                    ratio, corners = visible_area_ratio_and_corners(pos, tgt, so, aabbs_all, K, WIDTH, HEIGHT)
                    if ratio > 0.1 and corners >= 1:
                        visibles.append((so, ratio, corners))
                if len(visibles) < 3:
                    continue

                # Number duplicates left-to-right in the current view (by image x)
                camtoworld = camtoworld_from_pos_target(pos, tgt)
                # compute image x for sorting; fall back to world x if projection inf
                per_label: Dict[str, List[Tuple[float, SceneObject]]] = {}
                for so, _, _ in visibles:
                    u, v = project_point_to_image_xy(K, camtoworld, np.array(so.position, dtype=float))
                    key = str(so.label)
                    per_label.setdefault(key, []).append((u, so))
                display_name: Dict[str, str] = {}
                for lab, arr in per_label.items():
                    # sort by u ascending (left to right)
                    arr_sorted = sorted(arr, key=lambda t: t[0])
                    for idx, (_, so) in enumerate(arr_sorted, start=1):
                        display_name[so.id] = f"the {ordinal(idx)} {lab}"

                # pick triples satisfying pairwise XY distance >= 1.0m and small Z variation
                centers = {so.id: np.array(so.position, dtype=float) for (so, _, _) in visibles}

                ids = list(centers.keys())
                found = False
                triple_ids: Tuple[str, str, str] | None = None
                for i in range(len(ids)):
                    if found: break
                    for j in range(i+1, len(ids)):
                        if found: break
                        for k in range(j+1, len(ids)):
                            ida, idb, idc = ids[i], ids[j], ids[k]
                            ca, cb, cc = centers[ida], centers[idb], centers[idc]
                            def dxy(u, v):
                                return float(np.linalg.norm((u - v)[:2]))
                            if dxy(ca, cb) < 0.4 or dxy(cb, cc) < 0.4 or dxy(ca, cc) < 0.4:
                                continue
                            zs = [ca[2], cb[2], cc[2]]
                            if max(zs) - min(zs) > 1.8:  # avoid vertical stacking
                                continue
                            triple_ids = (ida, idb, idc)
                            found = True
                            break

                if not found or triple_ids is None:
                    continue

                # Order triple as (A,B,C)
                ida, idb, idc = triple_ids
                A, B, C = objects[ida], objects[idb], objects[idc]

                # We want to select among the triple a pair that is aligned in an exact
                # cardinal direction (North/South/East/West) in world XY. The relation
                # should be based on object-object positions (world axes), not camera.
                def cardinal_dir_between(P: SceneObject, Q: SceneObject, tol_align: float = 0.3, min_sep: float = 0.5) -> str | None:
                    dx = float(P.position[0]) - float(Q.position[0])
                    dy = float(P.position[1]) - float(Q.position[1])
                    # North/South: x nearly equal, y differs
                    if abs(dx) <= tol_align and abs(dy) >= min_sep:
                        return "North" if dy > 0 else "South"
                    # East/West: y nearly equal, x differs
                    if abs(dy) <= tol_align and abs(dx) >= min_sep:
                        return "East" if dx > 0 else "West"
                    return None

                def rel4_world(P: SceneObject, Q: SceneObject) -> str:
                    dx = float(P.position[0]) - float(Q.position[0])
                    dy = float(P.position[1]) - float(Q.position[1])
                    # prioritize dominant axis
                    if abs(dx) >= abs(dy):
                        return "East" if dx >= 0 else "West"
                    else:
                        return "North" if dy >= 0 else "South"

                def rel8_world(P: SceneObject, Q: SceneObject) -> str:
                    # use world axes: east=(1,0), north=(0,1)
                    v = np.array(P.position[:2], dtype=float) - np.array(Q.position[:2], dtype=float)
                    return categorize_relation(v, np.array([0.0, 1.0], dtype=float), np.array([1.0, 0.0], dtype=float))

                # Try all ordered pairs among the triple to find a cardinal-aligned premise
                triple_objs = [A, B, C]
                premise_pair = None
                premise_dir = None
                remaining_obj = None
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            continue
                        P = triple_objs[i]
                        Q = triple_objs[j]
                        R = triple_objs[3 - i - j]
                        # ensure none are blacklisted (defensive)
                        if str(P.label).lower() in BLACKLIST or str(Q.label).lower() in BLACKLIST or str(R.label).lower() in BLACKLIST:
                            continue
                        pd = cardinal_dir_between(P, Q)
                        if pd is None:
                            continue
                        # compute the answer relation of P vs R using world axes
                        rel_PR = rel8_world(P, R)
                        # avoid trivial case where the question's answer equals the premise
                        if rel_PR == pd:
                            continue
                        premise_pair = (P, Q)
                        premise_dir = pd
                        remaining_obj = R
                        break
                    if premise_pair is not None:
                        break

                if premise_pair is None:
                    # couldn't find a suitable cardinal-aligned premise among the triple
                    continue

                P, Q = premise_pair
                R = remaining_obj

                rel_PQ = premise_dir
                rel_PR = rel8_world(P, R)

                # Build choices and answer (8-way)
                choices = ["North","South","East","West","NorthEast","NorthWest","SouthEast","SouthWest"]
                labels_ = ['A','B','C','D','E','F','G','H']
                try:
                    ans_index = choices.index(rel_PR)
                except ValueError:
                    continue
                answer_letter = labels_[ans_index]

                # Compose question text using the chosen premise pair
                labelP = display_name.get(P.id, P.label)
                labelQ = display_name.get(Q.id, Q.label)
                labelR = display_name.get(R.id, R.label)
                qtext = f"If {labelP} is {rel_PQ} of {labelQ}, where is {labelP} relative to {labelR}?"

                # meta assembly
                render_cfg = create_intrinsics()
                render_cfg['camtoworld'] = camtoworld_from_pos_target(pos, tgt).tolist()
                # compute per-object visibility metrics for the chosen P/Q/R
                vvP, voP = compute_visibility_metrics(pos, tgt, P, aabbs_all, K, WIDTH, HEIGHT)
                vvQ, voQ = compute_visibility_metrics(pos, tgt, Q, aabbs_all, K, WIDTH, HEIGHT)
                vvR, voR = compute_visibility_metrics(pos, tgt, R, aabbs_all, K, WIDTH, HEIGHT)

                item = {
                    'qtype': 'chain_position_reasoning',
                    'scene': str(scene_path),
                    'question': qtext,
                    'choices': choices,
                    'answer': answer_letter,
                    'meta': {
                        'camera_pos': pos.tolist(),
                        'camera_target': tgt.tolist(),
                        'render': render_cfg,
                        'north_xy': north_xy.tolist(),
                        'east_xy': east_xy.tolist(),
                        'objects': [
                                {'id': P.id, 'label': P.label, 'display_label': labelP, 'center': np.array(P.position).tolist(), 'visible_view_percent': vvP, 'visible_object_percent': voP},
                                {'id': Q.id, 'label': Q.label, 'display_label': labelQ, 'center': np.array(Q.position).tolist(), 'visible_view_percent': vvQ, 'visible_object_percent': voQ},
                                {'id': R.id, 'label': R.label, 'display_label': labelR, 'center': np.array(R.position).tolist(), 'visible_view_percent': vvR, 'visible_object_percent': voR},
                        ],
                        'relations': {
                            'P_vs_Q': rel_PQ,
                            'P_vs_R': rel_PR,
                        }
                    }
                }

                # Append and optionally write files
                items.append(item)
                per_scene_count += 1

                if out_dir is not None:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    idx = len(items) - 1
                    base = f"{scene_path.name}_item_{idx:04d}_chain_mca"
                    item_dir = out_dir / base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    with open(item_dir / 'meta.json', 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # Render one main view image (camera view)
                    try:
                        img = render_thumbnail_for_pose(scene_path, {'position': pos, 'target': tgt}, thumb_size=WIDTH)
                        imageio.imwrite(str(item_dir / 'view_image.png'), img)
                    except Exception as e:
                        print(f"[warn] render view failed: {e}")

                    # preview and view
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break

        if verbose:
            print(f"[scene] {scene_path.name}: generated {per_scene_count} items (total {len(items)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"[done] total_items={len(items)} -> {out_path}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate chain_position_reasoning_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + images)')
    parser.add_argument('--per_room_points', type=int, default=20)
    parser.add_argument('--min_dist', type=float, default=0.4)
    parser.add_argument('--max_dist', type=float, default=7)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    iterate_and_generate(scenes_root, out_path, out_dir,
                         per_room_points=args.per_room_points,
                         min_dist=args.min_dist, max_dist=args.max_dist,
                         max_items=args.max_items, max_items_per_scene=args.max_items_per_scene,
                         verbose=args.verbose)


if __name__ == '__main__':
    main()
