#!/usr/bin/env python3
"""
Generate `frame_frame_distance` items using sampler-style view sampling.

Behavior:
 - Sample legal start views using `generate_camera_positions` (sampler generator).
 - From a start view pick a visible object (prefer large projected area).
 - For candidate move distances (forward/back), move camera along forward vector and
   pick an end view that also sees the object and yields a meaningful displacement.
 - Produce a 4-way MCQ asking: "By how many meters has the camera moved from begin to end?"

Per-item outputs (when --out-dir provided):
 - meta.json
 - preview.png (composed preview using project's preview composer)
 - view.png (top-down view map)
 - A.png, B.png, C.png, D.png (thumbnails for each choice)

Run example:
  python -m Data_generation.sampler.question_generator.frame_frame_distance_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/frame_frame_dist.jsonl \
    --out-dir /data/.../tmp/frame_frame_dist_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer and rendering helpers (no lazy import)
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


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def is_instance_visible(pos: np.ndarray, tgt: np.ndarray, so: SceneObject, aabbs_all, K, width: int, height: int) -> bool:
    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, width, height, corner_threshold=4, aabbs_all=aabbs_all)
    if corners_visible < 4:
        return False
    camtworld = camtoworld_from_pos_target(pos, tgt)
    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, width, height, target_id=so.id, depth_mode='min', return_per_occluder=False)
    target_px = float(occ.get('target_area_px', 0.0))
    target_ratio = float(target_px) / float(width * height)
    return target_ratio > 0.01


def make_numeric_distractors_meters(center: float) -> List[float]:
    """Create 3 numeric distractors around center (meters)."""
    deltas = [0.2, -0.2, 0.5, -0.5, 1.0, -1.0, 0.1, -0.1]
    out: List[float] = []
    for d in deltas:
        v = center + d
        # no strong bounds; just avoid zero-length identical
        if abs(v - center) < 1e-6:
            continue
        out.append(round(v, 2))
        if len(out) >= 3:
            break
    # if not enough, fill with small increments
    cand = [round(center + i * 0.3, 2) for i in range(-4, 5) if abs(i) > 0]
    for c in cand:
        if c == round(center, 2) or c in out:
            continue
        out.append(c)
        if len(out) >= 3:
            break
    return out[:3]


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(202502)
    items: List[Dict[str, Any]] = []

    scenes = [p for p in sorted(scenes_root.iterdir()) if p.is_dir()]

    for scene_path in scenes:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")

        labels_file = scene_path / 'labels.json'
        if not labels_file.exists():
            if verbose:
                print(f"[warn] no labels.json in {scene_path}, skip")
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
            if verbose:
                print(f"[info] no valid objects in {scene_path}")
            continue

        aabbs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs + load_scene_wall_aabbs(str(scene_path))

        per_scene_count = 0

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

                start_pos = np.array(p['position'], dtype=float)
                start_tgt = np.array(p['target'], dtype=float)

                # build visible list
                visible: List[SceneObject] = []
                K = np.array(create_intrinsics()['K'], dtype=float)
                for oid, so in objects.items():
                    if so.label in BLACKLIST:
                        continue
                    if is_instance_visible(start_pos, start_tgt, so, aabbs_all, K, WIDTH, HEIGHT):
                        visible.append(so)

                if not visible:
                    if verbose:
                        print(f"[info] no visible objects at pose {pi} in scene {scene_path.name}")
                    continue

                # pick anchor object among visible (prefer largest projected area)
                area_scores = []
                for so in visible:
                    camtworld = camtoworld_from_pos_target(start_pos, start_tgt)
                    occ = occluded_area_on_image(start_pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    target_px = float(occ.get('target_area_px', 0.0))
                    area_scores.append((so, target_px))
                area_scores.sort(key=lambda x: -x[1])
                target_obj = area_scores[0][0]

                # Candidate move distances (meters). Positive = forward, negative = backward
                candidate_distances = [0.5, -0.6, 1.2, -0.1, 0.2, -0.2]
                rng.shuffle(candidate_distances)
                chosen_end = None
                chosen_distance = None

                # forward vector from camera
                fwd = unit(start_tgt - start_pos)

                for dist in candidate_distances:
                    end_pos = start_pos + fwd * float(dist)
                    end_tgt = start_tgt + (end_pos - start_pos)
                    if is_instance_visible(end_pos, end_tgt, target_obj, aabbs_all, K, WIDTH, HEIGHT):
                        # require minimal meaningful displacement
                        if abs(dist) < 0.05:
                            continue
                        chosen_end = {'position': end_pos.tolist(), 'target': end_tgt.tolist()}
                        chosen_distance = float(dist)
                        break

                if chosen_end is None:
                    if verbose:
                        print(f"[info] no suitable end found for scene={scene_path.name} obj={target_obj.id}")
                    continue

                end_pos = np.array(chosen_end['position'], dtype=float)
                end_tgt = np.array(chosen_end['target'], dtype=float)

                # prepare choices: correct distance (rounded to 2 decimals) and 3 distractors
                true_val = round(float(chosen_distance), 2)
                distractors = make_numeric_distractors_meters(true_val)
                opts = [true_val] + distractors[:3]
                rng.shuffle(opts)
                choices = [f"{v}m" for v in opts]
                labels_ABC = ['A', 'B', 'C', 'D']
                answer = labels_ABC[choices.index(f"{true_val}m")]

                # prepare render metadata
                camtworld_start = camtoworld_from_pos_target(start_pos, start_tgt)
                camtworld_end = camtoworld_from_pos_target(end_pos, end_tgt)
                render_start = create_intrinsics()
                render_start['camtoworld'] = camtworld_start.tolist()
                render_end = create_intrinsics()
                render_end['camtoworld'] = camtworld_end.tolist()

                question_text = "By how many meters has the camera moved from begin to end?"
                item = {
                    'qtype': 'frame_frame_distance',
                    'scene': str(scene_path),
                    'question': question_text,
                    'choices': choices,
                    'answer': answer,
                    'meta': {
                        'begin_pos': start_pos.tolist(),
                        'begin_target': start_tgt.tolist(),
                        'begin_render': render_start,
                        'current_pos': start_pos.tolist(),
                        'current_target': start_tgt.tolist(),
                        'current_render': render_start,
                        'camera_pos': start_pos.tolist(),
                        'camera_target': start_tgt.tolist(),
                        # end for the correct choice
                        'end_pos': end_pos.tolist(),
                        'end_target': end_tgt.tolist(),
                        'end_render': render_end,
                        'target_object_id': getattr(target_obj, 'id', None),
                        'target_label': getattr(target_obj, 'label', None),
                        'distance_m': float(chosen_distance),
                        'choices_map': choices,
                    }
                }

                items.append(item)
                per_scene_count += 1

                if out_dir is not None:
                    item_dir = out_dir / f"item_{len(items)-1:06d}"
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta.json
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render preview + view
                    preview_path = item_dir / 'preview.png'
                    view_path = item_dir / 'view.png'
                    # compose preview (uses WIDTH for thumb size)
                    compose_preview_for_item(item, scene_path, preview_path, thumb_size=WIDTH)
                    compose_view_map(item, scene_path, view_path, thumb_size=WIDTH)

                    # render per-choice thumbnails (A/B/C/D)
                    # For each choice, compute end pose: move by that distance along fwd
                    for i, ch in enumerate(choices):
                        val = float(ch.replace('m', ''))
                        pos_ch = start_pos + fwd * val
                        tgt_ch = start_tgt + (pos_ch - start_pos)
                        pose = {'position': pos_ch.tolist(), 'target': tgt_ch.tolist()}
                        img = render_thumbnail_for_pose(scene_path, pose, thumb_size=WIDTH)
                        out_img_path = item_dir / f"{labels_ABC[i]}.png"
                        imageio.imwrite(str(out_img_path), img)

                if len(items) >= max_items:
                    break

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate frame_frame_distance QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png + A/B/C/D.png)')
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
    print(f"Wrote {len(items)} frame_frame_distance items to {out_path}")


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Generate `frame_frame_distance_mca` items using sampler-style view sampling.

This generator samples legal start views (using the same sampler helper as
`sampler.batch_generate_views` / `generate_camera_positions`). For each start
view it samples a forward/backward displacement in [-2.0, 2.0] meters, computes
an end view by translating the camera along its forward axis by that distance,
and produces a multiple-choice question asking how far the agent moved forward
(in meters) from start to end.

Per-item outputs (when --out-dir provided):
  - meta.json (contains start/end poses and renders)
  - preview.png (composed preview using project's preview composer)
  - view.png (top-down view map)
  - start.png, end.png (rendered thumbnails of start/end views)

Run example:
  python -m Data_generation.sampler.question_generator.frame_frame_distance_mca \
    --scenes_root /data/liubinglin/jijiatong/ViewSuite/data \
    --out /tmp/frame_frame_distance.jsonl \
    --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/tmp/frame_frame_dist_items \
    --per_room_points 12 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import imageio
import numpy as np

# sampler helpers for sampling/rendering single poses
from ..generate_view import SceneObject, generate_camera_positions

# preview composer (no lazy import)
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


# ------------------------- small copied helpers -------------------------
from pathlib import Path as _Path

def load_room_polys(scene_path: str) -> List[np.ndarray]:
    p = _Path(scene_path) / 'structure.json'
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    rooms = data.get('rooms', [])
    polys: List[np.ndarray] = []
    for r in rooms:
        profile = r.get('profile')
        if not profile or len(profile) < 3:
            continue
        arr = np.array(profile, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        polys.append(arr[:, :2])
    return polys


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    if poly is None or len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    px = poly[:, 0]
    py = poly[:, 1]
    j = n - 1
    for i in range(n):
        xi, yi = px[i], py[i]
        xj, yj = px[j], py[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def is_pos_inside_scene(scene_path: str, pos: np.ndarray, tol: float = 0.01) -> bool:
    aabbs = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    q = np.array(pos, dtype=float)
    for b in aabbs:
        if (q[0] >= b.bmin[0] - tol and q[0] <= b.bmax[0] + tol and
            q[1] >= b.bmin[1] - tol and q[1] <= b.bmax[1] + tol and
            q[2] >= b.bmin[2] - tol and q[2] <= b.bmax[2] + tol):
            return True
    return False


# ------------------------- generation logic -------------------------
def generate_distance_item_from_view(view: Dict[str, Any], scene_path: str, rng: random.Random, max_attempts: int = 8, verbose: bool = False) -> Dict[str, Any] | None:
    """Given a sampled start view dict, try to create a valid distance QA item.

    We attempt up to `max_attempts` to sample a translation distance in [-2,2]
    that yields a legal end position and keeps the chosen target reasonably
    visible in both start and end views.
    """
    pos = np.array(view['pos'], dtype=float)
    tgt = np.array(view['tgt'], dtype=float)
    visible = view.get('visible', [])
    if not visible:
        if verbose:
            print(f"[skip] no visible objects for scene={scene_path}")
        return None

    # select target object (prefer original when present)
    target = None
    for v in visible:
        if getattr(v, 'id', None) == view.get('pose_meta', {}).get('target_obj_id'):
            target = v
            break
    if target is None:
        target = rng.choice(visible)

    aabbs_all = load_scene_aabbs(scene_path) + load_scene_wall_aabbs(scene_path)
    K = np.array(create_intrinsics()['K'], dtype=float)

    start_c2w = camtoworld_from_pos_target(pos, tgt)
    start_pos = start_c2w[:3, 3].astype(float)
    forward = np.array(start_c2w[:3, 2], dtype=float)

    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        # sample distance in meters in [-2.0, 2.0]
        d = rng.uniform(-2.0, 2.0)
        end_pos = start_pos + forward * float(d)
        end_target = tgt + forward * float(d)

        # legality checks: end position must not be inside geometry
        if is_pos_inside_scene(scene_path, end_pos):
            if verbose:
                print(f"[try] end_pos inside scene, d={d:.3f}")
            continue

        # compute occlusion / visible area for target at both start and end
        start_occ = occluded_area_on_image(start_pos, np.array(target.bbox_min), np.array(target.bbox_max), aabbs_all, K, start_c2w, WIDTH, HEIGHT, target_id=target.id, depth_mode='min', return_per_occluder=False)
        start_px = float(start_occ.get('target_area_px', 0.0))
        if start_px <= 0.0:
            if verbose:
                print(f"[try] target not visible at start, d={d:.3f}")
            continue

        end_c2w = camtoworld_from_pos_target(end_pos, end_target)
        end_occ = occluded_area_on_image(end_pos, np.array(target.bbox_min), np.array(target.bbox_max), aabbs_all, K, end_c2w, WIDTH, HEIGHT, target_id=target.id, depth_mode='min', return_per_occluder=False)
        end_px = float(end_occ.get('target_area_px', 0.0))
        if end_px <= 0.0:
            if verbose:
                print(f"[try] target not visible at end, d={d:.3f}")
            continue

        # both views valid; build item
        # prepare display-friendly choices: include true distance and three distractors
        true_val = float(d)
        # round to 2 decimals for display
        def fmt(v: float) -> str:
            return f"{v:.2f}m"

        distractors = set()
        # create distractors by adding/subtracting offsets
        while len(distractors) < 3:
            offset = rng.choice([0.15, 0.3, 0.5, 0.8, 1.2]) * rng.choice([-1, 1])
            cand = true_val + offset
            # clamp to [-2,2]
            cand = max(-2.0, min(2.0, cand))
            if abs(cand - true_val) < 1e-4:
                continue
            distractors.add(round(cand, 2))
        choices_vals = [round(true_val, 2)] + list(distractors)
        rng.shuffle(choices_vals)
        choices = [fmt(v) for v in choices_vals]
        correct_idx = choices_vals.index(round(true_val, 2))
        labels = ['A', 'B', 'C', 'D']
        answer_label = labels[correct_idx]

        # prepare render configs (make sure camtoworld matrices are lists)
        start_render = create_intrinsics()
        start_render['camtoworld'] = np.array(start_c2w).tolist()
        end_render = create_intrinsics()
        end_render['camtoworld'] = np.array(end_c2w).tolist()

        item = {
            'qtype': 'distance_mca',
            'scene': str(scene_path),
            'question': 'Given the start and end views, how far did the agent move forward (meters)?',
            'choices': choices,
            'answer': answer_label,
            'meta': {
                'start_pos': start_pos.tolist(),
                'start_target': tgt.tolist(),
                'end_pos': end_pos.tolist(),
                'end_target': end_target.tolist(),
                'forward_move': round(true_val, 3),
                'start_render': start_render,
                'end_render': end_render,
                'visibility': {
                    'start': {'visible': True, 'visible_px': start_px},
                    'end': {'visible': True, 'visible_px': end_px},
                },
                'target_object_id': getattr(target, 'id', None),
                'target_label': getattr(target, 'label', None),
            }
        }
        return item

    # if attempts exhausted
    if verbose:
        print(f"[skip] failed to find valid end view after {max_attempts} attempts for scene={scene_path}")
    return None


def iterate_and_generate(scenes_root: Path, out_path: Path, out_dir: Path | None,
                         per_room_points: int = 20, min_dist: float = 0.4, max_dist: float = 3.5,
                         max_items: int = 200, max_items_per_scene: int = 200, render: bool = True, verbose: bool = False) -> List[Dict[str, Any]]:
    rng = random.Random(20231106)
    items: List[Dict[str, Any]] = []

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
                    corners_visible = count_visible_corners_for_box(pos, tgt, K, so.bbox_min, so.bbox_max, WIDTH, HEIGHT, corner_threshold=4, aabbs_all=aabbs_all)
                    camtoworld = camtoworld_from_pos_target(pos, tgt)
                    occ = occluded_area_on_image(pos, np.array(so.bbox_min), np.array(so.bbox_max), aabbs_all, K, camtoworld, WIDTH, HEIGHT, target_id=so.id, depth_mode='min', return_per_occluder=False)
                    target_px = float(occ.get('target_area_px', 0.0))
                    target_ratio = float(target_px) / float(WIDTH * HEIGHT)
                    if target_ratio <= 0.01:
                        continue
                    visible.append(so)

                if not visible:
                    if verbose:
                        print(f"[skip] no visible objects after filtering for scene={scene_path.name} pose_index={pi}")
                    continue

                view = {'pos': pos.tolist(), 'tgt': tgt.tolist(), 'visible': visible, 'pose_meta': p}
                view['pose_meta']['target_obj_id'] = obj_id

                item = generate_distance_item_from_view(view, str(scene_path), rng, max_attempts=8, verbose=verbose)
                if item is None:
                    if verbose:
                        print(f"[skip] generate_distance_item_from_view returned None for scene={scene_path.name} pose_index={pi}")
                    continue

                # save per-item folder if requested
                if out_dir is not None:
                    scene_out = out_dir
                    scene_out.mkdir(parents=True, exist_ok=True)
                    idx = len(items)
                    fname_base = f"{scene_path.name}_item_{idx:04d}_frame_frame_distance_mca"
                    item_dir = scene_out / fname_base
                    item_dir.mkdir(parents=True, exist_ok=True)
                    # write meta
                    meta_path = item_dir / 'meta.json'
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(item, mf, indent=2, ensure_ascii=False)

                    # render start and end thumbnails and save start.png / end.png using WIDTH as thumb size
                    start_pos = np.array(item['meta']['start_pos'], dtype=float)
                    start_tgt = np.array(item['meta']['start_target'], dtype=float)
                    end_pos = np.array(item['meta']['end_pos'], dtype=float)
                    end_tgt = np.array(item['meta']['end_target'], dtype=float)

                    img_start = render_thumbnail_for_pose(scene_path, {'position': start_pos, 'target': start_tgt}, thumb_size=WIDTH)
                    img_end = render_thumbnail_for_pose(scene_path, {'position': end_pos, 'target': end_tgt}, thumb_size=WIDTH)
                    imageio.imwrite(str(item_dir / 'start.png'), img_start)
                    imageio.imwrite(str(item_dir / 'end.png'), img_end)

                    # compose preview and view map using preview utilities (thumb_size uses WIDTH)
                    compose_preview_for_item(item, scene_path, item_dir / 'preview.png', thumb_size=WIDTH)
                    compose_view_map(item, scene_path, item_dir / 'view.png', thumb_size=WIDTH)

                items.append(item)
                per_scene_count += 1

        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Generation summary: total_items={len(items)}")
    return items


def main():
    parser = argparse.ArgumentParser(description='Generate frame_frame_distance_mca QA items using sampler views.')
    parser.add_argument('--scenes_root', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json + preview.png + view.png + start.png/end.png)')
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
    print(f"Wrote {len(items)} frame_frame_distance_mca items to {out_path}")


if __name__ == '__main__':
    main()
