#!/usr/bin/env python3
"""
Standalone generator for `distance_mca` QA items.

This module re-uses a number of helper utilities from the existing
`qa_batch_generator` module (same package) and provides a compact CLI that
accepts: --scene, --out, --out-dir, --max_items_per_view, --max_items,
--max_items_per_scene.

python -m Data_generation.bench_generation.object_camera_distance \
  --scene /data/liubinglin/jijiatong/ViewSuite/data \
  --out tmp.jsonl \
  --out-dir /tmp/distance_items \
  --max_items 50
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Reuse helpers from the existing module to avoid duplication.
from .batch_utils import (
    create_intrinsics,
    load_scene_aabbs,
    load_scene_wall_aabbs,
    camtoworld_from_pos_target,
    occluded_area_on_image,
    is_aabb_occluded,
    is_target_in_fov,
    count_visible_corners_for_box,
    BLACKLIST,
    WIDTH,
    HEIGHT,
    generate_distance_mca_from_view,
    ensure_position_legal,
)
from .camera_generation import SemanticCamera
from .preview import compose_preview_for_item
from .preview import compose_view_map


def iterate_and_generate(scene_root: Path, out_path: Path, out_dir: Path | None, max_items_per_view: int, max_items: int, max_items_per_scene: int, render: bool = True) -> List[Dict[str, Any]]:
    rng = random.Random(42)
    items: List[Dict[str, Any]] = []
    seen = set()

    if scene_root.is_dir():
        scene_paths = [p for p in sorted(scene_root.iterdir()) if p.is_dir()]
    else:
        scene_paths = [scene_root]

    distance_scales = [0.9]

    for scene_path in scene_paths:
        if len(items) >= max_items:
            break
        print(f"Processing scene: {scene_path}")
        sc = SemanticCamera(str(scene_path))
        aabbs_objs = load_scene_aabbs(str(scene_path))
        aabbs_all = aabbs_objs + load_scene_wall_aabbs(str(scene_path))
        presets = list(sc.presets.keys())

        per_scene_count = 0
        for obj in aabbs_objs:
            if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                break
            for preset in presets:
                for scale in distance_scales:
                    if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                        break
                    try:
                        cfg = sc.calculate_camera(obj.id, preset=preset, distance_scale=scale) if hasattr(sc, 'calculate_camera') else None
                    except Exception as e:
                        print(f"calculate_camera failed for {obj.id} {preset} {scale}: {e}")
                        cfg = None
                    if cfg is None:
                        continue
                    pos = np.array(cfg.camera_position, dtype=float)
                    tgt = np.array(cfg.target_position, dtype=float)

                    # basic legality check (reuse existing helper)
                    if not ensure_position_legal(str(scene_path), pos):
                        continue

                    # build visible list
                    visible = []
                    K = np.array(create_intrinsics()['K'], dtype=float)
                    width = WIDTH; height = HEIGHT
                    for b in aabbs_objs:
                        corners = count_visible_corners_for_box(pos, tgt, K, b.bmin, b.bmax, width, height)
                        if corners < 4:
                            continue
                        c2w = camtoworld_from_pos_target(pos, tgt)
                        K = np.array(create_intrinsics()['K'], dtype=float)
                        res = occluded_area_on_image(
                            ray_o=pos,
                            target_bmin=b.bmin,
                            target_bmax=b.bmax,
                            aabbs=aabbs_all,
                            K=K,
                            camtoworld=c2w,
                            width=width,
                            height=height,
                            target_id=getattr(b, 'id', None),
                            depth_mode="mean",
                        )
                        if res.get('occlusion_ratio_target', 1.0) > 0.4:
                            continue
                        if b.label in BLACKLIST:
                            continue
                        visible.append(b)

                    if not visible:
                        continue

                    view = {'preset': preset, 'pos': pos, 'tgt': tgt, 'visible': visible}
                    # use existing generator function
                    try:
                        it = generate_distance_mca_from_view(view, str(scene_path), seed=rng.getrandbits(32), rng=rng)
                    except Exception as e:
                        print(f"generate_distance_mca_from_view raised: {e}")
                        it = None
                    if it is None:
                        continue

                    # filter duplicates and limits
                    sig = (scene_path.name, it.get('qtype'), str(it.get('meta', {}).get('object_id')), tuple(round(float(x), 2) for x in (it.get('meta', {}).get('camera_pos') or [0,0,0])[:3]))
                    if sig in seen:
                        continue
                    seen.add(sig)

                    items.append(it)
                    per_scene_count += 1
                    # respect per-view
                    if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                        break
                # end scales
                if per_scene_count >= max_items_per_scene or len(items) >= max_items:
                    break
            # end presets
        # end objects
        print(f"Scene {scene_path.name}: generated {per_scene_count} items (accum total {len(items)})")

    # write JSONL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')

    # optional simple out-dir export: write meta.json per item
    if out_dir is not None:
        base = Path(out_dir)
        base.mkdir(parents=True, exist_ok=True)

        for idx, it in enumerate(items):
            scene_name = Path(it.get('scene') or '').name
            folder = base / f"{scene_name}_item_{idx:04d}_{it.get('qtype','unknown')}"
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder / 'meta.json', 'w', encoding='utf-8') as jf:
                json.dump(it, jf, indent=2, ensure_ascii=False)
            # render preview image if requested (exceptions allowed to propagate)
            if render:
                out_png = folder / 'preview.png'
                compose_preview_for_item(it, Path(it.get('scene') or scene_root), out_png)
                view_png = folder / 'view.png'
                compose_view_map(it, Path(it.get('scene') or scene_root), view_png)

    return items


def main():
    parser = argparse.ArgumentParser(description='Generate distance_mca QA items (extracted).')
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True, help='Output JSONL file path')
    parser.add_argument('--out-dir', required=False, help='Optional directory to write per-item folders (meta.json)')
    parser.add_argument('--max_items_per_view', type=int, default=10)
    parser.add_argument('--max_items', type=int, default=200)
    parser.add_argument('--max_items_per_scene', type=int, default=200)
    # rendering is enabled by default; use --no-render to disable if you really want to skip previews
    parser.add_argument('--no-render', action='store_false', dest='render', help='Disable preview rendering (default: render)')
    args = parser.parse_args()

    scene = Path(args.scene)
    out_path = Path(args.out)
    out_dir = Path(args.out_dir) if args.out_dir else None

    items = iterate_and_generate(scene, out_path, out_dir, args.max_items_per_view, args.max_items, args.max_items_per_scene, render=args.render)
    print(f"Wrote {len(items)} distance_mca items to {out_path}")


if __name__ == '__main__':
    main()
