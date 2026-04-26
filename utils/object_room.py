#!/usr/bin/env python3
"""
Assign each object in a scene to a room (by testing object centroid against room polygons)

Saves JSON mapping to the utils folder and writes a preview PNG to out_local.

Usage:
  python -m view_suite.envs.active_spatial_intelligence.utils.object_room --scene ./0013_840910 --out-dir ./out_local/
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


def load_structure_rooms(struct_path: Path):
    if not struct_path.exists():
        return {}
    with open(struct_path, 'r') as f:
        data = json.load(f)
    rooms = data.get('rooms', [])
    # Each room may contain 'profile' (list of [x,y])
    polys = []
    for idx, r in enumerate(rooms):
        profile = r.get('profile')
        if not profile:
            continue
        arr = np.array(profile, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        polys.append({'id': f'room_{idx}', 'poly': arr})
    return polys


def load_labels(labels_path: Path):
    if not labels_path.exists():
        return []
    with open(labels_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception:
            # fallback: try to read as a sequence
            txt = f.read()
            data = []
    return data


def point_in_poly(xy, poly_arr):
    # poly_arr shape (N,2)
    path = MplPath(poly_arr)
    return path.contains_point((xy[0], xy[1]))


def assign_objects_to_rooms(labels, rooms_polys):
    """
    Returns two things:
      - detailed: list of per-object dicts with fields {ins_id, id, label, category, room}
      - mapping: legacy dict mapping str(ins_id) -> room_id or None
    """
    detailed = []
    mapping = {}
    # Prepare room paths for faster testing
    room_paths = [(rp['id'], MplPath(rp['poly'])) for rp in rooms_polys]
    for obj in labels:
        ins_id = obj.get('ins_id')
        if ins_id is None:
            continue
        bb = obj.get('bounding_box')
        if not bb:
            assigned = None
            centroid = None
        else:
            pts = np.array([[p['x'], p['y']] for p in bb], dtype=float)
            centroid = pts.mean(axis=0)
            assigned = None
            for rid, path in room_paths:
                if path.contains_point((centroid[0], centroid[1])):
                    assigned = rid
                    break

        label = obj.get('label')
        # category: prefer explicit 'category' key if present, otherwise fall back to label
        category = obj.get('category', label)

        entry = {
            'ins_id': str(ins_id),
            # 'id': str(ins_id),
            'label': label,
            # 'category': category,
            'room': assigned,
        }
        detailed.append(entry)
        mapping[str(ins_id)] = assigned

    return detailed, mapping


def save_mapping(mapping, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        # mapping may be a list (detailed) or dict (legacy)
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def save_preview(mapping: dict, labels, rooms_polys, out_png: Path):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot rooms
    for rp in rooms_polys:
        arr = rp['poly']
        poly = np.vstack([arr, arr[0]])
        ax.plot(poly[:, 0], poly[:, 1], '-', linewidth=1)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        ax.text(cx, cy, rp['id'], fontsize=6, ha='center', va='center')
    # collect points and colors
    # mapping may be legacy dict or detailed list
    if isinstance(mapping, list):
        # build legacy mapping dict for preview lookup
        legacy_map = {str(e['ins_id']): e.get('room') for e in mapping}
    else:
        legacy_map = mapping

    room_ids = sorted({v for v in legacy_map.values() if v is not None})
    id_to_idx = {rid: i for i, rid in enumerate(room_ids)}
    xs = []
    ys = []
    cs = []
    for obj in labels:
        ins_id = obj.get('ins_id')
        if ins_id is None:
            continue
        bb = obj.get('bounding_box')
        if not bb:
            continue
        pts = np.array([[p['x'], p['y']] for p in bb], dtype=float)
        centroid = pts.mean(axis=0)
        xs.append(centroid[0])
        ys.append(centroid[1])
        rid = legacy_map.get(str(ins_id))
        if rid is None:
            cs.append(-1)
        else:
            cs.append(id_to_idx[rid])
    if len(xs) > 0:
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        # colormap: assign gray for -1
        cmap = plt.get_cmap('tab20')
        # build colors array
        colors = []
        for v in cs:
            if v == -1:
                colors.append('#888888')
            else:
                colors.append(cmap(v % 20))
        ax.scatter(xs, ys, c=colors, s=8)
    ax.set_aspect('equal')
    ax.set_title('Object centroids colored by room assignment')
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Scene folder path')
    parser.add_argument('--out-dir', default='./out_local', help='Directory to save preview PNG')
    args = parser.parse_args()

    scene = Path(args.scene)
    struct_path = scene / 'structure.json'
    labels_path = scene / 'labels.json'

    rooms_polys = load_structure_rooms(struct_path)
    labels = load_labels(labels_path)
    if not labels:
        print(f'No labels found in {labels_path}')
    detailed, mapping = assign_objects_to_rooms(labels, rooms_polys)

    # Save into the utils directory (same folder as this script)
    utils_dir = Path(__file__).parent
    scene_name = scene.name
    out_json_scene = utils_dir / f'object_room_{scene_name}.json'
    out_json_generic = utils_dir / 'object_room.json'
    # Filter detailed entries: keep only id,label,room and drop entries with room == None
    filtered = []
    for e in detailed:
        if e.get('room') is None:
            continue
        filtered.append({'id': e.get('id'), 'label': e.get('label'), 'room': e.get('room')})

    # write filtered per-scene file (list of objects with id,label,room)
    save_mapping(filtered, out_json_scene)
    # write legacy generic mapping (dict ins_id -> room)
    save_mapping(mapping, out_json_generic)
    print(f'Wrote filtered mapping to {out_json_scene} and legacy mapping to {out_json_generic}')

    # Save preview
    out_png = Path(args.out_dir) / f'object_rooms_preview_{scene_name}.png'
    # preview can use the detailed list for coloring too
    save_preview(detailed, labels, rooms_polys, out_png)
    print(f'Saved preview to {out_png}')


if __name__ == '__main__':
    main()
