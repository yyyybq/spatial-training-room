#!/usr/bin/env python3
"""
Visualize rooms and their assigned objects (object_room mapping).

Draws each room polygon with a distinct color and plots object centroids with the same color.

Usage:
  python -m view_suite.envs.active_spatial_intelligence.utils.object_room_visual \
      --scene ./0013_840910 \
      --object-room view_suite/envs/active_spatial_intelligence/utils/object_room_0013_840910.json \
      --out ./out_local/object_room_visual_0013_840910.png
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


def load_rooms(struct_path: Path):
    if not struct_path.exists():
        return []
    with open(struct_path, 'r') as f:
        data = json.load(f)
    rooms = data.get('rooms', [])
    polys = []
    idx = 0
    for r in rooms:
        profile = r.get('profile')
        if not profile or len(profile) < 3:
            idx += 1
            continue
        arr = np.array(profile, dtype=float)
        polys.append({'id': f'room_{idx}', 'poly': arr})
        idx += 1
    return polys


def load_labels(labels_path: Path):
    if not labels_path.exists():
        return []
    with open(labels_path, 'r') as f:
        return json.load(f)


def load_object_room(path: Path):
    if not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def plot_rooms_objects(rooms_polys, labels, mapping, out_png: Path, title=None):
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Build color map for rooms
    room_ids = [r['id'] for r in rooms_polys]
    cmap = plt.get_cmap('tab20')
    room_to_color = {rid: cmap(i % 20) for i, rid in enumerate(room_ids)}

    # Plot room polygons (edge colored)
    for rp in rooms_polys:
        arr = rp['poly']
        poly = np.vstack([arr, arr[0]])
        ax.plot(poly[:, 0], poly[:, 1], '-', linewidth=2, color=room_to_color[rp['id']])
        # Slight translucent fill to help visibility
        ax.fill(poly[:, 0], poly[:, 1], facecolor=room_to_color[rp['id']], alpha=0.08)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        ax.text(cx, cy, rp['id'], fontsize=8, ha='center', va='center')

    # Plot object centroids colored by assigned room
    xs = []
    ys = []
    cols = []
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
        rid = mapping.get(str(ins_id))
        if rid is None:
            cols.append('#444444')
        else:
            cols.append(room_to_color.get(rid, '#444444'))

    if xs:
        ax.scatter(xs, ys, c=cols, s=18, edgecolors='k', linewidths=0.3)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Scene folder path')
    parser.add_argument('--object-room', required=True, help='Path to object_room JSON mapping')
    parser.add_argument('--out', required=True, help='Output PNG path')
    args = parser.parse_args()

    scene = Path(args.scene)
    struct_path = scene / 'structure.json'
    labels_path = scene / 'labels.json'

    rooms_polys = load_rooms(struct_path)
    labels = load_labels(labels_path)
    mapping = load_object_room(Path(args.object_room))

    title = f'Objects colored by room — {scene.name}'
    plot_rooms_objects(rooms_polys, labels, mapping, Path(args.out), title=title)
    print(f'Saved visualization to {args.out}')


if __name__ == '__main__':
    main()
