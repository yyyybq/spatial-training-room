#!/usr/bin/env python3
"""
Quick sanity checker for rooms in structure.json.

- Loads rooms from structure.json and extracts the 2D room polygon profiles
- Prints basic stats and first few room polygons
- Optionally saves a top-down plot (XY) of room polygons to a PNG

Usage:
  python -m view_suite.envs.active_spatial_intelligence.utils.rooms_debug --scene ./0013_840910 --out ./out_local/rooms_xy.png
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_rooms_raw(structure_path: Path):
    if not structure_path.exists():
        return []
    try:
        with open(structure_path, 'r') as f:
            data = json.load(f)
        return data.get('rooms', [])
    except Exception:
        return []


def rooms_to_polys(rooms_raw):
    polys = []
    idx = 0
    for r in rooms_raw:
        try:
            profile = r.get('profile', None)
            if not profile or len(profile) < 3:
                continue
            arr = np.array(profile, dtype=float)
            # Expect shape (N,2)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            polys.append({'id': f'room_{idx}', 'poly': arr})
            idx += 1
        except Exception:
            continue
    return polys


def save_topdown_plot(polys, out_path: Path):
    if len(polys) == 0:
        return False
    xs = []
    ys = []
    for p in polys:
        arr = p['poly']
        xs.extend(list(arr[:, 0]))
        ys.extend(list(arr[:, 1]))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.5
    fig, ax = plt.subplots(figsize=(8, 8))
    for p in polys:
        arr = p['poly']
        poly = np.vstack([arr, arr[0]])
        ax.plot(poly[:, 0], poly[:, 1], '-o', linewidth=1)
        cx, cy = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        ax.text(cx, cy, p['id'], fontsize=6, ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_xlim([xmin - pad, xmax + pad])
    ax.set_ylim([ymin - pad, ymax + pad])
    ax.set_title('Rooms top-down (XY)')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Scene folder (InteriorGS scene path)')
    parser.add_argument('--out', default='', help='Optional output PNG path for top-down plot')
    args = parser.parse_args()

    scene = Path(args.scene)
    struct_path = scene / 'structure.json'
    rooms_raw = load_rooms_raw(struct_path)
    polys = rooms_to_polys(rooms_raw)
    print(f"Loaded {len(polys)} room polygons from {struct_path}")
    for i, p in enumerate(polys[:12]):
        print(f"  {p['id']}: {p['poly'].shape[0]} vertices, centroid=({float(p['poly'][:,0].mean()):.3f},{float(p['poly'][:,1].mean()):.3f})")
    if args.out:
        ok = save_topdown_plot(polys, Path(args.out))
        if ok:
            print(f"Saved top-down rooms plot to {args.out}")


if __name__ == '__main__':
    main()
