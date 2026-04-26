#!/usr/bin/env python3
"""
Quick sanity checker for walls in structure.json.

- Loads walls from structure.json and converts them to AABBs using the same logic as occlusion.load_scene_wall_aabbs
- Prints basic stats and first few AABBs
- Optionally saves a top-down plot (XY) of wall rectangles to a PNG

Usage:
  python -m view_suite.envs.active_spatial_intelligence.utils.walls_debug --scene ./0013_840910 --out ./out_local/walls_xy.png
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
import json
from typing import List, Tuple


def load_walls_raw(structure_path: Path):
    if not structure_path.exists():
        return []
    try:
        with open(structure_path, 'r') as f:
            data = json.load(f)
        return data.get('walls', [])
    except Exception:
        return []


def walls_to_aabbs(walls_raw):
    aabbs = []
    idx = 0
    for w in walls_raw:
        try:
            loc = w.get('location', None)
            th = float(w.get('thickness', 0.2) or 0.2)
            h = float(w.get('height', 2.8) or 2.8)
            if not loc or len(loc) != 2:
                continue
            x1, y1 = float(loc[0][0]), float(loc[0][1])
            x2, y2 = float(loc[1][0]), float(loc[1][1])
            p1 = np.array([x1, y1], dtype=float)
            p2 = np.array([x2, y2], dtype=float)
            seg = p2 - p1
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                half = th * 0.5
                xs = [x1 - half, x1 + half]
                ys = [y1 - half, y1 + half]
                rect = np.array([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]]])
            else:
                dir_xy = seg / (seg_len + 1e-12)
                n = np.array([-dir_xy[1], dir_xy[0]], dtype=float)
                half = th * 0.5
                q1 = p1 + n * half
                q2 = p1 - n * half
                q3 = p2 + n * half
                q4 = p2 - n * half
                rect = np.vstack([q1, q3, q4, q2])  # roughly in order
            bmin = np.array([rect[:,0].min(), rect[:,1].min(), 0.0], dtype=float)
            bmax = np.array([rect[:,0].max(), rect[:,1].max(), h], dtype=float)
            aabbs.append({'id': f'wall_{idx}', 'label': 'wall', 'bmin': bmin, 'bmax': bmax, 'rect': rect})
            idx += 1
        except Exception:
            continue
    return aabbs


def save_topdown_plot(aabbs, out_path: Path):
    if len(aabbs) == 0:
        return False
    xs = []
    ys = []
    for a in aabbs:
        r = a['rect']
        xs.extend(list(r[:,0]))
        ys.extend(list(r[:,1]))
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad = 0.5
    fig, ax = plt.subplots(figsize=(6,6))
    for a in aabbs:
        r = a['rect']
        poly = np.vstack([r, r[0]])
        ax.plot(poly[:,0], poly[:,1], 'k-')
        cx, cy = float(r[:,0].mean()), float(r[:,1].mean())
        ax.text(cx, cy, a['id'], fontsize=6, ha='center', va='center')
    # try to compute convex hull of all rectangles and draw it
    all_pts = []
    for a in aabbs:
        r = a['rect']
        for (xx, yy) in r.tolist():
            all_pts.append((float(xx), float(yy)))
    if len(all_pts) >= 3:
        hull = convex_hull(np.array(all_pts, dtype=float))
        if hull is not None and len(hull) > 0:
            hull_poly = np.vstack([hull, hull[0]])
            ax.plot(hull_poly[:,0], hull_poly[:,1], color='red', linewidth=1.2, linestyle='--', label='walls hull')
            ax.fill(hull_poly[:,0], hull_poly[:,1], color=(1.0,0.8,0.8,0.15))
            try:
                ax.legend()
            except Exception:
                pass
    ax.set_aspect('equal')
    ax.set_xlim([xmin - pad, xmax + pad])
    ax.set_ylim([ymin - pad, ymax + pad])
    ax.set_title('Walls top-down (XY)')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True


def convex_hull(points: np.ndarray) -> np.ndarray | None:
    """Compute 2D convex hull with monotonic chain. Returns Nx2 array in CCW order."""
    if points is None or len(points) < 3:
        return None
    pts = np.array(points, dtype=float)
    # sort by x then y
    idx = np.lexsort((pts[:,1], pts[:,0]))
    pts = pts[idx]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    if len(hull) == 0:
        return None
    return np.array(hull, dtype=float)


def point_in_polygon(pt: Tuple[float,float], poly: np.ndarray) -> bool:
    """Return True if pt is inside poly (Nx2) using matplotlib Path."""
    if poly is None or len(poly) < 3:
        return False
    path = MplPath(poly)
    return bool(path.contains_point((pt[0], pt[1])))


def nearest_point_on_segments(pt: Tuple[float,float], poly: np.ndarray) -> Tuple[float,float]:
    """Find nearest point to pt on polygon edges (including vertices). Returns (x,y)."""
    if poly is None or len(poly) == 0:
        return pt
    x0, y0 = float(pt[0]), float(pt[1])
    best = (x0, y0)
    best_d2 = float('inf')
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1)%n]
        # project point onto segment
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            projx, projy = x1, y1
        else:
            t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx*dx + dy*dy)
            t = max(0.0, min(1.0, t))
            projx = x1 + t * dx
            projy = y1 + t * dy
        d2 = (projx - x0)**2 + (projy - y0)**2
        if d2 < best_d2:
            best_d2 = d2
            best = (projx, projy)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help='Scene folder (InteriorGS scene path)')
    parser.add_argument('--out', default='', help='Optional output PNG path for top-down plot')
    parser.add_argument('--range-out', default='', help='Optional output JSON path to save computed hull and range')
    parser.add_argument('--check-pos', nargs=3, type=float, help='Check if the given x y z position is inside the walls hull/bounds')
    parser.add_argument('--clamp-pos', nargs=3, type=float, help='Clamp given x y z position to the computed bounds/hull (prints clamped position)')
    args = parser.parse_args()

    scene = Path(args.scene)
    struct_path = scene / 'structure.json'
    walls_raw = load_walls_raw(struct_path)
    aabbs = walls_to_aabbs(walls_raw)
    print(f"Loaded {len(aabbs)} wall segments from {struct_path}")
    for i, a in enumerate(aabbs[:8]):
        bmin = a['bmin']
        bmax = a['bmax']
        print(f"  {a['id']}: bmin={bmin.tolist()} bmax={bmax.tolist()}")
    # compute hull and bounding range
    all_pts = []
    zmax = 0.0
    for a in aabbs:
        r = a['rect']
        for (xx, yy) in r.tolist():
            all_pts.append((float(xx), float(yy)))
        zmax = max(zmax, float(a['bmax'][2]))

    hull = None
    bbox = None
    if len(all_pts) >= 3:
        hull = convex_hull(np.array(all_pts, dtype=float))
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        bbox = (xmin, ymin, xmax, ymax)
        print(f"Computed hull with {len(hull)} vertices; bbox=({xmin:.3f},{ymin:.3f})-({xmax:.3f},{ymax:.3f}); zmax={zmax:.3f}")
    else:
        print('Not enough wall geometry to compute hull')

    if args.out:
        ok = save_topdown_plot(aabbs, Path(args.out))
        if ok:
            print(f"Saved top-down plot to {args.out}")

    # save range JSON if requested
    if args.range_out and hull is not None:
        outd = {
            'hull': hull.tolist(),
            'bbox': {'xmin': float(bbox[0]), 'ymin': float(bbox[1]), 'xmax': float(bbox[2]), 'ymax': float(bbox[3])},
            'zmin': 0.0,
            'zmax': float(zmax)
        }
        try:
            Path(args.range_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.range_out, 'w', encoding='utf-8') as fo:
                json.dump(outd, fo, indent=2)
            print(f"Saved range JSON to {args.range_out}")
        except Exception as e:
            print('Failed to write range JSON:', e)

    # helper: check or clamp position
    if args.check_pos:
        x, y, z = args.check_pos
        inside = False
        if hull is not None:
            inside = point_in_polygon((x, y), hull)
        in_z = (0.0 <= z <= zmax) if z is not None else True
        print(f"Position ({x},{y},{z}) inside hull: {inside}, z in range: {in_z}")

    if args.clamp_pos:
        x, y, z = args.clamp_pos
        if bbox is not None:
            # clamp XY to bbox first
            cx = min(max(x, bbox[0]), bbox[2])
            cy = min(max(y, bbox[1]), bbox[3])
        else:
            cx, cy = x, y
        cz = min(max(z, 0.0), zmax)
        # if original point outside hull, project to nearest edge point
        if hull is not None and not point_in_polygon((x, y), hull):
            px, py = nearest_point_on_segments((x, y), hull)
            # ensure projected point is inside bbox too
            px = min(max(px, bbox[0]), bbox[2])
            py = min(max(py, bbox[1]), bbox[3])
            cx, cy = px, py
        print(f"Clamped position -> ({cx:.6f}, {cy:.6f}, {cz:.6f})")


if __name__ == '__main__':
    main()
