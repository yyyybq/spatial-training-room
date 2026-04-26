#!/usr/bin/env python3
"""Visualize a selected pair/group of objects in a scene.

Creates two images:
- top-down room map with object bboxes and labels (targets colored)
- schematic camera view (no rendering): projected object bbox polygons on image plane

Usage (example):
  python -m Data_generation.active_data.view_pair \
    --data-dir /data/liubinglin/jijiatong/ViewSuite/data \
    --scene 0013_840910 \
    --pairs /tmp/choose_group_test2.json \
    --pair-index 0 \
    --out-dir /tmp/view_pair_out

Notes:
- This script does not perform full rendering; it draws AABB boxes.
- It attempts to sample a camera pose inside the room that contains the targets
  and checks simple occlusion+FOV using the utilities in Data_generation.utils.occlusion.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon


def load_labels(labels_path: Path):
    if not labels_path.exists():
        return []
    with open(labels_path, 'r') as f:
        return json.load(f)


def load_rooms_from_structure(struct_path: Path):
    if not struct_path.exists():
        return []
    with open(struct_path, 'r') as f:
        data = json.load(f)
    rooms = data.get('rooms', [])
    polys = []
    for idx, r in enumerate(rooms):
        profile = r.get('profile')
        if not profile or len(profile) < 3:
            continue
        arr = np.array(profile, dtype=float)
        polys.append(arr)
    return polys


def point_in_poly(x: float, y: float, poly: np.ndarray) -> bool:
    # ray casting algorithm for point-in-polygon (2D)
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def scene_aabbs_from_labels(labels):
    objs = []
    for obj in labels:
        if 'bounding_box' not in obj:
            continue
        verts = obj['bounding_box']
        xs = [p['x'] for p in verts]
        ys = [p['y'] for p in verts]
        zs = [p['z'] for p in verts]
        bmin = np.array([min(xs), min(ys), min(zs)], dtype=float)
        bmax = np.array([max(xs), max(ys), max(zs)], dtype=float)
        center = 0.5 * (bmin + bmax)
        objs.append({
            'ins_id': str(obj.get('ins_id')),
            'label': obj.get('label', ''),
            'bmin': bmin,
            'bmax': bmax,
            'center': center,
        })
    return objs


def sample_points_in_room(poly, count=40, min_height=0.6, max_height=1.2):
    # copy of sampling logic: grid inside polygon
    poly = np.array(poly)
    xs, ys = poly[:, 0], poly[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    aspect = width / height
    nx = max(1, int(round(math.sqrt(count * aspect))))
    ny = max(1, int(math.ceil(count / float(nx))))
    xs_lin = np.linspace(xmin + 1e-6, xmax - 1e-6, nx)
    ys_lin = np.linspace(ymin + 1e-6, ymax - 1e-6, ny)
    gx, gy = np.meshgrid(xs_lin, ys_lin)
    grid_pts = np.vstack([gx.ravel(), gy.ravel()]).T
    inside_pts = []
    for (x, y) in grid_pts:
        try:
            if point_in_poly(float(x), float(y), poly):
                inside_pts.append((float(x), float(y)))
        except Exception:
            continue
    if len(inside_pts) >= count:
        indices = np.linspace(0, len(inside_pts) - 1, num=count, dtype=int)
        selected = [inside_pts[i] for i in indices]
    else:
        selected = inside_pts.copy()
        tries = 0
        while len(selected) < count and tries < count * 60:
            tries += 1
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            try:
                if not point_in_poly(float(x), float(y), poly):
                    continue
            except Exception:
                continue
            selected.append((float(x), float(y)))
    z_mid = float(0.5 * (min_height + max_height))
    pts3 = [np.array([x, y, z_mid], dtype=float) for (x, y) in selected]
    return pts3


def draw_topdown_map(scene_path: Path, objs, rooms_polys, target_ids, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 8))
    # plot rooms
    for poly in rooms_polys:
        arr = poly
        poly_closed = np.vstack([arr, arr[0]])
        ax.plot(poly_closed[:, 0], poly_closed[:, 1], '-', color='#666666', linewidth=1)
        ax.fill(poly_closed[:, 0], poly_closed[:, 1], alpha=0.03, color='#888888')
    # plot object bboxes and centers
    for o in objs:
        bmin = o['bmin']
        bmax = o['bmax']
        w = bmax[0] - bmin[0]
        h = bmax[1] - bmin[1]
        x = bmin[0]
        y = bmin[1]
        cid = str(o['ins_id'])
        if cid in target_ids:
            color = '#d62728'
        else:
            color = '#999999'
        rect = Rectangle((x, y), w, h, facecolor='none', edgecolor=color, linewidth=1.6)
        ax.add_patch(rect)
        cx, cy = float(o['center'][0]), float(o['center'][1])
        ax.scatter([cx], [cy], c=color, s=18)
        ax.text(cx + 0.02, cy + 0.02, f"{o['label']}\n{cid}", fontsize=7, color=color)
    ax.set_aspect('equal')
    ax.set_title(f"Top-down: {scene_path.name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def project_and_draw_view(scene_path: Path, cam_pos, cam_target, objs, target_ids, out_path: Path, width=400, height=400):
    # lazy import occlusion utilities (use existing projection/occlusion helpers)
    try:
        from Data_generation.utils import occlusion as occl
    except Exception:
        # fallback to relative import
        from ..utils import occlusion as occl  # type: ignore

    K = np.array([[width * 0.4, 0, width / 2.0], [0, width * 0.4, height / 2.0], [0, 0, 1]], dtype=float)
    camtoworld = occl.camtoworld_from_pos_target(cam_pos, cam_target)
    view = np.linalg.inv(camtoworld)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_title('Schematic camera view (bbox projections)')

    for o in objs:
        corners3d = occl.aabb_corners(o['bmin'], o['bmax'])
        uvs = []
        zs = []
        for c in corners3d:
            pc = occl.world_to_camera(view, c)
            try:
                proj = occl.project_point(K, pc)
            except Exception:
                continue
            u, v, z = proj
            uvs.append((u, v))
            zs.append(z)
        if len(uvs) < 3:
            continue
        poly = Polygon(uvs, closed=True)
        cid = str(o['ins_id'])
        if cid in target_ids:
            poly.set_edgecolor('#1f77b4')
            poly.set_facecolor('#1f77b433')
            ax.add_patch(poly)
            ax.text(np.mean([u for u, v in uvs]), np.mean([v for u, v in uvs]), f"{o['label']}\n{cid}", color='#1f77b4', fontsize=8)
        else:
            poly.set_edgecolor('#666666')
            poly.set_facecolor('#cccccc22')
            ax.add_patch(poly)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180)
    plt.close(fig)


def find_room_containing_all(objs, rooms_polys):
    # returns the first room poly that contains all object centers
    for poly in rooms_polys:
        ok = True
        for o in objs:
            c = o['center']
            if not point_in_poly(float(c[0]), float(c[1]), poly):
                ok = False
                break
        if ok:
            return poly
    return None


def pick_camera_for_group(scene_path: Path, group_objs, rooms_polys, all_objs, tries_per_room=60):
    # attempt to find a camera pos inside the room that sees all target boxes
    # use occlusion and fov checks from occlusion.py
    try:
        from Data_generation.utils import occlusion as occl
    except Exception:
        from ..utils import occlusion as occl  # type: ignore

    # choose room
    room = find_room_containing_all(group_objs, rooms_polys)
    if room is None:
        # fallback: use room of first object (approx)
        first = group_objs[0]
        for poly in rooms_polys:
            if point_in_poly(float(first['center'][0]), float(first['center'][1]), poly):
                room = poly
                break
    if room is None:
        return None, None

    # sample candidates
    candidates = sample_points_in_room(room, count=tries_per_room, min_height=0.8, max_height=1.2)
    aabbs = occl.load_scene_aabbs(str(scene_path))

    for pos in candidates:
        # skip if inside any object
        inside_any = False
        for b in aabbs:
            if (pos[0] >= b.bmin[0] - 0.2 and pos[0] <= b.bmax[0] + 0.2 and
                    pos[1] >= b.bmin[1] - 0.2 and pos[1] <= b.bmax[1] + 0.2 and
                    pos[2] >= b.bmin[2] - 0.2 and pos[2] <= b.bmax[2] + 0.2):
                inside_any = True
                break
        if inside_any:
            continue

        # point camera at group centroid
        centroid = np.mean([o['center'] for o in group_objs], axis=0)
        target = centroid
        camtoworld = occl.camtoworld_from_pos_target(pos, target)

        # check FOV and occlusion for all targets
        all_ok = True
        for o in group_objs:
            in_fov = occl.is_target_in_fov(K=np.array([[400*0.4,0,200],[0,400*0.4,200],[0,0,1]]), camtoworld=camtoworld, target_bmin=o['bmin'], target_bmax=o['bmax'], width=400, height=400, require_center=False)
            if not in_fov:
                all_ok = False
                break
            occ = occl.is_box_occluded_by_any(pos, o['bmin'], o['bmax'], aabbs, target_id=o['ins_id'])
            if occ:
                all_ok = False
                break
        if all_ok:
            return pos, target
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--pairs', required=True, help='JSON file with generated pairs/groups')
    parser.add_argument('--pair-index', type=int, default=0)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    scene = args.scene
    scene_path = data_dir / scene
    labels_path = scene_path / 'labels.json'
    struct_path = scene_path / 'structure.json'

    labels = load_labels(labels_path)
    rooms_polys = load_rooms_from_structure(struct_path)
    objs = scene_aabbs_from_labels(labels)
    objs_by_id = {str(o['ins_id']): o for o in objs}

    with open(args.pairs, 'r') as f:
        pairs = json.load(f)

    if args.pair_index < 0 or args.pair_index >= len(pairs):
        raise IndexError('pair-index out of range')

    group = pairs[args.pair_index]
    # group may be list of dicts {id,label} or single dict
    if isinstance(group, dict):
        group_list = [group]
    else:
        group_list = group

    target_ids = [str(item.get('id') or item.get('ins_id') or item.get('insid') or item.get('ins')) for item in group_list]
    group_objs = []
    for tid in target_ids:
        if tid in objs_by_id:
            group_objs.append(objs_by_id[tid])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    topdown_out = out_dir / f"{scene}_pair{args.pair_index}_topdown.png"
    view_out = out_dir / f"{scene}_pair{args.pair_index}_view.png"

    draw_topdown_map(scene_path, objs, rooms_polys, set(target_ids), topdown_out)

    cam_pos, cam_target = pick_camera_for_group(scene_path, group_objs, rooms_polys)
    if cam_pos is None:
        # still save a placeholder view (empty) and exit
        print('[warn] No camera pose found that sees all targets; saved top-down only.')
    else:
        project_and_draw_view(scene_path, cam_pos, cam_target, objs, set(target_ids), view_out)
        print(f'[ok] Saved schematic view to {view_out}')

    print(f'[ok] Saved top-down map to {topdown_out}')


if __name__ == '__main__':
    main()
