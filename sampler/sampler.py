#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Room-based camera view sampler and renderer.

- Hardcode scene path
- Load room polygons
- Uniformly sample valid camera positions inside rooms
- Enforce min distance from walls and objects
- Fix height at 0.8 meters
- For each valid point, generate 8 horizontal forward directions
- Render each view using your existing render_thumbnail_for_pose
- Concatenate all rendered images into one big mosaic image

This code depends on your original modules:
    - load_room_polys
    - point_in_poly
    - is_pos_inside_scene
    - load_scene_aabbs
    - setup_camtoworld
    - render_thumbnail_for_pose

"""

import os
import numpy as np
import imageio
from pathlib import Path

from bench_generation.qa_batch_generator import (
    load_room_polys,
    point_in_poly,
    is_pos_inside_scene,
    load_scene_aabbs,
    setup_camtoworld,
)
from bench_generation.preview import render_thumbnail_for_pose

# =============================
# Hardcode scene path
# =============================
SCENE_ROOT = Path("/data/liubinglin/jijiatong/ViewSuite/data/0013_840910")  # TODO: change this


# =========================================================
# Utility: check distance from objects (AABBs)
# =========================================================
def is_too_close_to_objects(pos, aabbs, min_dist=0.4):
    """Return True if |pos - object center| < min_dist."""
    p = np.array(pos, dtype=float)
    for b in aabbs:
        center = 0.5 * (b.bmin + b.bmax)
        d = float(np.linalg.norm(p - center))
        if d < min_dist:
            return True
    return False


# =========================================================
# Uniform grid sampling inside polygon
# =========================================================
def sample_points_in_polygon(poly, num_points=10, height=0.8, min_dist_to_wall=0.15):
    """
    Uniformly sample 1~num_points points inside polygon (XY plane).
    """
    poly = np.array(poly)
    xs = poly[:, 0]
    ys = poly[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

    pts = []
    tries = 0
    max_tries = num_points * 50

    # bounding box rejection sampling
    while len(pts) < num_points and tries < max_tries:
        tries += 1
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        if point_in_poly(x, y, poly):
            # optional: keep margin from boundary by checking shrink polygon
            pts.append(np.array([x, y, height]))

    return pts


# =========================================================
# 8 horizontal directions
# =========================================================
def eight_forward_directions():
    dirs = []
    for k in range(8):
        ang = k * (np.pi / 4)  # 0°,45°,90°...
        forward = np.array([np.cos(ang), np.sin(ang), 0.0], dtype=float)
        dirs.append(forward)
    return dirs


# =========================================================
# Main camera pose generation for all rooms
# =========================================================
def generate_camera_poses(scene_root, per_room_points=5):
    polys = load_room_polys(str(scene_root))
    aabbs = load_scene_aabbs(str(scene_root))

    all_poses = []

    for room_idx, poly in enumerate(polys):
        # sample points inside this room
        pts = sample_points_in_polygon(
            poly,
            num_points=per_room_points,
            height=0.8
        )

        for pos in pts:
            # reject if too close to any object
            if is_pos_inside_scene(str(scene_root), pos):
                continue
            if is_too_close_to_objects(pos, aabbs, min_dist=0.5):
                continue

            # For each point, generate 8 directions
            for fwd in eight_forward_directions():
                tgt = pos + fwd
                pose = {"position": pos.copy(), "target": tgt.copy()}
                all_poses.append(pose)

    print(f"[info] Generated {len(all_poses)} camera poses")
    return all_poses


# =========================================================
# Render all views and make mosaic
# =========================================================
def render_and_mosaic(poses, scene_root, out_path, thumb=256, per_row=8):
    imgs = []
    for idx, pose in enumerate(poses):
        img = render_thumbnail_for_pose(
            scene_root,
            pose,
            thumb_size=thumb
        )
        imgs.append(img)

    # make mosaic: rows of per_row images
    H, W, C = imgs[0].shape
    num = len(imgs)
    rows = (num + per_row - 1) // per_row

    canvas = np.zeros((rows * H, per_row * W, C), dtype=np.uint8)
    for i, img in enumerate(imgs):
        r = i // per_row
        c = i % per_row
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = img

    imageio.imwrite(out_path, canvas)
    print(f"[save] mosaic image written to {out_path}")


# =========================================================
# Entry
# =========================================================
def main():
    poses = generate_camera_poses(
        SCENE_ROOT,
        per_room_points=2  # 1~10 random points per room
    )
    render_and_mosaic(
        poses,
        SCENE_ROOT,
        out_path="mosaic_views.png",
        thumb=256,
        per_row=8
    )


if __name__ == "__main__":
    main()
