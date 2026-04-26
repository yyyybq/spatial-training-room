#!/usr/bin/env python3
"""Generate diagnostic images comparing occlusion computations.

Creates images with:
 - left: 3D schematic of camera, target, AABBs and walls
 - right-top: simulated render (projected filled polygons by depth)
 - right-bottom: ranking of occlusion_ratio_image and colored patches for visible objects with their occlusion_ratio_target

Usage: python3 -m tools.generate_occlusion_checks --scene ./data/0013_840910 --n_views 1 --outdir ./tmp/occlusion_checks_debug
"""
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
import numpy as np

from view_suite.envs.active_spatial_intelligence.data_gen.camera_generation import SemanticCamera
from view_suite.envs.active_spatial_intelligence.utils import occlusion

# Try to import camera_test renderer utilities (optional)
try:
    from camera_test import setup_camera as ct_setup_camera, prepare_gaussian_data as ct_prepare_gaussian_data, create_camera_intrinsics as ct_create_camera_intrinsics, render_gaussians as ct_render_gaussians
    from ply_gaussian_loader import PLYGaussianLoader
    import torch as _torch
    CAMERA_TEST_AVAILABLE = True
except Exception:
    ct_setup_camera = None
    ct_prepare_gaussian_data = None
    ct_create_camera_intrinsics = None
    ct_render_gaussians = None
    PLYGaussianLoader = None
    _torch = None
    CAMERA_TEST_AVAILABLE = False


def pick_random_views(camera: SemanticCamera, n_views: int = 6) -> List[dict]:
    objs = list(camera.objects.keys())
    presets = list(camera.presets.keys())
    views = []
    for _ in range(n_views):
        obj_id = random.choice(objs)
        preset = random.choice(presets)
        # small random jitter in distance/height
        ds = 0.9 + random.random() * 0.4
        ho = (random.random() - 0.5) * 0.3
        cfg = camera.calculate_camera(obj_id, preset, distance_scale=ds, height_offset=ho)
        if cfg is None:
            continue
        views.append({'cfg': cfg, 'obj_id': obj_id, 'preset': preset})
    return views


def draw_aabb_wire(ax3d, bmin, bmax, color='gray', alpha=0.7, linewidth=0.6):
    # draw wireframe box by its 8 corners and 12 edges
    corners = occlusion.aabb_corners(bmin, bmax)
    # edges as pairs of indices
    E = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    for i,j in E:
        xs = [corners[i,0], corners[j,0]]
        ys = [corners[i,1], corners[j,1]]
        zs = [corners[i,2], corners[j,2]]
        ax3d.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=linewidth)


def project_and_paint(ax, cam_pos, tgt_pos, aabbs, K, width, height, depth_mode='min', camtoworld=None, origin_upper: bool = False, visible_only: bool = False, color_map: Dict[str, tuple] = None):
    """
    Project AABBs and paint them into `ax`.
    If `visible_only` is True, perform a pixel-level depth test to determine whether
    a polygon has any visible pixels; only those polygons with >=1 visible pixel
    will be included in the final image.
    Returns id->color mapping used for visible polygons.
    """
    items = []
    if camtoworld is None:
        camtoworld = occlusion.camtoworld_from_pos_target(cam_pos, tgt_pos)
    for b in aabbs:
        poly, zs = occlusion._project_box_polygon(K, camtoworld, b.bmin, b.bmax, width, height)
        if len(poly) < 3:
            continue
        if depth_mode == 'min':
            depth = min(zs) if zs else float('inf')
        else:
            depth = float(np.mean(zs)) if zs else float('inf')
        items.append({'id': b.id, 'label': b.label, 'poly': np.array(poly), 'depth': float(depth)})

    # If not doing pixel-level visibility, fall back to painter's algorithm
    if not visible_only:
        # sort far to near
        items_sorted = sorted(items, key=lambda x: x['depth'], reverse=True)
        cmap = plt.get_cmap('tab20')
        id_to_color = {}
        for i, it in enumerate(items_sorted):
            col = cmap(i % 20)
            id_to_color[it['id']] = col
            poly = np.array(it['poly'])
            patch = patches.Polygon(poly, closed=True, facecolor=col, edgecolor='k', alpha=0.65)
            ax.add_patch(patch)
        ax.set_xlim(0, width)
        # If origin_upper is True, image coordinates have v increasing downwards (origin at top)
        # so invert the y-axis to match that convention (use height..0). Otherwise use 0..height.
        if origin_upper:
            ax.set_ylim(height, 0)
        else:
            ax.set_ylim(0, height)
        ax.axis('off')
        return id_to_color

    # ------------ visible_only == True: pixel-level z-test --------------
    depth_buf = np.full((height, width), np.inf, dtype=float)
    id_buf = -np.ones((height, width), dtype=int)

    cmap = plt.get_cmap('tab20')
    id_to_color = {}
    id_to_visible = {}
    # if a color_map is provided, use it for consistent coloring across figures
    provided_color_map = color_map if isinstance(color_map, dict) else None

    # iterate polygons and update depth buffer where polygon is closer
    for i, it in enumerate(items):
        pid = it['id']
        poly = it['poly']
        if poly.shape[0] < 3:
            id_to_visible[pid] = 0
            continue
        poly_depth = it['depth']

        xs = poly[:, 0]
        ys = poly[:, 1]
        x0 = max(0, int(np.floor(np.min(xs))))
        x1 = min(width - 1, int(np.ceil(np.max(xs))))
        y0 = max(0, int(np.floor(np.min(ys))))
        y1 = min(height - 1, int(np.ceil(np.max(ys))))
        if x1 < x0 or y1 < y0:
            id_to_visible[pid] = 0
            continue

        gx = np.arange(x0, x1 + 1)
        gy = np.arange(y0, y1 + 1)
        PX, PY = np.meshgrid(gx + 0.5, gy + 0.5)
        pts = np.vstack((PX.ravel(), PY.ravel())).T

        path = MplPath(poly)
        mask = path.contains_points(pts)
        if not np.any(mask):
            id_to_visible[pid] = 0
            continue

        mask_inds = np.nonzero(mask)[0]
        xi = pts[mask_inds, 0].astype(int)
        yi = pts[mask_inds, 1].astype(int)

        updated = 0
        for xx, yy in zip(xi, yi):
            if poly_depth < depth_buf[yy, xx] - 1e-9:
                depth_buf[yy, xx] = poly_depth
                id_buf[yy, xx] = i
                updated += 1

        id_to_visible[pid] = updated
        if provided_color_map and pid in provided_color_map:
            id_to_color[pid] = provided_color_map[pid]
        else:
            id_to_color[pid] = cmap(i % 20)

    # build image from id_buf
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    for it_idx, it in enumerate(items):
        pid = it['id']
        mask_pixels = (id_buf == it_idx)
        if not np.any(mask_pixels):
            continue
        col = id_to_color.get(pid, (0.7, 0.7, 0.7, 1.0))
        rgb = np.array(col[:3]) * 255.0
        img[mask_pixels] = rgb.astype(np.uint8)

    origin = 'upper' if origin_upper else 'lower'
    ax.imshow(img, origin=origin)
    for i, it in enumerate(items):
        pid = it['id']
        if id_to_visible.get(pid, 0) <= 0:
            continue
        poly = it['poly']
        patch = patches.Polygon(poly, closed=True, facecolor='none', edgecolor='k', linewidth=0.5)
        ax.add_patch(patch)

    ax.set_xlim(0, width)
    if origin_upper:
        ax.set_ylim(height, 0)
    else:
        ax.set_ylim(0, height)
    ax.axis('off')
    items_ids = [it['id'] for it in items]
    return id_to_color, id_to_visible, id_buf, items_ids


def run_scene(scene_path: str, n_views: int = 6, outdir: str = './tmp/occlusion_checks'):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cam = SemanticCamera(scene_path)
    if len(cam.objects) == 0:
        print('No objects in scene')
        return
    aabbs_objs = occlusion.load_scene_aabbs(scene_path)
    aabbs_walls = occlusion.load_scene_wall_aabbs(scene_path)
    aabbs_all = aabbs_objs + aabbs_walls

    views = pick_random_views(cam, n_views=n_views)
    width, height = 400, 400
    focal = float(width * 0.4)
    K = np.array([[focal, 0.0, width/2.0], [0.0, focal, height/2.0], [0.0, 0.0, 1.0]], dtype=float)

    # attempt to prepare camera_test renderer data (optional)
    renderer_prepared = False
    means = quats = scales = opacities = colors = sh_degree = None
    device = None
    if CAMERA_TEST_AVAILABLE:
        try:
            ply_path = Path(scene_path) / '3dgs_compressed.ply'
            if ply_path.exists():
                loader = PLYGaussianLoader()
                gs_data = loader.load_ply(str(ply_path))
                device = 'cuda' if _torch and _torch.cuda.is_available() else 'cpu'
                means, quats, scales, opacities, colors, sh_degree = ct_prepare_gaussian_data(gs_data, device, use_sh=True)
                renderer_prepared = True
                print('camera_test renderer prepared')
            else:
                print('camera_test PLY not found; real renderer unavailable for this scene')
        except Exception as e:
            print('camera_test prepare failed:', e)

    for idx, vv in enumerate(views):
        cfg = vv['cfg']
        cam_pos = np.array(cfg.camera_position, dtype=float)
        tgt_pos = np.array(cfg.target_position, dtype=float)
        # compute a unified camtoworld that matches camera_test convention when available
        if CAMERA_TEST_AVAILABLE and ct_setup_camera is not None:
            try:
                camtoworld = ct_setup_camera(cam_pos, tgt_pos, np.array([0.0, 0.0, -1.0]))
            except Exception:
                camtoworld = occlusion.camtoworld_from_pos_target(cam_pos, tgt_pos)
        else:
            camtoworld = occlusion.camtoworld_from_pos_target(cam_pos, tgt_pos)

        # compute occlusion per object using the same camtoworld
        results = []
        for b in aabbs_all:
            res = occlusion.occluded_area_on_image(cam_pos, b.bmin, b.bmax, aabbs_all, K, camtoworld, width, height, target_id=b.id, depth_mode='min', return_per_occluder=True)
            results.append({'id': b.id, 'label': b.label, 'res': res})

        # ranking by occlusion_ratio_image descending
        ranked = sorted(results, key=lambda x: x['res']['occlusion_ratio_image'], reverse=True)

        # --- DEBUG: print target occlusion stats for this view ---
        target_id = vv.get('obj_id', None)
        target_res = next((r for r in results if r['id'] == target_id), None)
        if target_res is not None:
            tr = target_res['res']
            per_occ = tr.get('per_occluder', [])
            print(f"View {idx}: target id={target_id} target_area_px={tr.get('target_area_px',0):.0f} occluded_px={tr.get('occluded_area_px',0):.0f} visible_px={tr.get('visible_area_px',0):.0f} occl_ratio_img={tr.get('occlusion_ratio_image',0.0):.4f}")
            if per_occ:
                # show top 6 occluders by pixel overlap
                top_occ = sorted(per_occ, key=lambda x: x['pixels'], reverse=True)[:6]
                for o in top_occ:
                    print(f"  occluder {o['label']} ({o['id']}): pixels={o['pixels']}")
            else:
                print('  no per-occluder overlaps recorded')
        else:
            print(f"View {idx}: target id {target_id} not found among aabbs_all")

        # create figure: left column has real render (top) and top-down map (bottom)
        # right columns: simulated render (top-left), pixel-mask (top-right), ranking (bottom spanning both cols)
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 1.0], hspace=0.2, wspace=0.18)
        ax_real = fig.add_subplot(gs[0, 0])
        ax_map = fig.add_subplot(gs[1, 0])
        ax_img = fig.add_subplot(gs[0, 1])
        ax_mask = fig.add_subplot(gs[0, 2])
        ax_rank = fig.add_subplot(gs[1, 1:3])

    # Left-top: real render (camera_test) if available, else placeholder
    real_render_img = None
    if renderer_prepared and CAMERA_TEST_AVAILABLE:
        try:
            # build camtoworld and K using camera_test helpers if available
            camtoworld = ct_setup_camera(cam_pos, tgt_pos, np.array([0.0, 0.0, -1.0])) if ct_setup_camera is not None else occlusion.camtoworld_from_pos_target(cam_pos, tgt_pos)
            if ct_create_camera_intrinsics is not None:
                K_real = ct_create_camera_intrinsics(width, height)
            else:
                K_real = K
            device_t = device if device is not None else ('cuda' if _torch and _torch.cuda.is_available() else 'cpu')
            viewmat = _torch.linalg.inv(_torch.from_numpy(camtoworld).to(device_t)).unsqueeze(0)
            K_tensor_local = _torch.from_numpy(K_real).to(device_t).unsqueeze(0)
            render_colors, render_alphas, info = ct_render_gaussians(means, quats, scales, opacities, colors, viewmat, K_tensor_local, width, height, sh_degree)
            rendered_image = render_colors[0].cpu().numpy()
            rendered_image = np.clip(rendered_image, 0, 1)
            real_render_img = (rendered_image * 255).astype(np.uint8)
        except Exception as e:
            print('Real render failed for view', idx, e)

    if real_render_img is None:
        # draw placeholder (same as before)
        real_render_img = np.ones((height, width, 3), dtype=np.uint8) * 200
        x = int((cam_pos[0] % 1.0) * width)
        y = int((cam_pos[1] % 1.0) * height)
        real_render_img[min(height-1,y):, :min(width-1,x), 0] = 120

    ax_real.imshow(real_render_img)
    ax_real.axis('off')
    ax_real.set_title('Real render (camera_test)')

    # Left-bottom: top-down map (ax_map)
    # Draw walls as rectangles (use aabbs_walls) and plot object centers for other objects
    # walls as filled polys
    for w in aabbs_walls:
        # derive 2D rectangle corners from bmin/bmax
        bmin = np.array(w.bmin, dtype=float)
        bmax = np.array(w.bmax, dtype=float)
        rect = np.array([[bmin[0], bmin[1]], [bmax[0], bmin[1]], [bmax[0], bmax[1]], [bmin[0], bmax[1]]])
        poly = np.vstack([rect, rect[0]])
        patch = patches.Polygon(rect, closed=True, facecolor=(0.8,0.6,0.4,0.6), edgecolor='saddlebrown')
        ax_map.add_patch(patch)
    # plot non-wall object centers
    obj_xs = []
    obj_ys = []
    for b in aabbs_objs:
        cx = float(b.bmin[0] + 0.5 * (b.bmax[0] - b.bmin[0]))
        cy = float(b.bmin[1] + 0.5 * (b.bmax[1] - b.bmin[1]))
        obj_xs.append(cx); obj_ys.append(cy)
    if obj_xs:
        ax_map.scatter(obj_xs, obj_ys, c='lightgray', s=10, alpha=0.6, label='objects')
    # camera and target
    ax_map.scatter(cam_pos[0], cam_pos[1], c='blue', s=60, label='camera')
    ax_map.scatter(tgt_pos[0], tgt_pos[1], c='red', s=60, label='target')
    fwd = tgt_pos - cam_pos
    if np.linalg.norm(fwd) > 1e-6:
        fwd2 = fwd / np.linalg.norm(fwd)
        ax_map.arrow(cam_pos[0], cam_pos[1], fwd2[0]*0.4, fwd2[1]*0.4, head_width=0.03, color='green')
    ax_map.set_aspect('equal')
    ax_map.set_title('Top-down map (walls + object centers)')
    try:
        ax_map.legend(loc='upper left')
    except Exception:
        pass

    # Determine camera intrinsics to use for occlusion/projected render
    K_used = K
    if CAMERA_TEST_AVAILABLE and ct_create_camera_intrinsics is not None:
        try:
            K_used = ct_create_camera_intrinsics(width, height)
        except Exception:
            K_used = K

    # build a consistent color mapping for all boxes so masks, boxes and ranking share colors
    cmap_global = plt.get_cmap('tab20')
    id2color: Dict[str, tuple] = {}
    for i_b, b in enumerate(aabbs_all):
        id2color[b.id] = cmap_global(i_b % 20)

    # Right-top: simulated render (existing) using the same camtoworld as occlusion/real render
    # origin_upper True means v increases downward (image coords) to match camera_test
    origin_upper = True if CAMERA_TEST_AVAILABLE else False
    id_to_color, id_to_visible, id_buf, items_ids = project_and_paint(
        ax_img, cam_pos, tgt_pos, aabbs_all, K_used, width, height, depth_mode='min', camtoworld=camtoworld, origin_upper=origin_upper, visible_only=True, color_map=id2color)

    # quick diagnostic from raster pass: show target visible pixel count and top visible objects
    try:
        target_id = vv.get('obj_id', None)
        tgt_vis = id_to_visible.get(target_id, 0)
        sorted_vis = sorted([(pid, cnt) for pid,cnt in id_to_visible.items()], key=lambda x: x[1], reverse=True)
        print(f"Raster pass: target id={target_id} visible_pixels={tgt_vis}")
        for pid, cnt in sorted_vis[:6]:
            print(f"  visible: id={pid} pixels={cnt}")
    except Exception:
        pass

    # Prepare pixel-mask visualization (compute per-pixel tgt vs occluder depth maps)
    # We'll rasterize target and occluders similarly to occlusion.occluded_area_on_image
    xs = np.arange(width, dtype=float) + 0.5
    ys = np.arange(height, dtype=float) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    # Build pixel-mask visualization from the raster id_buf so it matches the projected render.
    # find the target box by object id
    target_box = next((b for b in aabbs_all if b.id == vv['obj_id'] or b.id == cfg.object_id or b.id == getattr(vv['cfg'], 'object_id', None)), None)
    if target_box is None:
        target_box = next((b for b in aabbs_all if b.label == cfg.object_label), None)

    if target_box is None:
        ax_mask.text(0.5, 0.5, 'No target bbox available', ha='center', va='center')
        ax_mask.axis('off')
    else:
        # project target polygon to get its mask region
        tgt_poly, tgt_zs = occlusion._project_box_polygon(K_used, camtoworld, target_box.bmin, target_box.bmax, width, height)
        if len(tgt_poly) < 3 or not tgt_zs:
            ax_mask.text(0.5, 0.5, 'Target not in view', ha='center', va='center')
            ax_mask.axis('off')
        else:
            tgt_path = MplPath(np.array(tgt_poly, dtype=float))
            tgt_mask = tgt_path.contains_points(pts).reshape((height, width))

            # mapping from items index -> box id
            # items_ids was returned by project_and_paint and indexes correspond to id_buf values
            try:
                target_idx = items_ids.index(target_box.id)
            except Exception:
                # types may differ; try string match
                target_idx = None
                for ii, pid in enumerate(items_ids):
                    if str(pid) == str(target_box.id):
                        target_idx = ii
                        break

            visible_mask = (id_buf == target_idx) if target_idx is not None else np.zeros((height, width), dtype=bool)
            occluded_mask = tgt_mask & (~visible_mask)

            # Build a full-image id-colored visualization from the raster id_buf so we can
            # see occluders across the whole image (not only inside the target footprint).
            full_img = np.ones((height, width, 3), dtype=np.uint8) * 255
            uniq = np.unique(id_buf)
            for owner_idx in uniq:
                if owner_idx < 0:
                    continue
                owner_idx = int(owner_idx)
                mask_pixels = (id_buf == owner_idx)
                if not np.any(mask_pixels):
                    continue
                try:
                    owner_id = items_ids[owner_idx]
                    col = np.array(id2color.get(owner_id, (0.7, 0.7, 0.7, 1.0))[:3]) * 255.0
                except Exception:
                    col = np.array((0.6, 0.6, 0.6)) * 255.0
                full_img[mask_pixels] = col.astype(np.uint8)

            # For pixels not owned by any polygon (id_buf == -1), keep white background.
            # Optionally highlight the target footprint by overlaying a thin contour.
            origin = 'upper' if origin_upper else 'lower'
            ax_mask.imshow(full_img, origin=origin)
            # draw target outline in black to show where the target projects
            try:
                poly = np.array(tgt_poly)
                patch = patches.Polygon(poly, closed=True, facecolor='none', edgecolor='k', linewidth=0.8)
                ax_mask.add_patch(patch)
            except Exception:
                pass
            ax_mask.set_title('Id-colored raster (target outline shown in black)')
            ax_mask.axis('off')

        # Right-bottom: ranking and visible object color blocks (use global id2color for consistency)
        ax_rank.axis('off')
        topk = ranked[:12]
        y = 0.92
        dy = 0.075
        for i, item in enumerate(topk):
            lbl = f"{item['label']} ({item['id']})"
            occ_img = item['res']['occlusion_ratio_image']
            occ_tgt = item['res']['occlusion_ratio_target']
            color = id2color.get(item['id'], (0.7,0.7,0.7,0.9))
            rect = patches.Rectangle((0.02, y - 0.06), 0.06, 0.055, transform=ax_rank.transAxes, facecolor=color, edgecolor='k')
            ax_rank.add_patch(rect)
            ax_rank.text(0.1, y - 0.03, f"{lbl}: img={occ_img:.3f} tgt={occ_tgt:.3f}", transform=ax_rank.transAxes, va='center')
            y -= dy

        fig.suptitle(f"Scene {Path(scene_path).name} - obj {cfg.object_label} preset {vv['preset']} ({idx+1}/{len(views)})")
        out_file = Path(outdir) / f"occlusion_{Path(scene_path).name}_{idx:02d}.png"
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        print('Saved', out_file)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True)
    parser.add_argument('--n_views', type=int, default=6)
    parser.add_argument('--outdir', default='./tmp/occlusion_checks')
    args = parser.parse_args()
    run_scene(args.scene, n_views=args.n_views, outdir=args.outdir)


if __name__ == '__main__':
    main()
