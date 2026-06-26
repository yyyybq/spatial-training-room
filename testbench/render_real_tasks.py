#!/usr/bin/env python3
"""Render real task views with gsplat from 3D Gaussian scene data.

Example:
  python testbench/render_real_tasks.py \
      --scene "C:/Users/user/Desktop/0267_840790" \
      --jsonl "out/batch/T21.jsonl" \
      --out "out/real" \
      --max 2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
try:
    from gsplat.rendering import rasterization
except Exception:
    rasterization = None


def _ensure_loader_importable(repo_root: Path) -> None:
    """Ensure `ply_gaussian_loader` can be imported.

    Prefer local module; fallback to sibling ViewSuite_InGS/ViewSuite.
    """
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    view_suite_loader_dir = repo_root.parent / "ViewSuite_InGS" / "ViewSuite"
    if view_suite_loader_dir.exists() and str(view_suite_loader_dir) not in sys.path:
        sys.path.insert(0, str(view_suite_loader_dir))


def setup_camera(camera_pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create camera-to-world matrix for gsplat rendering."""
    forward = target - camera_pos
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / fn

    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(forward[2]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, temp_up)
        rn = np.linalg.norm(right)
        right = right / max(rn, 1e-8)
    else:
        right = right / rn

    up_corrected = np.cross(right, forward)

    camtoworld = np.eye(4, dtype=np.float32)
    camtoworld[:3, 0] = -right
    camtoworld[:3, 1] = up_corrected
    camtoworld[:3, 2] = forward
    camtoworld[:3, 3] = camera_pos
    return camtoworld


def create_intrinsics(width: int, height: int, fov_factor: float = 0.4) -> np.ndarray:
    focal = width * fov_factor
    return np.array(
        [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]],
        dtype=np.float32,
    )


def prepare_gaussian_tensors(gs_data, device: str):
    means = torch.from_numpy(gs_data.positions).float().to(device)
    quats = torch.from_numpy(gs_data.get_rotations_xyzw()).float().to(device)
    scales = torch.exp(torch.from_numpy(gs_data.scales).float().to(device))
    opacities = torch.sigmoid(torch.from_numpy(gs_data.opacities.squeeze()).float().to(device))
    if gs_data.sh_rest is not None:
        colors = torch.from_numpy(gs_data.get_sh_coefficients()).float().to(device)
        sh_degree = gs_data.sh_bands
    else:
        colors = torch.from_numpy(gs_data.get_linear_colors()).float().to(device)
        sh_degree = None
    return means, quats, scales, opacities, colors, sh_degree


def render_view(
    means,
    quats,
    scales,
    opacities,
    colors,
    sh_degree,
    position: List[float],
    target: List[float],
    width: int,
    height: int,
    device: str,
    up_vec: Optional[np.ndarray] = None,
    fov_factor: float = 0.4,
) -> np.ndarray:
    cam_pos = np.array(position, dtype=np.float32)
    tgt_pos = np.array(target, dtype=np.float32)
    up = up_vec if up_vec is not None else np.array([0.0, 0.0, -1.0], dtype=np.float32)

    camtoworld = setup_camera(cam_pos, tgt_pos, up)
    viewmat = torch.linalg.inv(torch.from_numpy(camtoworld).to(device)).unsqueeze(0)
    K = create_intrinsics(width, height, fov_factor=fov_factor)
    K_tensor = torch.from_numpy(K).to(device).unsqueeze(0)

    # Prefer gsplat rasterization when available (typically requires CUDA backend).
    if rasterization is not None:
        try:
            render_colors, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=K_tensor,
                width=width,
                height=height,
                sh_degree=sh_degree,
                packed=False,
            )
            img = render_colors[0].detach().cpu().numpy()
            img = np.clip(img, 0.0, 1.0)
            return (img * 255).astype(np.uint8)
        except Exception as e:
            print(f"[warn] gsplat rasterization failed, fallback to CPU splat: {e}")

    # CPU fallback: depth-tested point splat from Gaussian centers/colors.
    means_np = means.detach().cpu().numpy()
    opac_np = opacities.detach().cpu().numpy()
    if colors.ndim == 3:
        # SH mode: use DC coeff (index 0) as base color estimate.
        base = colors[:, 0, :].detach().cpu().numpy()
        color_np = np.clip(base * 0.28209479177387814 + 0.5, 0.0, 1.0)
    else:
        color_np = np.clip(colors.detach().cpu().numpy(), 0.0, 1.0)

    world2cam = np.linalg.inv(camtoworld)
    pts_h = np.concatenate([means_np, np.ones((means_np.shape[0], 1), dtype=np.float32)], axis=1)
    cam_pts = (world2cam @ pts_h.T).T[:, :3]

    z = cam_pts[:, 2]
    valid = z > 1e-4
    cam_pts = cam_pts[valid]
    z = z[valid]
    col = color_np[valid]
    alp = opac_np[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    px = (fx * (cam_pts[:, 0] / z) + cx).astype(np.int32)
    py = (fy * (cam_pts[:, 1] / z) + cy).astype(np.int32)

    in_img = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    px = px[in_img]
    py = py[in_img]
    z = z[in_img]
    col = col[in_img]
    alp = np.clip(alp[in_img], 0.0, 1.0)

    # Compute per-point projected Gaussian radius from 3D scales.
    scales_np_all = scales.detach().cpu().numpy()
    if scales_np_all.ndim > 1:
        max_s_all = np.max(scales_np_all, axis=-1)
    else:
        max_s_all = scales_np_all.ravel()
    max_s_filt = max_s_all[valid][in_img]
    # Projected radius in pixels: r = max_scale * focal / depth, clamped to [2, 14]
    px_radii = np.clip((max_s_filt * fx / (z + 1e-6)).astype(np.int32), 2, 14)

    img = np.ones((height, width, 3), dtype=np.float32) * 0.95

    # Back-to-front (painter's algorithm) for proper alpha compositing.
    order = np.argsort(z)[::-1]
    for idx in order:
        base_a = float(alp[idx])
        if base_a < 0.02:
            continue
        x0 = int(px[idx])
        y0 = int(py[idx])
        r = int(px_radii[idx])
        r2 = float(r * r + 1e-6)
        c = col[idx]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dist2 = float(dx * dx + dy * dy)
                if dist2 > r2:
                    continue
                weight = math.exp(-2.5 * dist2 / r2)
                xx = x0 + dx
                yy = y0 + dy
                if not (0 <= xx < width and 0 <= yy < height):
                    continue
                a = base_a * weight
                img[yy, xx, :] = c * a + img[yy, xx, :] * (1.0 - a)

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def render_floorplan_topdown(
    scene_dir: Path,
    trajectory_views: List[Tuple[List[float], List[float]]],
    target_object_ids: List[str],
    out_path: Path,
) -> bool:
    """Draw a comprehensive matplotlib floor plan with room layout, object bboxes,
    and camera trajectory overlaid.

    Returns True on success, False if matplotlib is unavailable or data missing.
    """
    if not _HAS_MPL:
        return False

    labels_path = scene_dir / "labels.json"
    structure_path = scene_dir / "structure.json"

    # Load room polygons from structure.json
    rooms_polys: List[np.ndarray] = []
    if structure_path.exists():
        with structure_path.open("r", encoding="utf-8") as f:
            struct = json.load(f)
        for room in struct.get("rooms", []):
            profile = room.get("profile")
            if profile and len(profile) >= 3:
                arr = np.array(profile, dtype=float)
                # profile entries may be [x, y] or [x, y, z]; keep first two
                rooms_polys.append(arr[:, :2])

    # Load object AABBs from labels.json
    objs: List[dict] = []
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        for obj in labels:
            bb = obj.get("bounding_box")
            if not bb:
                continue
            label_name = obj.get("label", "")
            if label_name in ("wall", "ceiling", "floor", "other"):
                continue  # skip structural elements to reduce clutter
            xs = [p["x"] for p in bb]
            ys = [p["y"] for p in bb]
            bmin = np.array([min(xs), min(ys)], dtype=float)
            bmax = np.array([max(xs), max(ys)], dtype=float)
            objs.append({
                "ins_id": str(obj.get("ins_id", "")),
                "label": label_name,
                "bmin": bmin,
                "bmax": bmax,
                "center": 0.5 * (bmin + bmax),
            })

    target_ids = set(str(tid) for tid in target_object_ids if tid is not None)

    fig, ax = plt.subplots(figsize=(9, 9))

    # --- Draw rooms ---
    for poly in rooms_polys:
        closed = np.vstack([poly, poly[0]])
        ax.fill(closed[:, 0], closed[:, 1], alpha=0.06, color="#7799bb")
        ax.plot(closed[:, 0], closed[:, 1], "-", color="#445566", linewidth=1.5)

    # --- Draw object bounding boxes ---
    for o in objs:
        bmin = o["bmin"]
        bmax = o["bmax"]
        w = float(bmax[0] - bmin[0])
        h = float(bmax[1] - bmin[1])
        is_target = o["ins_id"] in target_ids
        edge_color = "#cc2222" if is_target else "#aaaaaa"
        face_color = "#ffdddd" if is_target else "none"
        lw = 2.0 if is_target else 0.7
        rect = Rectangle(
            (float(bmin[0]), float(bmin[1])), w, h,
            facecolor=face_color, edgecolor=edge_color, linewidth=lw, alpha=0.75,
        )
        ax.add_patch(rect)
        cx, cy = float(o["center"][0]), float(o["center"][1])
        ax.text(
            cx, cy, o["label"],
            fontsize=5.5, ha="center", va="center",
            color=edge_color,
            fontweight="bold" if is_target else "normal",
        )

    # --- Draw camera trajectory ---
    n = len(trajectory_views)
    if n > 1:
        xs_traj = [float(p[0]) for p, _ in trajectory_views]
        ys_traj = [float(p[1]) for p, _ in trajectory_views]
        ax.plot(xs_traj, ys_traj, "-", color="#888888", linewidth=0.8, alpha=0.55, zorder=5)

    for k, (pos, tgt) in enumerate(trajectory_views):
        px_c = float(pos[0])
        py_c = float(pos[1])
        tx = float(tgt[0])
        ty = float(tgt[1])

        if k == 0:
            pt_color = "#dd2222"   # red = init
            pt_label = "INIT"
            zord = 10
        elif k == n - 1:
            pt_color = "#22aa44"   # green = target/final
            pt_label = "END"
            zord = 9
        else:
            pt_color = "#2255cc"   # blue = intermediate
            pt_label = None
            zord = 7

        # Camera position dot
        ax.scatter([px_c], [py_c], c=pt_color, s=70, zorder=zord,
                   edgecolors="white", linewidths=1.0)

        # View-direction arrow (normalized to 0.4 m)
        ddx = tx - px_c
        ddy = ty - py_c
        norm = max(math.sqrt(ddx * ddx + ddy * ddy), 1e-6)
        arr_len = 0.4
        ax.annotate(
            "",
            xy=(px_c + ddx / norm * arr_len, py_c + ddy / norm * arr_len),
            xytext=(px_c, py_c),
            arrowprops=dict(arrowstyle="->", color=pt_color, lw=1.5),
            zorder=zord,
        )

        if pt_label:
            ax.text(
                px_c + 0.12, py_c + 0.12, pt_label,
                fontsize=7, color=pt_color, fontweight="bold", zorder=zord + 1,
            )

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Floor plan: {scene_dir.name}", fontsize=10)
    ax.grid(True, alpha=0.15, linestyle=":")
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Y (m)", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    return True


def _draw_marker(
    img: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[float, float, float],
    outline: bool = True,
) -> None:
    """Draw a filled circle marker on img (uint8 HxWx3, in-place)."""
    h, w = img.shape[:2]
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist2 = dx * dx + dy * dy
            if dist2 > radius * radius:
                continue
            px_ = cx + dx
            py_ = cy + dy
            if not (0 <= px_ < w and 0 <= py_ < h):
                continue
            if outline and dist2 > (radius - 2) * (radius - 2):
                # White outline ring
                img[py_, px_] = [255, 255, 255]
            else:
                img[py_, px_] = [int(c * 255) for c in color]


def render_topdown(
    means,
    quats,
    scales,
    opacities,
    colors,
    sh_degree,
    trajectory_views: List[Tuple[List[float], List[float]]],
    width: int,
    height: int,
    device: str,
) -> np.ndarray:
    """Render a bird's-eye top-down view with trajectory waypoint markers.

    The camera is placed vertically above the centroid of trajectory positions
    looking straight down.  A wider FOV (fov_factor=0.5, ~90 deg) is used so
    more of the floor plan is visible.  Waypoints are overlaid as coloured dots:
      red   = init (first waypoint)
      green = target (last waypoint)
      blue  = intermediate waypoints
    """
    if not trajectory_views:
        return np.full((height, width, 3), 200, dtype=np.uint8)

    all_pos = [p for p, _ in trajectory_views]
    cx = float(np.mean([p[0] for p in all_pos]))
    cy = float(np.mean([p[1] for p in all_pos]))
    cz = float(np.mean([p[2] for p in all_pos]))  # typically ~0.8 (eye height)

    # Place camera 8 m above the trajectory centroid, looking straight down.
    cam_height = cz + 8.0
    cam_pos = [cx, cy, cam_height]
    cam_target = [cx, cy, cz]

    # Use Y-axis as image-up so the top-down view is consistently oriented.
    up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    img = render_view(
        means, quats, scales, opacities, colors, sh_degree,
        cam_pos, cam_target, width, height, device,
        up_vec=up_vec, fov_factor=0.5,
    )

    # -- Overlay trajectory markers -----------------------------------------
    K = create_intrinsics(width, height, fov_factor=0.5)
    cam_pos_arr = np.array(cam_pos, dtype=np.float32)
    cam_tgt_arr = np.array(cam_target, dtype=np.float32)
    camtoworld = setup_camera(cam_pos_arr, cam_tgt_arr, up_vec)
    world2cam = np.linalg.inv(camtoworld)

    n = len(trajectory_views)
    for k, (pos, _) in enumerate(trajectory_views):
        p_w = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float32)
        p_c = world2cam @ p_w
        if p_c[2] < 1e-4:
            continue
        px_ = int(K[0, 0] * (p_c[0] / p_c[2]) + K[0, 2])
        py_ = int(K[1, 1] * (p_c[1] / p_c[2]) + K[1, 2])
        if k == 0:
            color = (0.85, 0.10, 0.10)   # red = init
        elif k == n - 1:
            color = (0.10, 0.75, 0.10)   # green = target
        else:
            color = (0.15, 0.40, 0.90)   # blue = intermediate
        _draw_marker(img, px_, py_, radius=7, color=color)

    return img


def _load_jsonl(path: Path, max_n: int) -> List[Dict]:
    tasks: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
            if len(tasks) >= max_n:
                break
    return tasks


def _normalize_view(view_like) -> Optional[Tuple[List[float], List[float]]]:
    if isinstance(view_like, dict):
        pos = view_like.get("position") or view_like.get("pos")
        tgt = view_like.get("target") or view_like.get("look_at")
    elif isinstance(view_like, (list, tuple)) and len(view_like) == 2:
        pos, tgt = view_like
    else:
        return None

    if pos is None or tgt is None:
        return None
    if len(pos) != 3 or len(tgt) != 3:
        return None
    try:
        p = [float(pos[0]), float(pos[1]), float(pos[2])]
        t = [float(tgt[0]), float(tgt[1]), float(tgt[2])]
    except Exception:
        return None
    return p, t


def _extract_trajectory_views(task: Dict) -> List[Tuple[List[float], List[float]]]:
    views: List[Tuple[List[float], List[float]]] = []
    for item in task.get("expert_trajectory") or []:
        parsed = _normalize_view(item)
        if parsed is not None:
            views.append(parsed)

    if views:
        return views

    init_v = _normalize_view(task.get("init_view") or {})
    target_v = _normalize_view(task.get("target_view") or {})
    if init_v is not None:
        views.append(init_v)
    if target_v is not None and (not views or target_v != views[-1]):
        views.append(target_v)
    return views


def _iter_scene_dirs(scene_root: Path) -> List[Path]:
    dirs: List[Path] = []
    if not scene_root.exists():
        return dirs
    for child in sorted(scene_root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "3dgs_compressed.ply").exists():
            dirs.append(child)
    return dirs


def _collect_jobs(args) -> List[Tuple[Path, Path, Path]]:
    jobs: List[Tuple[Path, Path, Path]] = []
    out_base = Path(args.out)

    # Single-scene mode.
    if args.scene and args.jsonl:
        scene_dir = Path(args.scene)
        jsonl_path = Path(args.jsonl)
        # Use out_base directly (no extra stem sub-folder) to avoid double nesting.
        jobs.append((scene_dir, jsonl_path, out_base))
        return jobs

    # Multi-scene sweep mode.
    if args.scene_root and args.jsonl_root:
        scene_root = Path(args.scene_root)
        jsonl_root = Path(args.jsonl_root)
        for scene_dir in _iter_scene_dirs(scene_root):
            scene_name = scene_dir.name
            per_scene_jsonl_dir = jsonl_root / scene_name
            if not per_scene_jsonl_dir.exists():
                continue
            for jsonl_path in sorted(per_scene_jsonl_dir.glob(args.jsonl_glob)):
                jobs.append((scene_dir, jsonl_path, out_base / scene_name / jsonl_path.stem))
        return jobs

    raise ValueError(
        "Provide either (--scene + --jsonl) for single-scene mode, "
        "or (--scene-root + --jsonl-root) for multi-scene sweep mode."
    )


def _render_jsonl(
    *,
    scene_dir: Path,
    jsonl_path: Path,
    out_root: Path,
    loader,
    device: str,
    width: int,
    height: int,
    max_tasks: int,
) -> None:
    ply_path = scene_dir / "3dgs_compressed.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    gs_data = loader.load_ply(str(ply_path))
    means, quats, scales, opacities, colors, sh_degree = prepare_gaussian_tensors(gs_data, device)

    tasks = _load_jsonl(jsonl_path, max_tasks)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene": str(scene_dir),
        "jsonl": str(jsonl_path),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "width": int(width),
        "height": int(height),
        "max_tasks": int(max_tasks),
        "tasks": [],
    }

    for i, t in enumerate(tasks):
        task_id = t.get("task_id", f"task_{i:03d}")
        template_id = t.get("template_id", "unknown")
        action_descriptions = list(t.get("action_descriptions") or [])
        trajectory = _extract_trajectory_views(t)
        if not trajectory:
            continue

        task_dir = out_root / f"task_{i:03d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        frame_files: List[str] = []
        for k, (pos, tgt) in enumerate(trajectory):
            img = render_view(
                means,
                quats,
                scales,
                opacities,
                colors,
                sh_degree,
                pos,
                tgt,
                width,
                height,
                device,
            )
            frame_name = f"step_{k:03d}.png"
            frame_path = task_dir / frame_name
            imageio.imwrite(frame_path, img)
            frame_files.append(frame_name)

            if k == 0:
                imageio.imwrite(task_dir / "init.png", img)
            if k == len(trajectory) - 1:
                imageio.imwrite(task_dir / "target.png", img)

        # Floor plan topdown: matplotlib room-layout diagram with trajectory overlay.
        # Falls back to Gaussian bird's-eye if matplotlib is unavailable.
        target_ids_raw = t.get("target_object_id") or t.get("target_id") or []
        if isinstance(target_ids_raw, str):
            target_ids_raw = [target_ids_raw]
        fp_ok = render_floorplan_topdown(
            scene_dir=scene_dir,
            trajectory_views=trajectory,
            target_object_ids=list(target_ids_raw),
            out_path=task_dir / "topdown.png",
        )
        if not fp_ok:
            # Fallback: Gaussian bird's-eye render
            topdown_img = render_topdown(
                means, quats, scales, opacities, colors, sh_degree,
                trajectory,
                width, height, device,
            )
            imageio.imwrite(task_dir / "topdown.png", topdown_img)

        manifest["tasks"].append(
            {
                "task_index": i,
                "task_id": task_id,
                "template_id": template_id,
                "num_frames": len(frame_files),
                "num_actions": len(action_descriptions),
                "action_descriptions": action_descriptions,
                "task_dir": task_dir.name,
                "frames": frame_files,
                "init_frame": "init.png",
                "target_frame": "target.png",
                "topdown_frame": "topdown.png",
            }
        )
        print(
            f"[saved] {scene_dir.name} {jsonl_path.stem} task {i:03d} "
            f"({len(frame_files)} frames + topdown)"
        )

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[done] outputs: {out_root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="Absolute path to single scene directory with 3dgs_compressed.ply")
    ap.add_argument("--jsonl", help="Single task JSONL path")
    ap.add_argument("--scene-root", help="Root containing many scene dirs (multi-scene mode)")
    ap.add_argument("--jsonl-root", help="Root containing per-scene JSONL folders (multi-scene mode)")
    ap.add_argument("--jsonl-glob", default="*.jsonl", help="JSONL glob in each per-scene folder")
    ap.add_argument("--out", default="out/real")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--height", type=int, default=384)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_loader_importable(repo_root)

    try:
        from ply_gaussian_loader import PLYGaussianLoader
    except Exception as e:
        raise RuntimeError("Cannot import PLYGaussianLoader. Ensure ViewSuite_InGS/ViewSuite is present.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[render] device={device}")

    loader = PLYGaussianLoader()
    jobs = _collect_jobs(args)
    if not jobs:
        print("[warn] no jobs found; check --scene-root/--jsonl-root structure")
        return

    print(f"[render] jobs={len(jobs)}")
    for scene_dir, jsonl_path, out_root in jobs:
        _render_jsonl(
            scene_dir=scene_dir,
            jsonl_path=jsonl_path,
            out_root=out_root,
            loader=loader,
            device=device,
            width=args.width,
            height=args.height,
            max_tasks=args.max,
        )


if __name__ == "__main__":
    main()
