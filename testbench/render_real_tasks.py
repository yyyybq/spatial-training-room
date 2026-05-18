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
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
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
) -> np.ndarray:
    cam_pos = np.array(position, dtype=np.float32)
    tgt_pos = np.array(target, dtype=np.float32)
    up = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    camtoworld = setup_camera(cam_pos, tgt_pos, up)
    viewmat = torch.linalg.inv(torch.from_numpy(camtoworld).to(device)).unsqueeze(0)
    K = create_intrinsics(width, height)
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

    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    # Paint nearest points first by depth test.
    # Use a tiny 3x3 kernel for denser, less sparse output.
    kernel = [
        (-1, -1, 0.20), (0, -1, 0.40), (1, -1, 0.20),
        (-1,  0, 0.40), (0,  0, 1.00), (1,  0, 0.40),
        (-1,  1, 0.20), (0,  1, 0.40), (1,  1, 0.20),
    ]
    order = np.argsort(z)
    for idx in order:
        x = px[idx]
        y = py[idx]
        base_a = float(alp[idx])
        for dx, dy, w in kernel:
            xx = x + dx
            yy = y + dy
            if xx < 0 or xx >= width or yy < 0 or yy >= height:
                continue
            z_local = z[idx] + 0.001 * (abs(dx) + abs(dy))
            if z_local >= zbuf[yy, xx]:
                continue
            zbuf[yy, xx] = z_local
            a = min(1.0, base_a * w)
            img[yy, xx, :] = col[idx] * a + img[yy, xx, :] * (1.0 - a)

    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Absolute path to scene directory with 3dgs_compressed.ply")
    ap.add_argument("--jsonl", required=True, help="Task JSONL path")
    ap.add_argument("--out", default="out/real")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _ensure_loader_importable(repo_root)

    try:
        from ply_gaussian_loader import PLYGaussianLoader
    except Exception as e:
        raise RuntimeError("Cannot import PLYGaussianLoader. Ensure ViewSuite_InGS/ViewSuite is present.") from e

    scene_dir = Path(args.scene)
    ply_path = scene_dir / "3dgs_compressed.ply"
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[render] device={device}")

    loader = PLYGaussianLoader()
    gs_data = loader.load_ply(str(ply_path))
    means, quats, scales, opacities, colors, sh_degree = prepare_gaussian_tensors(gs_data, device)

    jsonl_path = Path(args.jsonl)
    tasks = _load_jsonl(jsonl_path, args.max)
    out_root = Path(args.out) / jsonl_path.stem
    out_root.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(tasks):
        init_v = t.get("init_view") or {}
        target_v = t.get("target_view") or {}
        if not init_v or not target_v:
            continue

        init_img = render_view(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree,
            init_v["position"],
            init_v["target"],
            args.width,
            args.height,
            device,
        )
        tgt_img = render_view(
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree,
            target_v["position"],
            target_v["target"],
            args.width,
            args.height,
            device,
        )

        imageio.imwrite(out_root / f"{jsonl_path.stem}_task_{i:03d}_init.png", init_img)
        imageio.imwrite(out_root / f"{jsonl_path.stem}_task_{i:03d}_target.png", tgt_img)
        print(f"[saved] {jsonl_path.stem} task {i:03d} (init/target)")

    print(f"[done] outputs: {out_root}")


if __name__ == "__main__":
    main()
