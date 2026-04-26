#!/usr/bin/env python3
"""
Camera rendering test for Gaussian Splatting
Tests rendering from specified camera position with proper Gaussian point adjustment
"""

import numpy as np
import torch
import imageio
from ply_gaussian_loader import PLYGaussianLoader
from gsplat.rendering import rasterization


def setup_camera(camera_pos, target, up):
    """
    Setup camera matrices from position, target and up vector

    Args:
        camera_pos: Camera position in world coordinates
        target: Target point to look at
        up: Up vector

    Returns:
        camtoworld: 4x4 camera-to-world transform matrix
    """
    # compute forward vector and normalize safely
    forward = target - camera_pos
    fn = np.linalg.norm(forward)
    if fn < 1e-8:
        # degenerate: pick sensible default forward
        forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        forward = forward / fn
    forward = forward / (np.linalg.norm(forward)+1e-8)

    # compute right vector as cross(forward, up) with robust fallback if up is colinear
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        # up and forward are (nearly) colinear; pick a temporary up to produce a valid right
        # Choose a world axis that is not parallel to forward
        if abs(forward[2]) < 0.9:
            temp_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            temp_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, temp_up)
        rn = np.linalg.norm(right)
        if rn < 1e-8:
            # final fallback
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right = right / rn
    else:
        right = right / rn

    up_corrected = np.cross(right, forward)

    camtoworld = np.eye(4, dtype=np.float32)
    camtoworld[:3, 0] = -right
    camtoworld[:3, 1] = up_corrected
    camtoworld[:3, 2] = forward
    camtoworld[:3, 3] = camera_pos

    return camtoworld


def prepare_gaussian_data(gs_data, device, use_sh=True):
    """
    Convert Gaussian splat data to PyTorch tensors for rendering

    Args:
        gs_data: GaussianSplatData object
        device: PyTorch device (cuda/cpu)
        use_sh: Whether to use spherical harmonics (True) or RGB colors (False)

    Returns:
        means, quats, scales, opacities, colors, sh_degree: PyTorch tensors and SH degree
    """
    # Convert positions
    means = torch.from_numpy(gs_data.positions).float().to(device)

    # Get rotations in XYZW format (normalized)
    quats = torch.from_numpy(gs_data.get_rotations_xyzw()).float().to(device)

    # Convert scales (already in log space)
    scales_log = torch.from_numpy(gs_data.scales).float().to(device)
    scales = torch.exp(scales_log)

    # Convert opacities from logit to probability space
    opacities_logit = torch.from_numpy(gs_data.opacities.squeeze()).float().to(device)
    opacities = torch.sigmoid(opacities_logit)

    # Handle colors/SH coefficients
    sh_degree = None
    if use_sh and gs_data.sh_rest is not None:
        # Use spherical harmonics
        sh_coeffs = gs_data.get_sh_coefficients()
        colors = torch.from_numpy(sh_coeffs).float().to(device)
        sh_degree = gs_data.sh_bands
    else:
        # Use RGB colors (fallback)
        colors_rgb = gs_data.get_linear_colors()
        colors = torch.from_numpy(colors_rgb).float().to(device)

    return means, quats, scales, opacities, colors, sh_degree


def create_camera_intrinsics(width, height, fov_factor=0.4):
    """
    Create camera intrinsic matrix

    Args:
        width, height: Image dimensions
        fov_factor: FOV scaling factor

    Returns:
        K: 3x3 intrinsic matrix
    """
    focal = width * fov_factor
    K = np.array([
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ], dtype=np.float32)

    return K


def render_gaussians(means, quats, scales, opacities, colors, viewmat, K_tensor, width, height, sh_degree=None):
    """
    Render Gaussian splats using gsplat

    Args:
        means, quats, scales, opacities, colors: Gaussian parameters
        viewmat: View matrix
        K_tensor: Intrinsic matrix
        width, height: Image dimensions
        sh_degree: Spherical harmonics degree (None for RGB colors)

    Returns:
        render_colors, render_alphas, info: Rendering results
    """
    return rasterization(
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


def save_rendered_image(render_colors, output_path):
    """
    Save rendered image to file

    Args:
        render_colors: Rendered color tensor
        output_path: Output file path
    """
    rendered_image = render_colors[0].cpu().numpy()
    rendered_image = np.clip(rendered_image, 0, 1)
    imageio.imwrite(output_path, (rendered_image * 255).astype(np.uint8))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Camera configuration
    camera_pos = np.array( [
      2.5857170000000003,
      7.152832,
      0.85999999077824
    ], dtype=np.float32)
    target = np.array([
        2.5857170000000003,
        10.152832,
        1.055999999077824
      ], dtype=np.float32)
    up = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    print(f"Camera settings:")
    print(f"  Position: [{camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}]")
    print(f"  Target: [{target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}]")

    # Load PLY data
    print(f"\n=== Loading PLY data ===")
    ply_path = "/data/liubinglin/jijiatong/ViewSuite/data/0017_840813/3dgs_compressed.ply"
    output_path = "camera_test_render.png"

    loader = PLYGaussianLoader()
    gs_data = loader.load_ply(ply_path)
    gs_data.print_info()

    print(f"Scene center: {gs_data.positions.mean(axis=0)}")

    # Prepare data for rendering with SH support
    means, quats, scales, opacities, colors, sh_degree = prepare_gaussian_data(gs_data, device, use_sh=True)

    print(f"\nData statistics:")
    print(f"  Position range: [{means.min().item():.2f}, {means.max().item():.2f}]")
    print(f"  Scale range: [{scales.min().item():.6f}, {scales.max().item():.6f}]")
    print(f"  Opacity range: [{opacities.min().item():.3f}, {opacities.max().item():.3f}]")

    if sh_degree is not None:
        print(f"  Using SH degree: {sh_degree}, Colors shape: {colors.shape}")
    else:
        print(f"  Using RGB colors, Colors shape: {colors.shape}")

    # Setup camera
    camtoworld = setup_camera(camera_pos, target, up)

    # Camera intrinsics
    width, height = 400, 400
    K = create_camera_intrinsics(width, height)

    # Convert to PyTorch tensors
    viewmat = torch.linalg.inv(torch.from_numpy(camtoworld).to(device)).unsqueeze(0)
    K_tensor = torch.from_numpy(K).to(device).unsqueeze(0)

    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)

    print(f"\nCamera configuration:")
    print(f"  Forward vector: [{forward[0]:.3f}, {forward[1]:.3f}, {forward[2]:.3f}]")
    print(f"  Image size: {width}x{height}")
    print(f"  Focal length: {K[0,0]:.1f}")

    # Render
    print(f"\n=== Starting render ===")
    try:
        render_colors, render_alphas, info = render_gaussians(
            means, quats, scales, opacities, colors, viewmat, K_tensor, width, height, sh_degree
        )

        visible_count = (info['radii'] > 0).sum().item()
        print(f"Visible Gaussians: {visible_count} / {len(means)}")

        # Save result
        save_rendered_image(render_colors, output_path)

        print(f"Rendering complete!")
        print(f"Image saved to: {output_path}")

    except Exception as e:
        print(f"Rendering failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()