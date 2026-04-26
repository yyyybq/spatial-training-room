from __future__ import annotations
import json
from pathlib import Path
import math
import matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
import matplotlib.pyplot as plt
from ply_gaussian_loader import PLYGaussianLoader

from .camera_generation import SemanticCamera
import torch

def create_camera_intrinsics(width, height, fov_factor=0.4):
    focal = width * fov_factor
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1]], dtype=np.float32)
    return K


def render_gaussians(means, quats, scales, opacities, colors, viewmat, K_tensor, width, height, sh_degree=None):
    from gsplat.rendering import rasterization
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


def render_thumbnail_for_pose(scene_path: Path, pose: dict, thumb_size: int = 400, gpu_id: int = 0):
    """
    Render a thumbnail for a given pose using the specified GPU.

    Args:
        scene_path (Path): Path to the scene folder.
        pose (dict): Camera pose with 'position' and 'target'.
        thumb_size (int): Size of the thumbnail (width and height).
        gpu_id (int): GPU ID to use for rendering.

    Returns:
        np.ndarray: Rendered image as a NumPy array.
    """
    loader = PLYGaussianLoader()
    gs_data = loader.load_ply(scene_path / '3dgs_compressed.ply')

    camera_pos = np.array(pose['position'], dtype=np.float32)
    target = np.array(pose['target'], dtype=np.float32)
    up = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    camtoworld = setup_camera(camera_pos, target, up)
    width = thumb_size
    height = thumb_size
    K = create_camera_intrinsics(width, height)

    # Explicitly set the GPU device
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    print(f"Using device: {device}")

    viewmat = torch.linalg.inv(torch.from_numpy(camtoworld).to(device)).unsqueeze(0)
    K_tensor_local = torch.from_numpy(K).to(device).unsqueeze(0)
    means, quats, scales, opacities, colors, sh_degree = prepare_gaussian_data(gs_data, device, use_sh=True)
    render_colors, render_alphas, info = render_gaussians(means, quats, scales, opacities, colors, viewmat, K_tensor_local, width, height, sh_degree)
    rendered_image = render_colors[0].cpu().numpy()
    rendered_image = np.clip(rendered_image, 0, 1)
    return (rendered_image * 255).astype(np.uint8)

def compose_preview_for_item(item: dict, scene_path: Path, out_path: Path, tvgen=None, means=None, quats=None, scales=None, opacities=None, colors=None, sh_degree=None, device=None, thumb_size=256, annotations=None):
    """Create a single image combining top-down indicators and rendered thumbnails for one QA item, and save it to out_path."""
    # create figure with left top-down and right thumbnail(s)
    qtype = item.get('qtype')
    fig_w = 10
    fig_h = 5
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax_map = fig.add_subplot(gs[0,0])
    ax_thumb = fig.add_subplot(gs[0,1])

    # draw scene objects if available
    sc = None
    obj_lookup = {}
    if SemanticCamera is not None and scene_path.exists():
        sc = SemanticCamera(str(scene_path))
        objs = sc.list_objects()
        xs = []
        ys = []
        for oid in list(objs.keys())[:500]:
            obj = sc.get_object(oid)
            if obj is None:
                continue
            obj_lookup[str(oid)] = obj
            xs.append(float(obj.position[0]))
            ys.append(float(obj.position[1]))
        if xs:
            ax_map.scatter(xs, ys, c='lightgray', s=8, alpha=0.6, label='objects')


    # Collect all relevant render configs and labels for thumbnails
    thumbs = []
    thumb_labels = []
    choices_text = None
    if qtype == 'action_next_frame_mca':
        choices = item.get('choices', [])
        meta = item.get('meta', {})
        current_frame = meta.get('current_frame', None)

        # Include QUESTION image (current view) when possible
        current_thumb = None
        # Prefer matching by current_frame if provided
        if current_frame is not None:
            for c in choices:
                if isinstance(c, dict) and c.get('frame') == current_frame:
                    pos = np.array(c.get('position', [0, 0, 0]), dtype=float)
                    tgt = np.array(c.get('target', pos + np.array([0, 1, 0])), dtype=float)
                    current_thumb = {'position': pos, 'target': tgt, 'render': c.get('render')}
                    break
        # Fallback to explicit current_pos/target if available
        if current_thumb is None and isinstance(meta, dict) and ('current_pos' in meta or 'current_render' in meta):
            pos = np.array(meta.get('current_pos', [0, 0, 0]), dtype=float)
            tgt = np.array(meta.get('current_target', pos + np.array([0, 1, 0])), dtype=float)
            current_thumb = {'position': pos, 'target': tgt, 'render': meta.get('current_render')}

        if current_thumb is not None:
            # Only keep pos-based preview
            thumbs.append({'position': current_thumb['position'], 'target': current_thumb['target'], 'render': None})
            thumb_labels.append('current_view')
            # also overlay on map (legend-based annotation)
            pos = np.array(current_thumb['position'], dtype=float)
            tgt = np.array(current_thumb['target'], dtype=float)
            fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
            ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='current_view')
            ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        for idx, c in enumerate(choices):
            # Use the render field if present, else fallback to position/target
            if 'render' in c:
                render_cfg = c['render']
                pos = np.array(c['position'], dtype=float)
                tgt = np.array(c['target'], dtype=float)
            else:
                render_cfg = None
                pos = np.array(c['position'], dtype=float)
                tgt = np.array(c['target'], dtype=float)
            # add pos-based only
            thumbs.append({'position': pos, 'target': tgt, 'render': None})
            choice_label = chr(ord('A') + idx)
            thumb_labels.append(choice_label)
            # plot on map with legend label instead of per-point text
            fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
            ax_map.scatter(pos[0], pos[1], s=80, marker='o', label=choice_label)
            ax_map.arrow(pos[0], pos[1], fwd[0]*0.3, fwd[1]*0.3, head_width=0.05, color='k')
        ax_map.set_title(item.get('question',''))
    elif qtype == 'distance_mca':
        meta = item.get('meta', {})
        # Only include pos-based
        start = np.array(meta.get('start_pos', [0,0,0]), dtype=float)
        end = np.array(meta.get('end_pos', [0,0,0]), dtype=float)
        start_t = np.array(meta.get('start_target', start + np.array([0,1,0])), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0,1,0])), dtype=float)
        thumbs.append({'position': start, 'target': start_t, 'render': None})
        thumb_labels.append('start')
        thumbs.append({'position': end, 'target': end_t, 'render': None})
        thumb_labels.append('end')
        # plot on map
        start = np.array(meta.get('start_pos', [0,0,0]), dtype=float)
        end = np.array(meta.get('end_pos', [0,0,0]), dtype=float)
        start_t = np.array(meta.get('start_target', start + np.array([0,1,0])), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0,1,0])), dtype=float)
        ax_map.scatter([start[0]], [start[1]], c='green', s=150, label='start')
        ax_map.scatter([end[0]], [end[1]], c='blue', s=150, label='end')
        fwd_s = start_t - start; fwd_s = fwd_s / (np.linalg.norm(fwd_s)+1e-8)
        fwd_e = end_t - end; fwd_e = fwd_e / (np.linalg.norm(fwd_e)+1e-8)
        ax_map.arrow(start[0], start[1], fwd_s[0]*0.4, fwd_s[1]*0.4, head_width=0.05, color='green')
        ax_map.arrow(end[0], end[1], fwd_e[0]*0.4, fwd_e[1]*0.4, head_width=0.05, color='blue')
        ax_map.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.6)
        ax_map.set_title(item.get('question',''))
        # Prepare choices text (A: xxm ...); highlight correct
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer', '')
        lines = []
        for lab, ch in zip(labels, choices):
            if lab == ans:
                lines.append(f"{lab}: {ch}  ✓")
            else:
                lines.append(f"{lab}: {ch}")
        choices_text = "Choices:\n" + "\n".join(lines)
    elif qtype == 'object_count_mca':
        meta = item.get('meta', {})
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        # camera
        thumbs.append({'position': pos, 'target': tgt, 'render': None})
        thumb_labels.append('view')
        fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
        ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
        ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # highlight category instances
        cat = str(meta.get('category','')).lower()
        if sc is not None:
            xs, ys = [], []
            for oid, obj in obj_lookup.items():
                if str(obj.label).lower() == cat:
                    xs.append(float(obj.position[0])); ys.append(float(obj.position[1]))
            if xs:
                ax_map.scatter(xs, ys, s=80, marker='s', c='tab:purple', alpha=0.8, label=f"{cat}")
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer', '')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'nearest_object_mca':
        meta = item.get('meta', {})
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        thumbs.append({'position': pos, 'target': tgt, 'render': None})
        thumb_labels.append('view')
        fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
        ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
        ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # plot candidate labels A/B/C at centers
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        vis = meta.get('visible_instances', [])
        for lab, vi in zip(labels, vis):
            c = np.array(vi.get('center', [np.nan,np.nan,np.nan]), dtype=float)
            ax_map.scatter(c[0], c[1], s=80, marker='^', label=f"{lab}:{vi.get('label','')}")
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype in ('relative_direction_mca', 'relative_position_mca'):
        meta = item.get('meta', {})
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        thumbs.append({'position': pos, 'target': tgt, 'render': None})
        thumb_labels.append('view')
        fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
        ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
        ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # plot target object center
        obj_id = str(meta.get('object_id',''))
        if obj_id in obj_lookup:
            oc = obj_lookup[obj_id].position
            ax_map.scatter(float(oc[0]), float(oc[1]), s=120, marker='*', c='tab:red', label='target')
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'object_object_distance_mca':
        meta = item.get('meta', {})
        # camera for context
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        if np.isfinite(pos).all():
            thumbs.append({'position': pos, 'target': tgt, 'render': None})
            thumb_labels.append('view')
            fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
            ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
            ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # plot objects A and B centers if found
        oa = str(meta.get('object_a',''))
        ob = str(meta.get('object_b',''))
        if oa in obj_lookup:
            ca = obj_lookup[oa].position
            ax_map.scatter(float(ca[0]), float(ca[1]), s=120, marker='s', c='tab:green', label=f"A: {meta.get('label_a','')}")
        if ob in obj_lookup:
            cb = obj_lookup[ob].position
            ax_map.scatter(float(cb[0]), float(cb[1]), s=120, marker='s', c='tab:blue', label=f"B: {meta.get('label_b','')}")
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'object_size_mca':
        meta = item.get('meta', {})
        # render view if camera provided
        pos = np.array(meta.get('camera_pos', [np.nan, np.nan, np.nan]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        if np.isfinite(pos).all():
            thumbs.append({'position': pos, 'target': tgt, 'render': meta.get('render')})
            thumb_labels.append('view')
            fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
            ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
            ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # mark object center if available
        oid = str(meta.get('object_id',''))
        if oid in obj_lookup:
            oc = obj_lookup[oid].position
            ax_map.scatter(float(oc[0]), float(oc[1]), s=140, marker='D', c='tab:purple', label=f"{meta.get('label','object')}")
        # choices text (in cm)
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch} cm{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'chain_position_reasoning_mca':
        # Camera view thumbnail + map markers for three objects and a North arrow
        meta = item.get('meta', {})
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        # add camera view to thumbs
        thumbs.append({'position': pos, 'target': tgt, 'render': None})
        thumb_labels.append('view')
        fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
        ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
        ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        # plot A/B/C object centers if available
        objs = meta.get('objects', [])
        marks = ['A','B','C']
        colors = ['tab:green','tab:blue','tab:red']
        for i, ob in enumerate(objs[:3]):
            c = np.array(ob.get('center', [np.nan, np.nan, np.nan]), dtype=float)
            if np.isfinite(c).all():
                disp = ob.get('display_label', ob.get('label', ''))
                ax_map.scatter(c[0], c[1], s=120, marker='s', c=colors[i % len(colors)], label=f"{marks[i]}: {disp}")
        # draw North arrow from a small anchor near map bounds
        nxy = np.array(meta.get('north_xy', [0.0,1.0]), dtype=float)
        # pick an anchor near lower-left of current scatter (use camera pos if finite)
        anchor = pos[:2]
        north_end = anchor + 0.8 * nxy
        ax_map.arrow(anchor[0], anchor[1], 0.8*nxy[0], 0.8*nxy[1], head_width=0.06, color='k')
        ax_map.text(anchor[0]+0.85*nxy[0], anchor[1]+0.85*nxy[1], 'N', fontsize=10, color='k', ha='center', va='center')
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D','E','F','G','H'][:len(choices)]
        ans = item.get('answer','')
        lines = []
        for lab, ch in zip(labels, choices):
            mark = '  ✓' if lab == ans else ''
            lines.append(f"{lab}: {ch}{mark}")
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'action_to_target_mca':
        meta = item.get('meta', {})
        pos = np.array(meta.get('camera_pos', [0,0,0]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0,1,0])), dtype=float)
        thumbs.append({'position': pos, 'target': tgt, 'render': None})
        thumb_labels.append('view')
        fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd)+1e-8)
        ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
        ax_map.arrow(pos[0], pos[1], fwd[0]*0.35, fwd[1]*0.35, head_width=0.06, color='tab:orange')
        tid = str(meta.get('target_id',''))
        if tid in obj_lookup:
            oc = obj_lookup[tid].position
            ax_map.scatter(float(oc[0]), float(oc[1]), s=140, marker='*', c='tab:red', label='target')
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    elif qtype == 'frame_frame_action_mca':
        # Show begin/current view and end view as thumbnails and map markers.
        meta = item.get('meta', {})
        # support both naming conventions: begin_* or current_*
        begin = np.array(meta.get('current_pos', meta.get('begin_pos', [0,0,0])), dtype=float)
        begin_t = np.array(meta.get('current_target', meta.get('begin_target', begin + np.array([0,1,0]))), dtype=float)
        end = np.array(meta.get('end_pos', [0,0,0]), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0,1,0])), dtype=float)
        thumbs.append({'position': begin, 'target': begin_t, 'render': None})
        thumb_labels.append('begin')
        thumbs.append({'position': end, 'target': end_t, 'render': None})
        thumb_labels.append('end')
        # plot on map
        ax_map.scatter([begin[0]], [begin[1]], c='green', s=150, label='begin')
        ax_map.scatter([end[0]], [end[1]], c='blue', s=150, label='end')
        fwd_b = begin_t - begin; fwd_b = fwd_b / (np.linalg.norm(fwd_b)+1e-8)
        fwd_e = end_t - end; fwd_e = fwd_e / (np.linalg.norm(fwd_e)+1e-8)
        ax_map.arrow(begin[0], begin[1], fwd_b[0]*0.4, fwd_b[1]*0.4, head_width=0.05, color='green')
        ax_map.arrow(end[0], end[1], fwd_e[0]*0.4, fwd_e[1]*0.4, head_width=0.05, color='blue')
        ax_map.plot([begin[0], end[0]], [begin[1], end[1]], 'k--', alpha=0.6)
        ax_map.set_title(item.get('question',''))
        # Prepare choices text from meta.choices_map (action names)
        choices = item.get('meta', {}).get('choices_map', [])
        if not choices:
            choices = ['move_forward','move_backward','turn_left','turn_right']
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = []
        for lab, ch in zip(labels, choices):
            display = ch.replace('_',' ')
            if lab == ans:
                lines.append(f"{lab}: {display}  ✓")
            else:
                lines.append(f"{lab}: {display}")
        choices_text = "Choices:\n" + "\n".join(lines)
    elif qtype == 'object_after_rotation_mca':
        # Show begin view and end view, similar to frame_frame_action_mca
        meta = item.get('meta', {})
        begin = np.array(meta.get('begin_pos', [0, 0, 0]), dtype=float)
        begin_t = np.array(meta.get('begin_target', begin + np.array([0, 1, 0])), dtype=float)
        end = np.array(meta.get('end_pos', [0, 0, 0]), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0, 1, 0])), dtype=float)
        thumbs.append({'position': begin, 'target': begin_t, 'render': None})
        thumb_labels.append('begin')
        thumbs.append({'position': end, 'target': end_t, 'render': None})
        thumb_labels.append('end')
        # plot on map
        ax_map.scatter([begin[0]], [begin[1]], c='green', s=150, label='begin')
        ax_map.scatter([end[0]], [end[1]], c='blue', s=150, label='end')
        fwd_b = begin_t - begin; fwd_b = fwd_b / (np.linalg.norm(fwd_b) + 1e-8)
        fwd_e = end_t - end; fwd_e = fwd_e / (np.linalg.norm(fwd_e) + 1e-8)
        ax_map.arrow(begin[0], begin[1], fwd_b[0] * 0.4, fwd_b[1] * 0.4, head_width=0.05, color='green')
        ax_map.arrow(end[0], end[1], fwd_e[0] * 0.4, fwd_e[1] * 0.4, head_width=0.05, color='blue')
        ax_map.plot([begin[0], end[0]], [begin[1], end[1]], 'k--', alpha=0.6)
        ax_map.set_title(item.get('question', ''))
        # Prepare choices text
        choices = item.get('choices', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices)]
        ans = item.get('answer', '')
        # find the human-readable choice text for the correct answer (if present)
        correct_choice_text = ''
        try:
            if ans in labels:
                correct_choice_text = choices[labels.index(ans)]
        except Exception:
            correct_choice_text = ''
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        # Prepend an explicit Correct line for visibility
        if correct_choice_text:
            choices_text = f"Correct: {ans}: {correct_choice_text}\nChoices:\n" + "\n".join(lines)
        else:
            choices_text = "Choices:\n" + "\n".join(lines)
    elif qtype == 'middle_frame_mca':
        # Follow the common multi-thumb rendering flow: populate thumbs and let the
        # generic layout/rendering code do the rest (no manual canvas operations).
        meta = item.get('meta', {})
        im1 = meta.get('image1', {})
        im3 = meta.get('image3', {})
        # add Image1 and Image3 first
        pos1 = np.array(im1.get('position', [np.nan, np.nan, np.nan]), dtype=float)
        tgt1 = np.array(im1.get('target', pos1 + np.array([0, 1, 0])), dtype=float)
        pos3 = np.array(im3.get('position', [np.nan, np.nan, np.nan]), dtype=float)
        tgt3 = np.array(im3.get('target', pos3 + np.array([0, 1, 0])), dtype=float)
        if np.isfinite(pos1).all():
            thumbs.append({'position': pos1, 'target': tgt1, 'render': None})
            thumb_labels.append('Image1')
            # also plot on map
            ax_map.scatter([pos1[0]], [pos1[1]], c='green', s=150, label='Image1')
            fwd1 = tgt1 - pos1; fwd1 = fwd1 / (np.linalg.norm(fwd1) + 1e-8)
            ax_map.arrow(pos1[0], pos1[1], fwd1[0] * 0.35, fwd1[1] * 0.35, head_width=0.06, color='green')
        if np.isfinite(pos3).all():
            thumbs.append({'position': pos3, 'target': tgt3, 'render': None})
            thumb_labels.append('Image3')
            ax_map.scatter([pos3[0]], [pos3[1]], c='blue', s=150, label='Image3')
            fwd3 = tgt3 - pos3; fwd3 = fwd3 / (np.linalg.norm(fwd3) + 1e-8)
            ax_map.arrow(pos3[0], pos3[1], fwd3[0] * 0.35, fwd3[1] * 0.35, head_width=0.06, color='blue')

        # then append A/B/C/D options
        choices_cfgs = meta.get('choices_map', [])
        choices_texts = meta.get('choices_texts', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices_cfgs)]
        for i, cfg in enumerate(choices_cfgs):
            p = np.array(cfg.get('position', [np.nan, np.nan, np.nan]), dtype=float)
            t = np.array(cfg.get('target', p + np.array([0, 1, 0])), dtype=float)
            if np.isfinite(p).all():
                thumbs.append({'position': p, 'target': t, 'render': None})
                thumb_labels.append(labels[i])
                # mark on map with small marker (optional)
                ax_map.scatter(p[0], p[1], s=60, marker='o', alpha=0.6, label=labels[i])

        # build choices text block with correct marking
        ans = item.get('answer', '')
        lines = []
        for i, lab in enumerate(labels):
            txt = choices_texts[i] if i < len(choices_texts) else lab
            mark = '  \u2713' if lab == ans else ''
            lines.append(f"{lab}: {txt}{mark}")
        if lines:
            choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question', ''))

    elif qtype == 'multi_step_navigation_mca':
        # Show camera view and plot the four candidate action trajectories on the map.
        meta = item.get('meta', {})
        cam_pos = np.array(meta.get('camera_pos', [0, 0, 0]), dtype=float)
        cam_tgt = np.array(meta.get('camera_target', cam_pos + np.array([0, 1, 0])), dtype=float)
        # determine start object center and heading (heading stored in meta as start_heading)
        start_obj = meta.get('start_object')
        goal_obj = meta.get('goal_object')
        if start_obj is not None:
            start_pos = np.array(start_obj.get('center', cam_pos), dtype=float)
        else:
            start_pos = cam_pos
        start_heading = np.array(meta.get('start_heading', cam_tgt - cam_pos), dtype=float)
        if np.linalg.norm(start_heading) < 1e-8:
            start_heading = cam_tgt - cam_pos
        fwd = start_heading; fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

        # show camera view thumbnail (optional reference) and mark it as current view
        thumbs.append({'position': cam_pos, 'target': cam_tgt, 'render': None})
        thumb_labels.append('current_view')
        # also plot camera position and heading on the top-down map
        try:
            cam_fwd = cam_tgt - cam_pos
            cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd) + 1e-8)
            ax_map.scatter(cam_pos[0], cam_pos[1], s=120, c='tab:orange', marker='o', label='current_view')
            ax_map.arrow(cam_pos[0], cam_pos[1], cam_fwd[0]*0.35, cam_fwd[1]*0.35, head_width=0.06, color='tab:orange')
        except Exception:
            pass

        # Draw room polygons and wall outlines on the main map for context
        try:
            from Data_generation.bench_generation.batch_utils import load_scene_wall_aabbs, load_room_polys
            # walls as rectangles
            try:
                walls = load_scene_wall_aabbs(str(scene_path))
                for w in walls:
                    try:
                        xmin, ymin, _ = w.bmin
                        xmax, ymax, _ = w.bmax
                        wx = [xmin, xmin, xmax, xmax, xmin]
                        wy = [ymin, ymax, ymax, ymin, ymin]
                        ax_map.plot(wx, wy, color='dimgray', linewidth=1.0, alpha=0.6)
                    except Exception:
                        continue
            except Exception:
                pass

            # room polygons
            try:
                room_polys = load_room_polys(str(scene_path))
                for poly in room_polys:
                    try:
                        ax_map.plot(poly[:,0], poly[:,1], color='gray', linewidth=1.0, alpha=0.5)
                    except Exception:
                        continue
            except Exception:
                pass
        except Exception:
            pass

        # start object position and heading are shown below using green markers/arrows
        # mark start/goal object centers if available
        try:
            if start_obj is not None:
                sc_pos = np.array(start_obj.get('center', [np.nan, np.nan, np.nan]), dtype=float)
                if np.isfinite(sc_pos).all():
                    lab = start_obj.get('display_label', start_obj.get('label', 'start'))
                    ax_map.scatter(float(sc_pos[0]), float(sc_pos[1]), s=120, marker='s', c='tab:green', label=f"Start: {lab}")
                    # draw heading arrow at the start object using stored start_heading
                    try:
                        sh = np.array(meta.get('start_heading', fwd), dtype=float)
                        if np.linalg.norm(sh) > 1e-8:
                            sh = sh / (np.linalg.norm(sh) + 1e-8)
                            ax_map.arrow(sc_pos[0], sc_pos[1], sh[0]*0.35, sh[1]*0.35, head_width=0.06, color='tab:green')
                    except Exception:
                        pass
            if goal_obj is not None:
                gc_pos = np.array(goal_obj.get('center', [np.nan, np.nan, np.nan]), dtype=float)
                if np.isfinite(gc_pos).all():
                    lab = goal_obj.get('display_label', goal_obj.get('label', 'goal'))
                    ax_map.scatter(float(gc_pos[0]), float(gc_pos[1]), s=120, marker='*', c='tab:red', label=f"Goal: {lab}")
        except Exception:
            pass

        # helper to apply actions and return sequence of XY points (start + after each move)
        def _apply_points(start_pos, start_heading_xy, actions):
            pts = [np.array(start_pos[:2], dtype=float).copy()]
            h = np.array(start_heading_xy, dtype=float)
            # normalize
            n = np.linalg.norm(h)
            if n < 1e-8:
                h = np.array([1.0, 0.0], dtype=float)
            else:
                h = h / n
            pos_local = np.array(start_pos, dtype=float).copy()
            for a in actions:
                if a.get('type') == 'turn':
                    ang = math.radians(float(a.get('angle', 0)))
                    c = math.cos(ang); s = math.sin(ang)
                    hx, hy = h[0], h[1]
                    h = np.array([c * hx - s * hy, s * hx + c * hy], dtype=float)
                    # renormalize
                    hn = np.linalg.norm(h)
                    if hn > 1e-8:
                        h = h / hn
                elif a.get('type') == 'move':
                    d = float(a.get('distance', 0.0))
                    pos_local[0] += h[0] * d
                    pos_local[1] += h[1] * d
                    pts.append(np.array(pos_local[:2], dtype=float).copy())
            return pts, pos_local, h

        choices_actions = meta.get('choices_actions', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices_actions)]
        colors = ['tab:green', 'tab:blue', 'tab:purple', 'tab:cyan']
        ans = item.get('answer', '')
        lines = []
        # (duplicate markers removed; start/goal markers with display labels are handled above)

        for i, acts in enumerate(choices_actions):
            try:
                pts, final_pos, final_h = _apply_points(start_pos, fwd, acts)
            except Exception:
                pts = [start_pos[:2]]
                final_pos = start_pos.copy()
                final_h = fwd

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            col = colors[i % len(colors)]
            ax_map.plot(xs, ys, color=col, linewidth=2.0, alpha=0.9)
            # mark end with label
            ax_map.scatter(xs[-1], ys[-1], s=100, c=col, marker='o', label=f"{labels[i]}{' ✓' if labels[i]==ans else ''}")
            ax_map.text(xs[-1], ys[-1], labels[i], fontsize=10, fontweight='bold', ha='center', va='center')

            # Do not render thumbnails for each option. The preview shows the
            # top-down trajectories and marks the final positions on the map.

            # prepare textual choices list
            ch_text = item.get('choices', [])
            txt = ch_text[i] if i < len(ch_text) else labels[i]
            mark = '  ✓' if labels[i] == ans else ''
            lines.append(f"{labels[i]}: {txt}{mark}")

        if lines:
            choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question', ''))

    elif qtype == 'appearance_order_mca':
        # Draw a simple timeline of first appearance indices
        meta = item.get('meta', {})
        first_idx = meta.get('first_idx', {})
        # layout categories on a 1D axis
        if isinstance(first_idx, dict) and first_idx:
            xs = []
            labels_txt = []
            for lab, idx in first_idx.items():
                try:
                    xs.append(float(idx))
                    labels_txt.append(str(lab))
                except Exception:
                    continue
            if xs:
                y0 = 0.0
                ax_map.hlines(y0, min(xs)-0.5, max(xs)+0.5, colors='gray', linestyles='--', alpha=0.5)
                for x, t in zip(xs, labels_txt):
                    ax_map.plot([x], [y0], 'o', c='tab:blue')
                    ax_map.text(x, y0+0.05, t, ha='center', va='bottom', fontsize=9)
                ax_map.set_ylim(-0.2, 1.0)
                ax_map.set_xlabel('path index (first appearance)')
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    else:
        ax_map.text(0.5, 0.5, f'Unsupported qtype: {qtype}', ha='center')

    # Add annotations to the map if provided
    if annotations:
        for annotation in annotations:
            label = annotation.get('label')
            position = annotation.get('position')
            if label and position:
                ax_map.text(position[0], position[1], label, fontsize=8, color='red', ha='center', va='center')

    # render thumbnails (attempt) for all thumbs
    imgs = []
    for p in thumbs:
        # Always render strictly from position/target (ignore provided render intrinsics)
        img = render_thumbnail_for_pose(scene_path, p, thumb_size=thumb_size)
        imgs.append(img)

    # layout thumbnails in a small grid and label them (A,B,C... or start/end)
    n = len(imgs)
    if n == 0:
        ax_thumb.text(0.5,0.5,'No thumbnail',ha='center')
    else:
        # Choose grid shape
        if n == 1:
            cols = 1
        elif n == 2:
            cols = 2
        else:
            cols = min(3, n)
        rows = int(math.ceil(n / cols))

        # remove the single ax_thumb; we'll place sub-axes inside its bbox
        bbox = ax_thumb.get_position()
        fig.delaxes(ax_thumb)

        pad = 0.02
        thumb_w = (bbox.width - pad*(cols-1)) / cols
        thumb_h = (bbox.height - pad*(rows-1)) / rows

        for idx, img in enumerate(imgs):
            r = idx // cols
            c = idx % cols
            left = bbox.x0 + c * (thumb_w + pad)
            bottom = bbox.y1 - (r+1) * thumb_h - r * pad
            ax_i = fig.add_axes([left, bottom, thumb_w, thumb_h])
            # show image
            if img is None:
                ax_i.text(0.5,0.5,'No img',ha='center')
            else:
                try:
                    ax_i.imshow(img)
                except Exception:
                    img2 = np.array(img, dtype=np.uint8)
                    ax_i.imshow(img2)
            ax_i.axis('off')
            # label (keep fully within axes to avoid clipping)
            label = thumb_labels[idx] if idx < len(thumb_labels) else chr(ord('A')+idx)
            ax_i.text(
                0.02, 0.98, label,
                transform=ax_i.transAxes,
                fontsize=12, fontweight='bold', color='yellow',
                ha='left', va='top', clip_on=False,
                bbox=dict(facecolor='black', alpha=0.6, pad=2, boxstyle='round,pad=0.2')
            )

        # If choices text is present, place it on the RIGHT side (thumbnail area)
        if choices_text:
            # When there are no thumbnails, ax_thumb still exists: draw inside it
            if n == 0:
                try:
                    ax_thumb.text(
                        0.98, 0.98, choices_text,
                        transform=ax_thumb.transAxes,
                        fontsize=11, family='monospace', va='top', ha='right',
                        bbox=dict(facecolor='white', alpha=0.95, pad=6, edgecolor='none')
                    )
                except Exception:
                    fig.text(0.75, 0.95, choices_text, fontsize=11, family='monospace', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.95, pad=6))
            else:
                # Place as figure-level text to the immediate right of the thumbnails bbox
                x = min(bbox.x1 + 0.01, 0.98)
                y = min(bbox.y1 + 0.02, 0.98)
                fig.text(x, y, choices_text, transform=fig.transFigure, fontsize=11, family='monospace', va='top', ha='left', bbox=dict(facecolor='white', alpha=0.95, pad=6, edgecolor='none'))

    ax_map.set_aspect('equal')
    # Place legend to the side of the map, grouped with objects/current_view/A/B/C
    # Place legend in the lower-left of the map (above choices text) to avoid overlapping thumbnails
    try:
        ax_map.legend(loc='lower left', bbox_to_anchor=(0.01, 0.12), bbox_transform=ax_map.transAxes, fontsize='small', framealpha=0.9)
    except Exception:
        ax_map.legend(loc='lower left', fontsize='small', framealpha=0.9)
    # generous margins so labels and legends are not cropped
    fig.subplots_adjust(left=0.06, right=0.985, top=0.955, bottom=0.08, wspace=0.25)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def compose_view_map(item: dict, scene_path: Path, out_path: Path, figsize: tuple[float, float] = (6, 6), thumb_size: int = 256):
    """Save a view image for a QA item.

    Behavior:
    - If the item meta contains a valid camera position/target, attempt to render a
      thumbnail using the scene renderer (`render_thumbnail_for_pose`). This will
      produce a scene-rendered image (preferred).
    - If no camera info is available, fall back to saving a top-down map layout
      (previous behavior).

    Note: rendering requires the full rendering stack (gsplat/torch/ply loader). If
    rendering is requested but the rendering call fails, the exception will propagate
    (so failures are visible to the caller).
    """
    meta = item.get('meta', {})
    pos = np.array(meta.get('camera_pos', [np.nan, np.nan, np.nan]), dtype=float)
    tgt = np.array(meta.get('camera_target', pos + np.array([0, 1, 0])), dtype=float)

    # If camera info present, render using the renderer (this may raise on missing deps)
    if np.isfinite(pos).all():
        # render thumbnail using existing helper (may raise if renderer/deps missing)
        img = render_thumbnail_for_pose(scene_path, {'position': pos, 'target': tgt}, thumb_size=thumb_size)
        # save image
        imageio.imwrite(str(out_path), img)
        return

    # Fallback: layout/map-only visualization
    # When a thumb_size is provided, prefer producing a compact square image
    # whose pixel height matches `thumb_size` so that `view.png` visually
    # matches the rendered thumbnails used in previews.
    if thumb_size is not None:
        # choose dpi so that figsize * dpi == thumb_size (we pick dpi=100)
        dpi = 100
        fig = plt.figure(figsize=(thumb_size / dpi, thumb_size / dpi))
        ax_map = fig.add_subplot(1, 1, 1)
        save_kwargs = {'dpi': dpi}
    else:
        fig = plt.figure(figsize=figsize)
        ax_map = fig.add_subplot(1, 1, 1)
        save_kwargs = {'dpi': 180}

    sc = None
    obj_lookup = {}
    if SemanticCamera is not None and scene_path.exists():
        sc = SemanticCamera(str(scene_path))
        objs = sc.list_objects()
        xs = []
        ys = []
        for oid in list(objs.keys())[:500]:
            obj = sc.get_object(oid)
            if obj is None:
                continue
            obj_lookup[str(oid)] = obj
            xs.append(float(obj.position[0]))
            ys.append(float(obj.position[1]))
        if xs:
            ax_map.scatter(xs, ys, c='lightgray', s=8, alpha=0.6, label='objects')

    qtype = item.get('qtype')
    # populate map markers depending on question type (subset of compose_preview_for_item logic)
    if qtype == 'distance_mca':
        start = np.array(meta.get('start_pos', [0, 0, 0]), dtype=float)
        end = np.array(meta.get('end_pos', [0, 0, 0]), dtype=float)
        start_t = np.array(meta.get('start_target', start + np.array([0, 1, 0])), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0, 1, 0])), dtype=float)
        ax_map.scatter([start[0]], [start[1]], c='green', s=150, label='start')
        ax_map.scatter([end[0]], [end[1]], c='blue', s=150, label='end')
        fwd_s = start_t - start; fwd_s = fwd_s / (np.linalg.norm(fwd_s) + 1e-8)
        fwd_e = end_t - end; fwd_e = fwd_e / (np.linalg.norm(fwd_e) + 1e-8)
        ax_map.arrow(start[0], start[1], fwd_s[0] * 0.4, fwd_s[1] * 0.4, head_width=0.05, color='green')
        ax_map.arrow(end[0], end[1], fwd_e[0] * 0.4, fwd_e[1] * 0.4, head_width=0.05, color='blue')
        ax_map.plot([start[0], end[0]], [start[1], end[1]], 'k--', alpha=0.6)
        ax_map.set_title(item.get('question', ''))
    elif qtype == 'object_size_mca':
        pos = np.array(meta.get('camera_pos', [np.nan, np.nan, np.nan]), dtype=float)
        tgt = np.array(meta.get('camera_target', pos + np.array([0, 1, 0])), dtype=float)
        if np.isfinite(pos).all():
            ax_map.scatter(pos[0], pos[1], s=120, c='tab:orange', marker='o', label='view')
            fwd = tgt - pos; fwd = fwd / (np.linalg.norm(fwd) + 1e-8)
            ax_map.arrow(pos[0], pos[1], fwd[0] * 0.35, fwd[1] * 0.35, head_width=0.06, color='tab:orange')
        oid = str(meta.get('object_id', ''))
        if oid in obj_lookup:
            oc = obj_lookup[oid].position
            ax_map.scatter(float(oc[0]), float(oc[1]), s=140, marker='D', c='tab:purple', label=f"{meta.get('label','object')}")
        ax_map.set_title(item.get('question', ''))
    elif qtype == 'object_after_rotation_mca':
        # draw begin and end camera positions and forward arrows
        begin = np.array(meta.get('begin_pos', [0, 0, 0]), dtype=float)
        begin_t = np.array(meta.get('begin_target', begin + np.array([0, 1, 0])), dtype=float)
        end = np.array(meta.get('end_pos', [0, 0, 0]), dtype=float)
        end_t = np.array(meta.get('end_target', end + np.array([0, 1, 0])), dtype=float)
        ax_map.scatter([begin[0]], [begin[1]], c='green', s=150, label='begin')
        ax_map.scatter([end[0]], [end[1]], c='blue', s=150, label='end')
        fwd_b = begin_t - begin; fwd_b = fwd_b / (np.linalg.norm(fwd_b) + 1e-8)
        fwd_e = end_t - end; fwd_e = fwd_e / (np.linalg.norm(fwd_e) + 1e-8)
        ax_map.arrow(begin[0], begin[1], fwd_b[0] * 0.4, fwd_b[1] * 0.4, head_width=0.05, color='green')
        ax_map.arrow(end[0], end[1], fwd_e[0] * 0.4, fwd_e[1] * 0.4, head_width=0.05, color='blue')
        ax_map.plot([begin[0], end[0]], [begin[1], end[1]], 'k--', alpha=0.6)
        ax_map.set_title(item.get('question', ''))
        # Prepare choices text
        choices = item.get('choices', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices)]
        ans = item.get('answer', '')
        # find the human-readable choice text for the correct answer (if present)
        correct_choice_text = ''
        try:
            if ans in labels:
                correct_choice_text = choices[labels.index(ans)]
        except Exception:
            correct_choice_text = ''
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        # Prepend an explicit Correct line for visibility
        if correct_choice_text:
            choices_text = f"Correct: {ans}: {correct_choice_text}\nChoices:\n" + "\n".join(lines)
        else:
            choices_text = "Choices:\n" + "\n".join(lines)
    elif qtype == 'middle_frame_mca':
        # Follow the common multi-thumb rendering flow: populate thumbs and let the
        # generic layout/rendering code do the rest (no manual canvas operations).
        meta = item.get('meta', {})
        im1 = meta.get('image1', {})
        im3 = meta.get('image3', {})
        # add Image1 and Image3 first
        pos1 = np.array(im1.get('position', [np.nan, np.nan, np.nan]), dtype=float)
        tgt1 = np.array(im1.get('target', pos1 + np.array([0, 1, 0])), dtype=float)
        pos3 = np.array(im3.get('position', [np.nan, np.nan, np.nan]), dtype=float)
        tgt3 = np.array(im3.get('target', pos3 + np.array([0, 1, 0])), dtype=float)
        if np.isfinite(pos1).all():
            thumbs.append({'position': pos1, 'target': tgt1, 'render': None})
            thumb_labels.append('Image1')
            # also plot on map
            ax_map.scatter([pos1[0]], [pos1[1]], c='green', s=150, label='Image1')
            fwd1 = tgt1 - pos1; fwd1 = fwd1 / (np.linalg.norm(fwd1) + 1e-8)
            ax_map.arrow(pos1[0], pos1[1], fwd1[0] * 0.35, fwd1[1] * 0.35, head_width=0.06, color='green')
        if np.isfinite(pos3).all():
            thumbs.append({'position': pos3, 'target': tgt3, 'render': None})
            thumb_labels.append('Image3')
            ax_map.scatter([pos3[0]], [pos3[1]], c='blue', s=150, label='Image3')
            fwd3 = tgt3 - pos3; fwd3 = fwd3 / (np.linalg.norm(fwd3) + 1e-8)
            ax_map.arrow(pos3[0], pos3[1], fwd3[0] * 0.35, fwd3[1] * 0.35, head_width=0.06, color='blue')

        # then append A/B/C/D options
        choices_cfgs = meta.get('choices_map', [])
        choices_texts = meta.get('choices_texts', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices_cfgs)]
        for i, cfg in enumerate(choices_cfgs):
            p = np.array(cfg.get('position', [np.nan, np.nan, np.nan]), dtype=float)
            t = np.array(cfg.get('target', p + np.array([0, 1, 0])), dtype=float)
            if np.isfinite(p).all():
                thumbs.append({'position': p, 'target': t, 'render': None})
                thumb_labels.append(labels[i])
                # mark on map with small marker (optional)
                ax_map.scatter(p[0], p[1], s=60, marker='o', alpha=0.6, label=labels[i])

        # build choices text block with correct marking
        ans = item.get('answer', '')
        lines = []
        for i, lab in enumerate(labels):
            txt = choices_texts[i] if i < len(choices_texts) else lab
            mark = '  \u2713' if lab == ans else ''
            lines.append(f"{lab}: {txt}{mark}")
        if lines:
            choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question', ''))

    elif qtype == 'multi_step_navigation_mca':
        # Show camera view and plot the four candidate action trajectories on the map.
        meta = item.get('meta', {})
        cam_pos = np.array(meta.get('camera_pos', [0, 0, 0]), dtype=float)
        cam_tgt = np.array(meta.get('camera_target', cam_pos + np.array([0, 1, 0])), dtype=float)
        # determine start object center and heading (heading stored in meta as start_heading)
        start_obj = meta.get('start_object')
        goal_obj = meta.get('goal_object')
        if start_obj is not None:
            start_pos = np.array(start_obj.get('center', cam_pos), dtype=float)
        else:
            start_pos = cam_pos
        start_heading = np.array(meta.get('start_heading', cam_tgt - cam_pos), dtype=float)
        if np.linalg.norm(start_heading) < 1e-8:
            start_heading = cam_tgt - cam_pos
        fwd = start_heading; fwd = fwd / (np.linalg.norm(fwd) + 1e-8)

        # show camera view thumbnail (optional reference) and mark it as current view
        thumbs.append({'position': cam_pos, 'target': cam_tgt, 'render': None})
        thumb_labels.append('current_view')
        # also plot camera position and heading on the top-down map
        try:
            cam_fwd = cam_tgt - cam_pos
            cam_fwd = cam_fwd / (np.linalg.norm(cam_fwd) + 1e-8)
            ax_map.scatter(cam_pos[0], cam_pos[1], s=120, c='tab:orange', marker='o', label='current_view')
            ax_map.arrow(cam_pos[0], cam_pos[1], cam_fwd[0]*0.35, cam_fwd[1]*0.35, head_width=0.06, color='tab:orange')
        except Exception:
            pass

        # Draw room polygons and wall outlines on the main map for context
        try:
            from Data_generation.bench_generation.batch_utils import load_scene_wall_aabbs, load_room_polys
            # walls as rectangles
            try:
                walls = load_scene_wall_aabbs(str(scene_path))
                for w in walls:
                    try:
                        xmin, ymin, _ = w.bmin
                        xmax, ymax, _ = w.bmax
                        wx = [xmin, xmin, xmax, xmax, xmin]
                        wy = [ymin, ymax, ymax, ymin, ymin]
                        ax_map.plot(wx, wy, color='dimgray', linewidth=1.0, alpha=0.6)
                    except Exception:
                        continue
            except Exception:
                pass

            # room polygons
            try:
                room_polys = load_room_polys(str(scene_path))
                for poly in room_polys:
                    try:
                        ax_map.plot(poly[:,0], poly[:,1], color='gray', linewidth=1.0, alpha=0.5)
                    except Exception:
                        continue
            except Exception:
                pass
        except Exception:
            pass

        # start object position and heading are shown below using green markers/arrows
        # mark start/goal object centers if available
        try:
            if start_obj is not None:
                sc_pos = np.array(start_obj.get('center', [np.nan, np.nan, np.nan]), dtype=float)
                if np.isfinite(sc_pos).all():
                    lab = start_obj.get('display_label', start_obj.get('label', 'start'))
                    ax_map.scatter(float(sc_pos[0]), float(sc_pos[1]), s=120, marker='s', c='tab:green', label=f"Start: {lab}")
                    # draw heading arrow at the start object using stored start_heading
                    try:
                        sh = np.array(meta.get('start_heading', fwd), dtype=float)
                        if np.linalg.norm(sh) > 1e-8:
                            sh = sh / (np.linalg.norm(sh) + 1e-8)
                            ax_map.arrow(sc_pos[0], sc_pos[1], sh[0]*0.35, sh[1]*0.35, head_width=0.06, color='tab:green')
                    except Exception:
                        pass
            if goal_obj is not None:
                gc_pos = np.array(goal_obj.get('center', [np.nan, np.nan, np.nan]), dtype=float)
                if np.isfinite(gc_pos).all():
                    lab = goal_obj.get('display_label', goal_obj.get('label', 'goal'))
                    ax_map.scatter(float(gc_pos[0]), float(gc_pos[1]), s=120, marker='*', c='tab:red', label=f"Goal: {lab}")
        except Exception:
            pass

        # helper to apply actions and return sequence of XY points (start + after each move)
        def _apply_points(start_pos, start_heading_xy, actions):
            pts = [np.array(start_pos[:2], dtype=float).copy()]
            h = np.array(start_heading_xy, dtype=float)
            # normalize
            n = np.linalg.norm(h)
            if n < 1e-8:
                h = np.array([1.0, 0.0], dtype=float)
            else:
                h = h / n
            pos_local = np.array(start_pos, dtype=float).copy()
            for a in actions:
                if a.get('type') == 'turn':
                    ang = math.radians(float(a.get('angle', 0)))
                    c = math.cos(ang); s = math.sin(ang)
                    hx, hy = h[0], h[1]
                    h = np.array([c * hx - s * hy, s * hx + c * hy], dtype=float)
                    # renormalize
                    hn = np.linalg.norm(h)
                    if hn > 1e-8:
                        h = h / hn
                elif a.get('type') == 'move':
                    d = float(a.get('distance', 0.0))
                    pos_local[0] += h[0] * d
                    pos_local[1] += h[1] * d
                    pts.append(np.array(pos_local[:2], dtype=float).copy())
            return pts, pos_local, h

        choices_actions = meta.get('choices_actions', [])
        labels = ['A', 'B', 'C', 'D'][:len(choices_actions)]
        colors = ['tab:green', 'tab:blue', 'tab:purple', 'tab:cyan']
        ans = item.get('answer', '')
        lines = []
        # (duplicate markers removed; start/goal markers with display labels are handled above)

        for i, acts in enumerate(choices_actions):
            try:
                pts, final_pos, final_h = _apply_points(start_pos, fwd, acts)
            except Exception:
                pts = [start_pos[:2]]
                final_pos = start_pos.copy()
                final_h = fwd

            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            col = colors[i % len(colors)]
            ax_map.plot(xs, ys, color=col, linewidth=2.0, alpha=0.9)
            # mark end with label
            ax_map.scatter(xs[-1], ys[-1], s=100, c=col, marker='o', label=f"{labels[i]}{' ✓' if labels[i]==ans else ''}")
            ax_map.text(xs[-1], ys[-1], labels[i], fontsize=10, fontweight='bold', ha='center', va='center')

            # Do not render thumbnails for each option. The preview shows the
            # top-down trajectories and marks the final positions on the map.

            # prepare textual choices list
            ch_text = item.get('choices', [])
            txt = ch_text[i] if i < len(ch_text) else labels[i]
            mark = '  ✓' if labels[i] == ans else ''
            lines.append(f"{labels[i]}: {txt}{mark}")

        if lines:
            choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question', ''))

    elif qtype == 'appearance_order_mca':
        # Draw a simple timeline of first appearance indices
        meta = item.get('meta', {})
        first_idx = meta.get('first_idx', {})
        # layout categories on a 1D axis
        if isinstance(first_idx, dict) and first_idx:
            xs = []
            labels_txt = []
            for lab, idx in first_idx.items():
                try:
                    xs.append(float(idx))
                    labels_txt.append(str(lab))
                except Exception:
                    continue
            if xs:
                y0 = 0.0
                ax_map.hlines(y0, min(xs)-0.5, max(xs)+0.5, colors='gray', linestyles='--', alpha=0.5)
                for x, t in zip(xs, labels_txt):
                    ax_map.plot([x], [y0], 'o', c='tab:blue')
                    ax_map.text(x, y0+0.05, t, ha='center', va='bottom', fontsize=9)
                ax_map.set_ylim(-0.2, 1.0)
                ax_map.set_xlabel('path index (first appearance)')
        # choices text
        choices = item.get('choices', [])
        labels = ['A','B','C','D'][:len(choices)]
        ans = item.get('answer','')
        lines = [f"{lab}: {ch}{'  ✓' if lab==ans else ''}" for lab, ch in zip(labels, choices)]
        choices_text = "Choices:\n" + "\n".join(lines)
        ax_map.set_title(item.get('question',''))
    else:
        ax_map.text(0.5, 0.5, f'Unsupported qtype: {qtype}', ha='center')

    # Add annotations to the map if provided
    if annotations:
        for annotation in annotations:
            label = annotation.get('label')
            position = annotation.get('position')
            if label and position:
                ax_map.text(position[0], position[1], label, fontsize=8, color='red', ha='center', va='center')

    ax_map.set_aspect('equal')
    try:
        ax_map.legend(loc='lower left', fontsize='small', framealpha=0.9)
    except Exception:
        ax_map.legend(loc='lower left', fontsize='small', framealpha=0.9)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.955, bottom=0.08)
    fig.savefig(str(out_path), bbox_inches='tight', **save_kwargs)
    plt.close(fig)


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
        # Use RGB colors directly
        colors = torch.from_numpy(gs_data.colors).float().to(device)

    return means, quats, scales, opacities, colors, sh_degree