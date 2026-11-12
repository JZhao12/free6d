"""
Template bank of the training-free 6D pose stage: per object, a set of views with dense
DINOv3 patch features and the model coordinates (mm) of every patch.

Pose sets:
- "seg42": the 42 level-0 segmentation templates read from disk (RGBA PNGs rendered by
  src/poses/pyrender.py); depth comes from re-rendering at the equivalent pose. If they do not
  match the renderer, the same 42 viewpoints are rendered in-process instead.
- "auto42" (the default of run_pose.py): like "seg42", but an object only shares the segmentation
  templates when none of them is cut by the image border (otherwise the in-process level-0
  templates are used).
- "<views>x<rolls>" (views in level0 / cube8 / oct14 / ico12): rendered in-process
  with a camera fitted to the model diameter, each view with `rolls` in-plane rotations.
"""
import itertools
import logging
import os
import os.path as osp
import re

import numpy as np
import torch
from PIL import Image

from src.pose.geometry import (
    PATCH,
    backproject_to_model,
    box_to_square_crop,
    crop_resize,
    mask_to_box,
    normalize_rgb,
    patch_pool_xyz,
)
from src.pose.render import SEG_TEMPLATE_HW, seg_template_K
from src.poses.utils import load_index_level_in_level2
from src.utils.inout import get_root_project

# Min mean IoU between the template alpha and the re-rendered mask. Only guards against grossly
# misaligned templates (thin objects reach ~0.96 because of half-pixel edge differences).
SEG_IOU_THRESH = 0.9
SEG_DISTANCE_M = 0.4  # camera distance of the segmentation templates
RENDER_SIZE = 448  # image size of the in-process templates
VIEW_SETS = ("level0", "cube8", "oct14", "ico12")
# bump when the way a bank is built changes (part of the cache folder name)
BANK_VERSION = 4


def parse_pose_set(name):
    """"seg42" -> ("seg42", 1); "auto42" -> ("auto42", 1); "cube8x5" -> ("cube8", 5)."""
    if name in ("seg42", "auto42"):
        return name, 1
    match = re.fullmatch(r"(level0|cube8|oct14|ico12)x(\d+)", str(name))
    if match is None or int(match.group(2)) < 1:
        raise ValueError(
            f"Unknown pose set '{name}', expected 'seg42', 'auto42' or '<{'|'.join(VIEW_SETS)}>x<rolls>'"
        )
    return match.group(1), int(match.group(2))


def _level0_rotations():
    """R_m2c of the 42 level-0 segmentation templates, (42, 3, 3) float64."""
    poses = np.load(
        osp.join(get_root_project(), "src/poses/predefined_poses/obj_poses_level2.npy")
    )
    index = load_index_level_in_level2(0, "all")
    return poses[index][:, :3, :3].astype(np.float64)


def _cube_directions():
    verts = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _camera_directions(views):
    """Unit camera-centre directions in the object frame."""
    if views == "cube8":
        return _cube_directions()
    if views == "oct14":
        axes = np.concatenate([np.eye(3), -np.eye(3)], axis=0)
        return np.concatenate([axes, _cube_directions()], axis=0)
    if views == "ico12":
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        verts = []
        for a in (-1.0, 1.0):
            for b in (-phi, phi):
                verts += [(0.0, a, b), (a, b, 0.0), (b, 0.0, a)]
        verts = np.array(verts)
        return verts / np.linalg.norm(verts, axis=1, keepdims=True)
    raise ValueError(f"Unknown views '{views}'")


def _look_at_origin(c):
    """R_m2c of a camera at direction c (object frame) looking at the origin, up = +Z."""
    z = -c
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(c, up))) > 0.99:
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def get_view_rotations(views, rolls=1):
    """R_m2c of every view and in-plane roll, (T, 3, 3) float64 (all rolls of a view are adjacent)."""
    if views == "level0":
        base = _level0_rotations()
    elif views in VIEW_SETS:
        base = np.stack([_look_at_origin(c) for c in _camera_directions(views)])
    else:
        raise ValueError(f"Unknown views '{views}'")
    out = []
    for R in base:
        for k in range(rolls):
            a = 2.0 * np.pi * k / rolls
            Rz = np.array(
                [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
            )
            out.append(Rz @ R)
    return np.stack(out).astype(np.float64)


class TemplateBank:
    def __init__(
        self,
        dataset_name,
        cad_dir,
        template_dir,
        models_info,
        pose_set,
        extractor,
        renderer,
        crop_size=448,
        crop_pad=1.1,
        min_cover=0.5,
        cache_dir=None,
        device="cuda",
        reset=False,
    ):
        assert crop_size % PATCH == 0, f"crop_size must be a multiple of {PATCH}"
        self.dataset_name = dataset_name
        self.cad_dir = cad_dir
        self.template_dir = template_dir
        self.models_info = models_info
        self.pose_set = pose_set
        self.views, self.rolls = parse_pose_set(pose_set)
        self.extractor = extractor
        self.renderer = renderer
        self.crop_size = crop_size
        self.crop_pad = crop_pad
        self.min_cover = min_cover
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.reset = reset
        self._banks = {}
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    def get(self, obj_id):
        obj_id = int(obj_id)
        if obj_id in self._banks:
            return self._banks[obj_id]
        path = None
        if self.cache_dir is not None:
            path = osp.join(self.cache_dir, f"obj_{obj_id:06d}.pth")
        use_cache = path is not None and osp.exists(path) and not self.reset
        if use_cache and self._cache_is_stale(obj_id, path):
            logging.info(
                f"[TemplateBank] obj {obj_id}: cache older than its templates / CAD model, rebuilding"
            )
            use_cache = False
        if use_cache:
            bank = torch.load(path, map_location="cpu")
            source = "cache"
        else:
            bank = self._build(obj_id)
            source = "built"
            if path is not None:
                tmp_path = path + ".tmp"
                torch.save(self._to_device(bank, "cpu"), tmp_path)
                os.replace(tmp_path, path)
        bank = self._to_device(bank, self.device)
        self._banks[obj_id] = bank
        msg = (
            f"[TemplateBank] obj {obj_id} ({source}): T={len(bank['poses'])}, "
            f"mean valid patches {bank['valid'].sum(dim=1).float().mean().item():.1f}"
            f"/{bank['valid'].shape[1]}"
        )
        if self.views in ("seg42", "auto42"):
            msg += (
                f", shared={bank['shared']}, seg_iou={bank['seg_iou']:.4f}"
                f", truncated {bank['n_truncated']}/{len(bank['poses'])}"
            )
        logging.info(msg)
        return bank

    def _cache_is_stale(self, obj_id, path):
        """True if the CAD model or (seg42 / auto42) a segmentation template is newer than the cache file."""
        sources = [osp.join(self.cad_dir, f"obj_{obj_id:06d}.ply")]
        if self.views in ("seg42", "auto42"):
            obj_dir = osp.join(self.template_dir, f"obj_{obj_id:06d}")
            sources += [
                osp.join(obj_dir, f"{int(idx):06d}.png")
                for idx in load_index_level_in_level2(0, "all")
            ]
        cache_mtime = osp.getmtime(path)
        return any(osp.exists(p) and osp.getmtime(p) > cache_mtime for p in sources)

    @staticmethod
    def _to_device(bank, device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in bank.items()
        }

    def _build(self, obj_id):
        shared, seg_iou, n_truncated = False, float("nan"), 0
        if self.views in ("seg42", "auto42"):
            samples, poses, Ks, shared, seg_iou, n_truncated = self._seg42_samples(obj_id)
            width = SEG_TEMPLATE_HW[1] if shared else RENDER_SIZE
        else:
            samples, poses, Ks = self._rendered_samples(obj_id)
            width = RENDER_SIZE
        bank = self._encode(samples, poses, Ks, width)
        bank["shared"] = shared
        bank["seg_iou"] = float(seg_iou)
        bank["n_truncated"] = int(n_truncated)
        if bank["valid"].sum() == 0:
            raise RuntimeError(
                f"obj {obj_id}: no valid template patch (pose set {self.pose_set}); "
                "check the CAD model and the templates"
            )
        return bank

    def _seg42_samples(self, obj_id):
        """Samples (rgb, mask, depth_mm) of the 42 segmentation templates, or of own renderings
        of the same viewpoints if the segmentation templates cannot be shared."""
        info = self.renderer.mesh_info(obj_id)
        centroid = np.asarray(info["centroid"], dtype=np.float64)
        indices = load_index_level_in_level2(0, "all")
        rotations = _level0_rotations()
        K = seg_template_K(self.dataset_name)
        H, W = SEG_TEMPLATE_HW
        obj_dir = osp.join(self.template_dir, f"obj_{obj_id:06d}")

        poses = np.tile(np.eye(4), (len(indices), 1, 1))
        rgbs, masks, depths, ious = [], [], [], []
        n_truncated = 0
        for i, (idx, R) in enumerate(zip(indices, rotations)):
            # the templates were rendered with X_c[m] = R (0.001 v_mm - centroid_raw) + (0, 0, 0.4)
            poses[i, :3, :3] = R
            poses[i, :3, 3] = 1000.0 * (R @ (-centroid) + np.array([0.0, 0.0, SEG_DISTANCE_M]))
            png = np.asarray(
                Image.open(osp.join(obj_dir, f"{int(idx):06d}.png")).convert("RGBA")
            )
            if png.shape[:2] != (H, W):
                raise ValueError(
                    f"Template {obj_dir}/{int(idx):06d}.png has size {png.shape[:2]}, expected {(H, W)}"
                )
            _, depth = self.renderer.render(obj_id, poses[i], K, (H, W))
            alpha = png[..., 3] > 0
            fg = depth > 0
            union = np.count_nonzero(alpha | fg)
            ious.append(np.count_nonzero(alpha & fg) / union if union > 0 else 0.0)
            n_truncated += int(
                alpha[0].any() or alpha[-1].any() or alpha[:, 0].any() or alpha[:, -1].any()
            )
            rgbs.append(png[..., :3])
            masks.append(alpha & fg)
            depths.append(depth)

        seg_iou = float(np.mean(ious))
        # the IoU alone detects templates that do not match the CAD model (e.g. wrong scale)
        shared = seg_iou >= SEG_IOU_THRESH
        if self.views == "auto42":
            shared = shared and n_truncated == 0
        if not shared:
            logging.warning(
                f"obj {obj_id}: own templates instead of the segmentation ones "
                f"(mean IoU {seg_iou:.4f}, truncated {n_truncated}/{len(indices)})"
            )
            samples, poses, Ks = self._rendered_samples(obj_id, views="level0", rolls=1)
            return samples, poses, Ks, False, seg_iou, n_truncated
        Ks = np.tile(K, (len(indices), 1, 1))
        return list(zip(rgbs, masks, depths)), poses, Ks, True, seg_iou, n_truncated

    def _rendered_samples(self, obj_id, views=None, rolls=None):
        """Poses, intrinsics and a generator of samples (rgb, mask, depth_mm) rendered in-process."""
        views = self.views if views is None else views
        rolls = self.rolls if rolls is None else rolls
        D = float(self.models_info[obj_id]["diameter"])
        dist = 5.0 * D
        f = 0.9 * RENDER_SIZE * dist / D
        c = (RENDER_SIZE - 1) / 2.0
        K = np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])
        rotations = get_view_rotations(views, rolls)
        poses = np.tile(np.eye(4), (len(rotations), 1, 1))
        poses[:, :3, :3] = rotations
        poses[:, :3, 3] = np.array([0.0, 0.0, dist])
        Ks = np.tile(K, (len(rotations), 1, 1))

        def generate():
            for pose in poses:
                rgb, depth = self.renderer.render(
                    obj_id, pose, K, (RENDER_SIZE, RENDER_SIZE)
                )
                yield rgb, depth > 0, depth

        return generate(), poses, Ks

    @torch.no_grad()
    def _encode(self, samples, poses, Ks, width):
        """Crop, back-project and extract patch features of every template."""
        T, S = len(poses), self.crop_size
        N = (S // PATCH) ** 2
        dev = self.device
        xyz = torch.zeros(T, N, 3, dtype=torch.float32, device=dev)
        valid = torch.zeros(T, N, dtype=torch.bool, device=dev)
        crops = torch.zeros(T, 3, dtype=torch.float64)
        crops[:, 2] = float(width)  # empty templates keep the crop (0, 0, W)
        inputs, kept = [], []
        for i, (rgb, mask, depth) in enumerate(samples):
            box = mask_to_box(mask)
            if box is None:
                continue
            left, top, side = box_to_square_crop(box, self.crop_pad)
            crops[i] = torch.tensor([left, top, side], dtype=torch.float64)

            img = normalize_rgb(torch.from_numpy(np.ascontiguousarray(rgb)).to(dev))
            mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
            depth_t = torch.from_numpy(np.ascontiguousarray(depth)).to(dev)
            rgb_c = crop_resize(img, left, top, side, S)
            m_c = crop_resize(mask_t[None].float(), left, top, side, S)[0] > 0.5
            xyz_map, v = backproject_to_model(depth_t, Ks[i], poses[i, :3, :3], poses[i, :3, 3])
            xyz_c = crop_resize(
                xyz_map.permute(2, 0, 1), left, top, side, S, mode="nearest"
            ).permute(1, 2, 0)
            v_c = crop_resize(v[None].float(), left, top, side, S, mode="nearest")[0] > 0.5
            xyz_patch, cover = patch_pool_xyz(xyz_c, m_c & v_c)
            xyz[i] = xyz_patch
            valid[i] = cover >= self.min_cover
            inputs.append(rgb_c * m_c[None].float())
            kept.append(i)

        feats = torch.zeros(T, N, self.extractor.dim, dtype=torch.float16, device=dev)
        if len(kept) > 0:
            out = self.extractor.extract(torch.stack(inputs))
            feats[torch.tensor(kept, device=dev)] = out.to(dev).half()
        return {
            "feats": feats,
            "valid": valid,
            "xyz": xyz,
            "poses": torch.from_numpy(np.asarray(poses, dtype=np.float64)),
            "K": torch.from_numpy(np.asarray(Ks, dtype=np.float64)),
            "crops": crops,
        }
