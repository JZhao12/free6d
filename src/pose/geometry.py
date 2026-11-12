"""
Image / crop / patch geometry helpers of the training-free 6D pose stage.

Conventions:
- Model coordinates in mm (BOP). Pose T_m2c: X_c = R X_m + t, OpenCV camera.
- Integer (u, v) is a pixel centre and K follows it: u = fx * X / Z + cx.
  A "continuous" coordinate is the pixel index + 0.5.
- A square crop is (left, top, side) in continuous source coordinates, resampled to
  S x S. With s = side / S, output pixel (j=col, i=row) maps to the source pixel
  index u = left + (j + 0.5) * s - 0.5, v = top + (i + 0.5) * s - 0.5.
- ViT patches (size 16) are indexed row-major: index = pi * G + pj, G = S // 16.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH = 16


def normalize_rgb(rgb_uint8):
    """(H, W, 3) uint8 numpy / torch -> (3, H, W) float32 ImageNet-normalised tensor."""
    if isinstance(rgb_uint8, np.ndarray):
        rgb = torch.from_numpy(np.ascontiguousarray(rgb_uint8))
    else:
        rgb = rgb_uint8
    rgb = rgb.permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32, device=rgb.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32, device=rgb.device).view(3, 1, 1)
    return (rgb - mean) / std


def mask_to_box(mask):
    """
    Bool (H, W) mask -> (x1, y1, x2, y2) continuous extents
    (x2 = max col + 1, y2 = max row + 1), None if the mask is empty.
    """
    mask = torch.as_tensor(mask).bool()
    cols = torch.nonzero(mask.any(dim=0)).flatten()
    rows = torch.nonzero(mask.any(dim=1)).flatten()
    if cols.numel() == 0:
        return None
    return (
        float(cols.min()),
        float(rows.min()),
        float(cols.max()) + 1.0,
        float(rows.max()) + 1.0,
    )


def box_to_square_crop(box, pad=1.1):
    """(x1, y1, x2, y2) -> (left, top, side): square centred on the box, side = max(w, h) * pad."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * pad
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return cx - 0.5 * side, cy - 0.5 * side, side


def crop_resize(img, left, top, side, out_size, mode="bilinear"):
    """
    Resample the square crop (left, top, side) of a (C, H, W) float tensor to
    (C, S, S) with grid_sample (zero padding outside the image).
    """
    _, H, W = img.shape
    s = side / out_size
    centers = (torch.arange(out_size, device=img.device, dtype=torch.float64) + 0.5) * s
    xs = 2.0 * (left + centers) / W - 1.0
    ys = 2.0 * (top + centers) / H - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((gx, gy), dim=-1)[None].to(img.dtype)  # (1, S, S, 2), (x, y)
    out = F.grid_sample(
        img[None], grid, mode=mode, padding_mode="zeros", align_corners=False
    )
    return out[0]


def crop_intrinsics(K, left, top, side, out_size):
    """Intrinsics (3x3 float64, pixel-index convention) of the S x S crop."""
    s = side / out_size
    K = np.asarray(K, dtype=np.float64)
    Kc = np.eye(3, dtype=np.float64)
    Kc[0, 0] = K[0, 0] / s
    Kc[1, 1] = K[1, 1] / s
    Kc[0, 2] = (K[0, 2] + 0.5 - left) / s - 0.5
    Kc[1, 2] = (K[1, 2] + 0.5 - top) / s - 0.5
    return Kc


def backproject_to_model(depth_mm, K, R, t):
    """
    Back-project a depth map (H, W) in mm (0 = background) to MODEL coordinates (mm):
    X_c = ((u - cx) * Z / fx, (v - cy) * Z / fy, Z), X_m = R^T (X_c - t).
    Returns xyz (H, W, 3) float32 (0 at invalid pixels) and valid (H, W) bool.
    """
    device = depth_mm.device
    K = torch.as_tensor(K, dtype=torch.float64).cpu()
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    R = torch.as_tensor(R, dtype=torch.float32, device=device)
    t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(3)
    H, W = depth_mm.shape
    z = depth_mm.float()
    u = torch.arange(W, device=device, dtype=torch.float32)[None, :]
    v = torch.arange(H, device=device, dtype=torch.float32)[:, None]
    xyz_c = torch.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), dim=-1)
    xyz_m = (xyz_c - t) @ R  # row-vector form of R^T (X_c - t)
    valid = z > 0
    xyz_m = torch.where(valid[..., None], xyz_m, torch.zeros_like(xyz_m))
    return xyz_m, valid


def patch_centers(out_size, device="cpu"):
    """(N, 2) float32 patch centres (x, y) in crop pixel-index coordinates, row-major."""
    G = out_size // PATCH
    pi, pj = torch.meshgrid(
        torch.arange(G, device=device), torch.arange(G, device=device), indexing="ij"
    )
    x = PATCH * pj.reshape(-1).float() + (PATCH - 1) / 2.0
    y = PATCH * pi.reshape(-1).float() + (PATCH - 1) / 2.0
    return torch.stack((x, y), dim=-1)


def rot90_points_back(pts, k, size):
    """
    pts (M, 2) float (x, y) pixel-index coordinates in an image obtained with
    torch.rot90(img, k, dims=(-2, -1)) of a size x size image -> coordinates in the
    un-rotated image.

    torch.rot90 (k = 1) is r[y', x'] = img[x', size - 1 - y'], i.e. x = size - 1 - y', y = x';
    k = 2: x = size - 1 - x', y = size - 1 - y'; k = 3: x = y', y = size - 1 - x'.
    """
    k = int(k) % 4
    x_r, y_r = pts[:, 0], pts[:, 1]
    if k == 0:
        x, y = x_r, y_r
    elif k == 1:
        x, y = (size - 1) - y_r, x_r
    elif k == 2:
        x, y = (size - 1) - x_r, (size - 1) - y_r
    else:
        x, y = y_r, (size - 1) - x_r
    if isinstance(pts, np.ndarray):
        return np.stack((x, y), axis=-1)
    return torch.stack((x, y), dim=-1)


def patch_pool_mask(mask):
    """(S, S) bool / float mask -> (N,) float32 coverage fraction per 16 x 16 patch."""
    return F.avg_pool2d(mask.float()[None, None], PATCH)[0, 0].reshape(-1)


def patch_pool_centers(mask):
    """
    (S, S) bool mask -> (N, 2) float32 (x, y): centroid, in crop pixel-index coordinates, of
    the masked pixels of each 16 x 16 patch (row-major); the patch centre where the patch
    has no masked pixel.
    """
    S = mask.shape[-1]
    ys, xs = torch.meshgrid(
        torch.arange(S, device=mask.device, dtype=torch.float32),
        torch.arange(S, device=mask.device, dtype=torch.float32),
        indexing="ij",
    )
    xy = torch.stack((xs, ys, torch.zeros_like(xs)), dim=-1)  # (S, S, 3), (x, y, 0) pixel indices
    cent, cover = patch_pool_xyz(xy, mask)
    return torch.where((cover <= 0)[:, None], patch_centers(S, mask.device), cent[:, :2])


def patch_pool_xyz(xyz, mask):
    """
    xyz (S, S, 3), mask (S, S) bool -> (xyz_patch (N, 3), cover (N,)).
    xyz_patch is the mean xyz over the masked pixels of each patch (0 where none),
    cover is the masked fraction of each patch.
    """
    m = mask.float()
    num = F.avg_pool2d((xyz.permute(2, 0, 1).float() * m[None])[None], PATCH)[0]
    cover = F.avg_pool2d(m[None, None], PATCH)[0, 0].reshape(-1)
    denom = cover.clamp_min(1.0 / (PATCH * PATCH))  # num is 0 wherever cover is 0
    xyz_patch = num.reshape(3, -1).T / denom[:, None]
    return xyz_patch, cover
