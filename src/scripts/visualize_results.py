r"""
Qualitative figures of Free6D on BOP test images. Every requested image gives any of the panels
  input  the untouched test image,
  masks  the localisation masks the poses were estimated from, filled at alpha 0.5 in the colour
         of their object id, with a 2 px contour,
  pose   the CAD models rendered at the estimated 6D poses with the image intrinsics, composed by
         depth (nearest wins), shaded as neutral grey clay, blended over the image at alpha 0.7 and
         outlined by their visible silhouette in the colour of their object id; with --box also the
         projected 3D bounding box of every model in the same colour,
laid out as one row per image (--layout rows, optional column titles) or flowing into a grid of
--cols columns (--layout grid). Object ids are coloured from a fixed palette of 30 colours,
cycling (HB has 33 objects: ids 31-33 share the colours of ids 1-3).

The drawn instances are the rows of the pose csv of the image with score >= --score_thr. The
masks are the detections run_pose.py estimated those poses from (for every target object its
inst_count best-scoring detections), checked row by row against the csv (object id and score),
so both panels show the same instances.

usage: python -m src.scripts.visualize_results --items ycbv:50:1480 --panels input masks pose \
           --titles "RGB image" "Localisation" "6D pose" --label none --box --out /tmp/free6d_vis/rows.jpg
       python -m src.scripts.visualize_results --items lmo:2:535 tless:7:348 icbin:1:37 hb:5:990 \
           --panels pose --layout grid --cols 4 --panel_width 480 --out /tmp/free6d_vis/grid.jpg
"""
import argparse
import functools
import logging
import os
import os.path as osp
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bop_toolkit_lib import inout
from src.dataloader.bop import load_rgb_uint8
from src.pose.bop_io import (
    SceneReader,
    cad_dir,
    det_mask,
    load_detections,
    load_models_info,
    load_targets,
    split_name,
)
from src.pose.render import ObjectRenderer
from src.scripts.summarize_bop_results import PAPER_NAMES, PAPER_ORDER

PANELS = ("input", "masks", "pose")
OUT_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_ROOT = "./datasets/bop23_challenge"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# 30 saturated colours, greedy farthest-point selection in CIELAB (lightness >= 42), indexed by
# (obj_id - 1) % 30, i.e. cycling for datasets with more than 30 objects
PALETTE = [
    (219, 25, 11), (13, 255, 13), (13, 77, 255), (64, 229, 255), (255, 217, 64), (255, 13, 174),
    (46, 184, 101), (9, 102, 184), (223, 13, 255), (184, 110, 46), (184, 46, 83), (207, 255, 13),
    (165, 46, 184), (147, 184, 46), (64, 255, 217), (255, 126, 13), (13, 255, 126), (46, 147, 184),
    (44, 184, 9), (255, 13, 94), (184, 46, 138), (64, 127, 255), (184, 156, 46), (46, 184, 165),
    (184, 64, 46), (13, 174, 255), (142, 13, 255), (140, 255, 64), (255, 64, 230), (255, 174, 13),
]
MASK_ALPHA = 0.5
CLAY_ALPHA = 0.7
# grey level (fraction of 255) of grazing surfaces (normal perpendicular to the viewing ray) and of
# surfaces facing the camera; in between linear in |cos| of the angle between normal and viewing ray
CLAY_GREY = (0.3, 0.92)
CONTOUR_PX = 2
TITLE_COLOR = (31, 35, 40)  # #1f2328
NEAR_MM = 10.0  # box edges are clipped at this depth in front of the camera
# the 12 edges of a box as pairs of corner indices (bit k of a corner index set = max along axis k)
BOX_EDGES = np.array([(a, a | 1 << k) for a in range(8) for k in range(3) if not a & 1 << k])
BOX_CORNERS = np.array([[(a >> k) & 1 for k in range(3)] for a in range(8)], dtype=np.float64)


def obj_color(obj_id):
    return PALETTE[(int(obj_id) - 1) % len(PALETTE)]


def parse_item(spec):
    """'ycbv:48:1' -> ('ycbv', 48, 1)."""
    parts = spec.split(":")
    if len(parts) != 3 or parts[0] not in PAPER_ORDER or not (parts[1].isdigit() and parts[2].isdigit()):
        raise ValueError(f"bad item '{spec}': expected <dataset>:<scene_id>:<im_id>, dataset in {PAPER_ORDER}")
    return parts[0], int(parts[1]), int(parts[2])


def _require(path, what):
    if not osp.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


# per-process caches: the detection files are tens of MB
@functools.lru_cache(maxsize=None)
def _targets(dataset_dir):
    return load_targets(dataset_dir)


@functools.lru_cache(maxsize=None)
def _detections(path):
    return load_detections(path)


@functools.lru_cache(maxsize=None)
def _pose_rows(path):
    """{(scene_id, im_id): [rows in file order]} of a BOP'19 pose csv."""
    rows = {}
    for r in inout.load_bop_results(path, version="bop19"):
        rows.setdefault((r["scene_id"], r["im_id"]), []).append(r)
    return rows


@functools.lru_cache(maxsize=None)
def _reader(dataset_dir, dataset):
    return SceneReader(dataset_dir, dataset)


@functools.lru_cache(maxsize=None)
def _models_info(dataset_dir, dataset):
    return load_models_info(dataset_dir, dataset)


def selected_detections(dets_img, targets_img):
    """
    The detections run_pose.py estimates a pose for, in the order of its csv rows (same selection
    as run_pose.main): per target object its inst_count best-scoring detections; an empty mask
    gives no estimate (PoseEstimator.estimate_image returns None for it).
    """
    out = []
    for obj_id, inst_count in targets_img.items():
        cand = sorted((d for d in dets_img if int(d["category_id"]) == obj_id), key=lambda d: -float(d["score"]))
        for d in cand[:inst_count]:
            mask = det_mask(d)
            if mask.any():
                out.append({"obj_id": obj_id, "score": float(d["score"]), "mask": mask})
    return out


def load_item(dataset, scene_id, im_id, root=DEFAULT_ROOT, method="free6d", tag="refine", score_thr=0.5):
    """
    One test image with its drawn instances: dict with dataset, scene_id, im_id, image (H, W, 3)
    uint8 RGB, K (3, 3) and instances [{obj_id, score, mask (H, W) bool, R (3, 3), t (3,) mm,
    info (models_info entry of the object)}].
    """
    dataset_dir = osp.join(root, "datasets", dataset)
    split = split_name(dataset).replace("_", "-")
    det_path = _require(osp.join(
        root, "results", "seg_exps", f"CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_{dataset}.json",
    ), f"localisation output of {dataset} (run_inference.py)")
    pose_path = _require(osp.join(root, "results", "pose_exps", f"{method}{tag}_{dataset}-{split}.csv"),
                         f"pose csv of {dataset} (run_pose.py --tag '{tag}')")
    targets_img = _targets(dataset_dir).get((scene_id, im_id))
    if targets_img is None:
        raise ValueError(f"{dataset} {scene_id}/{im_id} is not a BOP'19 test target image")
    reader = _reader(dataset_dir, dataset)
    image = load_rgb_uint8(_require(reader.image_path(scene_id, im_id), "test image"))
    cam_path = _require(osp.join(reader.split_dir, f"{scene_id:06d}", "scene_camera.json"), "camera file")
    try:
        K = reader.K(scene_id, im_id)
    except KeyError:
        raise ValueError(f"{dataset} {scene_id}/{im_id}: no camera entry in {cam_path}") from None

    dets = selected_detections(_detections(det_path).get((scene_id, im_id), []), targets_img)
    rows = _pose_rows(pose_path).get((scene_id, im_id), [])
    if len(dets) != len(rows) or any(
        d["obj_id"] != r["obj_id"] or abs(d["score"] - r["score"]) > 1e-6 for d, r in zip(dets, rows)
    ):
        raise ValueError(f"{dataset} {scene_id}/{im_id}: the {len(rows)} rows of {pose_path} do not match the "
                         f"{len(dets)} detections selected from {det_path}; were the poses estimated from it?")
    info = _models_info(dataset_dir, dataset)
    instances = [
        {**d, "R": r["R"], "t": r["t"].reshape(3), "info": info[d["obj_id"]]}
        for d, r in zip(dets, rows) if r["score"] >= score_thr
    ]
    return {"dataset": dataset, "scene_id": scene_id, "im_id": im_id, "image": image, "K": K, "instances": instances}


def make_renderer(dataset, root=DEFAULT_ROOT):
    """ObjectRenderer of the CAD models of a dataset; close() it when done."""
    return ObjectRenderer(cad_dir(osp.join(root, "datasets", dataset), dataset), dataset)


def _resize(image, width):
    if not width or width == image.shape[1]:
        return image
    return cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)


def _paint(image, fill, region, outlines, colors, alpha, width):
    """
    Blend `fill` (H, W, 3) over `image` inside `region` at image resolution, resize to `width`, then
    draw the contour of every mask of `outlines` in its colour at panel resolution (later on top).
    """
    out = image.astype(np.float32)
    out[region] = (1.0 - alpha) * out[region] + alpha * fill[region]
    out = _resize(np.clip(out + 0.5, 0, 255).astype(np.uint8), width)
    for mask, color in zip(outlines, colors):
        m = cv2.resize(mask.astype(np.float32), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_AREA) >= 0.5
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), CONTOUR_PX, cv2.LINE_AA)
    return out


def draw_input(item, width=None):
    return _resize(item["image"], width)


def draw_masks(item, width=None):
    """Instance masks filled in their object colour, larger masks first so that smaller ones stay visible."""
    insts = sorted(item["instances"], key=lambda inst: -int(inst["mask"].sum()))
    fill = np.zeros(item["image"].shape, dtype=np.float32)
    region = np.zeros(item["image"].shape[:2], dtype=bool)
    for inst in insts:
        fill[inst["mask"]] = obj_color(inst["obj_id"])
        region |= inst["mask"]
    masks = [inst["mask"] for inst in insts]
    return _paint(item["image"], fill, region, masks, [obj_color(i["obj_id"]) for i in insts], MASK_ALPHA, width)


def _clay_shade(depth, K):
    """
    Headlight Lambert shading in [0, 1] of a rendered depth map (mm): |cos| of the angle between the
    viewing ray and the normal of the back-projected points.
    """
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    P = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], axis=-1)
    valid = depth > 0

    def diff(axis):
        """
        Difference to the next pixel along `axis`, to the previous one where the next is background
        or outside the image (no wrap-around), 0 where neither exists.
        """
        fwd, bwd = np.zeros_like(P), np.zeros_like(P)
        ok_f, ok_b = np.zeros_like(valid), np.zeros_like(valid)
        head, tail = [slice(None)] * 2, [slice(None)] * 2  # index i and i + 1 along `axis`, no wrap-around
        head[axis], tail[axis] = slice(None, -1), slice(1, None)
        head, tail = tuple(head), tuple(tail)
        step = P[tail] - P[head]
        fwd[head], bwd[tail] = step, step
        ok_f[head] = valid[head] & valid[tail]
        ok_b[tail] = ok_f[head]
        return np.where(ok_f[..., None], fwd, np.where(ok_b[..., None], bwd, 0.0))

    n = np.cross(diff(1), diff(0))
    n_len = np.linalg.norm(n, axis=-1)
    view = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-6)
    shade = np.abs((n * view).sum(-1)) / np.maximum(n_len, 1e-9)
    return np.where(n_len > 0, shade, 1.0)


def _draw_box(out, K, hw, inst):
    """
    The 12 edges of the model's bounding box (models_info, mm) at the pose of `inst`, clipped at
    NEAR_MM, projected with K of an (H, W) image and drawn anti-aliased with sub-pixel precision on
    the panel `out` (any size) in the colour of the object.
    """
    info, R, t = inst["info"], inst["R"], inst["t"]
    lo = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=np.float64)
    size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float64)
    corners = (lo + BOX_CORNERS * size) @ R.T + t  # camera frame
    color = tuple(int(c) for c in obj_color(inst["obj_id"]))
    sx, sy = out.shape[1] / hw[1], out.shape[0] / hw[0]
    for p, q in corners[BOX_EDGES]:
        if p[2] < NEAR_MM and q[2] < NEAR_MM:
            continue
        if p[2] < NEAR_MM:
            p = q + (p - q) * (q[2] - NEAR_MM) / (q[2] - p[2])
        elif q[2] < NEAR_MM:
            q = p + (q - p) * (p[2] - NEAR_MM) / (p[2] - q[2])
        a, b = K @ p, K @ q
        # pixel index in the panel, in 1/16 px (shift = 4)
        a = (round(16 * ((a[0] / a[2] + 0.5) * sx - 0.5)), round(16 * ((a[1] / a[2] + 0.5) * sy - 0.5)))
        b = (round(16 * ((b[0] / b[2] + 0.5) * sx - 0.5)), round(16 * ((b[1] / b[2] + 0.5) * sy - 0.5)))
        cv2.line(out, a, b, color, CONTOUR_PX, cv2.LINE_AA, 4)


def draw_poses(item, renderer, width=None, box=False):
    """
    CAD models at the estimated poses, depth-composed, shaded as grey clay and outlined in their
    object colour; with `box`, also their projected 3D bounding boxes.
    """
    H, W = item["image"].shape[:2]
    K, insts = item["K"], item["instances"]
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    label = np.full((H, W), -1, dtype=np.int32)
    fill = np.zeros((H, W, 3), dtype=np.float32)
    lo, hi = CLAY_GREY
    for i, inst in enumerate(insts):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = inst["R"], inst["t"]
        _, depth = renderer.render(inst["obj_id"], T, K, (H, W))
        vis = (depth > 0) & (depth < zbuf)
        if not vis.any():
            continue
        fill[vis] = (255.0 * (lo + (hi - lo) * _clay_shade(depth, K)[vis]))[:, None]
        zbuf[vis], label[vis] = depth[vis], i
    # contours / boxes farthest first so that the nearer ones are drawn on top
    order = sorted(range(len(insts)), key=lambda i: -float(insts[i]["t"][2]))
    outlines = [label == i for i in order]
    colors = [obj_color(insts[i]["obj_id"]) for i in order]
    out = _paint(item["image"], fill, label >= 0, outlines, colors, CLAY_ALPHA, width)
    if box:
        for i in order:
            _draw_box(out, K, (H, W), insts[i])
    return out


def draw_panels(item, panels, renderer=None, width=None, box=False):
    if "pose" in panels and renderer is None:
        raise ValueError("a 'pose' panel needs a renderer (make_renderer(dataset))")
    draw = {"input": lambda: draw_input(item, width), "masks": lambda: draw_masks(item, width),
            "pose": lambda: draw_poses(item, renderer, width, box)}
    return [draw[p]() for p in panels]


@functools.lru_cache(maxsize=None)
def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:  # Pillow < 10.1: fixed-size bitmap font only
            return ImageFont.load_default()


def _text_box(canvas, text, xy, size):
    """White `text` on a semi-transparent dark rounded box with its top-left corner at `xy`."""
    font, pad = _font(size), max(2, round(0.35 * size))
    l, t, r, b = font.getbbox(text)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x, y = xy
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=max(2, round(0.3 * size)), fill=(20, 20, 20, 170))
    draw.text((x + pad - l, y + pad - t), text, font=font, fill=(255, 255, 255, 255))
    canvas.alpha_composite(layer)


def add_label(panel, text):
    """Small label in the top-left corner of a panel (uint8 RGB array), size scaled with its width."""
    if not text:
        return panel
    canvas = Image.fromarray(panel).convert("RGBA")
    size = max(9, round(0.035 * panel.shape[1]))
    _text_box(canvas, text, (round(0.4 * size), round(0.4 * size)), size)
    return np.asarray(canvas.convert("RGB"))


def item_label(dataset, scene_id, im_id, mode):
    return {"dataset": PAPER_NAMES[dataset], "id": f"{dataset} {scene_id}/{im_id}", "none": ""}[mode]


def compose(panels, cols, gap=6, titles=None):
    """Panels (uint8 RGB arrays, row-major) on a white grid of `cols` columns; optional column titles above."""
    rows = [panels[i:i + cols] for i in range(0, len(panels), cols)]
    col_w = [max(p.shape[1] for p in panels[c::cols]) for c in range(min(cols, len(panels)))]
    row_h = [max(p.shape[0] for p in row) for row in rows]
    top = round(0.07 * col_w[0]) if titles else 0  # white title strip
    xs = np.concatenate([[0], np.cumsum(np.add(col_w, gap))])
    ys = top + np.concatenate([[0], np.cumsum(np.add(row_h, gap))])
    canvas = Image.new("RGB", (int(xs[-1] - gap), int(ys[-1] - gap)), (255, 255, 255))
    for r, row in enumerate(rows):
        for c, panel in enumerate(row):
            canvas.paste(Image.fromarray(panel), (int(xs[c]), int(ys[r])))
    draw, font = ImageDraw.Draw(canvas), _font(max(10, round(0.036 * col_w[0])))
    for c, title in enumerate(titles or []):  # plain bold dark text centred on the strip
        draw.text((xs[c] + col_w[c] / 2, top / 2), title, font=font, fill=TITLE_COLOR, anchor="mm")
    return np.asarray(canvas)


def save_image(image, path, quality=90):
    """JPEG or PNG according to the extension of `path`."""
    ext = osp.splitext(path)[1].lower()
    if ext not in OUT_EXTS:
        raise ValueError(f"unsupported output extension '{ext}' (use {', '.join(OUT_EXTS)})")
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    kwargs = {"quality": quality, "subsampling": 0} if ext != ".png" else {"optimize": True}
    Image.fromarray(image).save(path, **kwargs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", nargs="+", required=True, help="test images as <dataset>:<scene_id>:<im_id>")
    p.add_argument("--panels", nargs="+", default=list(PANELS), choices=PANELS, help="panels per image, in order")
    p.add_argument("--layout", default="rows", choices=("rows", "grid"),
                   help="rows: one row per image; grid: all panels flow into --cols columns")
    p.add_argument("--cols", type=int, default=4, help="columns of --layout grid")
    p.add_argument("--panel_width", type=int, default=640, help="width of every panel (px), aspect ratio kept")
    p.add_argument("--gap", type=int, default=6, help="white gap between panels (px)")
    p.add_argument("--titles", nargs="+", default=None, help="column titles (--layout rows only), one per panel")
    p.add_argument("--label", default="dataset", choices=("dataset", "id", "none"),
                   help="top-left label of each panel: 'YCB-V', 'ycbv 48/1' or none")
    p.add_argument("--box", action="store_true",
                   help="pose panel: also draw the projected 3D bounding box of every model")
    p.add_argument("--score_thr", type=float, default=0.5, help="minimum score of the drawn pose estimates")
    p.add_argument("--method", default="free6d", help="method name of the pose csv")
    p.add_argument("--tag", default="refine", help="pose csv <method><tag>_<dataset>-<split>.csv ('' = coarse poses)")
    p.add_argument("--root", default=DEFAULT_ROOT, help="BOP root with datasets/ and results/")
    p.add_argument("--quality", type=int, default=90, help="JPEG quality")
    p.add_argument("--out", required=True, help="output image, .jpg / .jpeg or .png")
    args = p.parse_args()
    if osp.splitext(args.out)[1].lower() not in OUT_EXTS:
        p.error(f"--out '{args.out}': unsupported extension (use {', '.join(OUT_EXTS)})")
    try:
        args.items = [parse_item(s) for s in args.items]
    except ValueError as e:
        p.error(str(e))
    if args.titles is not None and (args.layout != "rows" or len(args.titles) != len(args.panels)):
        p.error("--titles needs --layout rows and one title per panel")
    if args.cols < 1 or args.panel_width < 16 or args.gap < 0:
        p.error("--cols >= 1, --panel_width >= 16 and --gap >= 0 required")
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    try:
        # input errors (missing file, image not a target, csv / detections mismatch) end with a one-line
        # message before anything is rendered; errors while drawing raise with their traceback
        try:
            items = {
                key: load_item(*key, root=args.root, method=args.method, tag=args.tag, score_thr=args.score_thr)
                for key in dict.fromkeys(args.items)
            }
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"visualize_results: {e}")
        drawn = {}
        # one renderer (GL context) at a time: the items are drawn dataset by dataset
        for dataset in dict.fromkeys(ds for ds, _, _ in items):
            renderer = make_renderer(dataset, args.root) if "pose" in args.panels else None
            try:
                for key in (k for k in items if k[0] == dataset):
                    t0 = time.perf_counter()
                    panels = draw_panels(items[key], args.panels, renderer, args.panel_width, args.box)
                    drawn[key] = [add_label(p, item_label(*key, args.label)) for p in panels]
                    logging.info(f"{dataset} {key[1]}/{key[2]}: {len(items[key]['instances'])} instances, "
                                 f"drawn in {time.perf_counter() - t0:.2f}s")
            finally:
                if renderer is not None:
                    renderer.close()
        panels = [p for key in args.items for p in drawn[key]]
        cols = len(args.panels) if args.layout == "rows" else args.cols
        save_image(compose(panels, cols, args.gap, args.titles), args.out, args.quality)
    finally:
        for cache in (_targets, _detections, _pose_rows, _reader, _models_info, _font):
            cache.cache_clear()
    logging.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
