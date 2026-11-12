"""
Training-free 6D pose of the Free6D detections on a BOP test split: DINOv3 patch matching
against rendered templates, PnP, optional render-and-match refinement. Writes a BOP'19 pose
csv (METHOD_DATASET-SPLIT[-SPLITTYPE].csv, ready for src/scripts/eval_bop_pose.sh) and a
stats json next to it.

usage: python run_pose.py --dataset ycbv                                          # coarse configuration
       python run_pose.py --dataset ycbv --pose_set seg42 --tag seg42             # force the segmentation templates (default: auto42)
       python run_pose.py --dataset ycbv --n_hyp 3 --refine_iters 2 --tag refine  # + render-and-match refinement
       python run_pose.py --dataset ycbv --image_stride 10 --tag fast             # quick ablation run

Subset / diagnostic runs (--image_stride, --max_images, --oracle) need --tag so that they do not
overwrite the full result.
"""
import argparse
import json
import logging
import math
import os.path as osp
import re
import time

import numpy as np
import torch

from src.dataloader.bop import load_rgb_uint8
from src.pose.bop_io import (
    QuickEval,
    SceneReader,
    cad_dir,
    det_mask,
    load_detections,
    load_models_info,
    load_targets,
    save_bop_csv,
    split_name,
)
from src.pose.estimator import PoseConfig, PoseEstimator, view_direction
from src.pose.features import DenseDINOv3
from src.pose.geometry import mask_to_box
from src.pose.render import ObjectRenderer
from src.pose.templates import BANK_VERSION, TemplateBank

TIMER_STAGES = ("prep", "extract", "match", "pnp", "refine")


def parse_args():
    cfg = PoseConfig()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="BOP dataset name, e.g. ycbv")
    p.add_argument("--root", default="./datasets/bop23_challenge")
    p.add_argument("--detections", default=None,
                   help="run_inference.py result json (default: <root>/results/seg_exps/"
                        "CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_<dataset>.json)")
    p.add_argument("--pose_set", default="auto42",
                   help="template pose set: auto42 (default: the 42 segmentation templates, own renderings of "
                        "the same 42 viewpoints for the objects whose segmentation templates are cut by the "
                        "image border), seg42 (always the segmentation templates), "
                        "<level0|cube8|oct14|ico12>x<rolls> (own renderings)")
    p.add_argument("--layers", default="19", help="comma-separated DINOv3 layers")
    p.add_argument("--dinov3_repo", default="./dinov3_repo", help="DINOv3 repository (torch.hub local source)")
    p.add_argument("--dinov3_weights", default=None,
                   help="ViT-L/16 checkpoint (default: <dinov3_repo>/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth)")
    p.add_argument("--crop_size", type=int, default=cfg.crop_size)
    p.add_argument("--crop_pad", type=float, default=cfg.crop_pad)
    p.add_argument("--q_min_cover", type=float, default=cfg.q_min_cover)
    p.add_argument("--topk", type=int, default=cfg.topk)
    p.add_argument("--pnp_thr", type=float, default=cfg.pnp_thr)
    p.add_argument("--pnp_iters", type=int, default=cfg.pnp_iters)
    p.add_argument("--min_inliers", type=int, default=cfg.min_inliers)
    p.add_argument("--refine_iters", type=int, default=cfg.refine_iters)
    p.add_argument("--refine_radius", type=float, default=cfg.refine_radius,
                   help="match gate as a fraction of crop_size, <= 0 disables it")
    p.add_argument("--q_rolls", type=int, default=cfg.q_rolls, choices=(1, 2, 4),
                   help="query rotations by multiples of 90 deg (1 = none, 2 = 0/180, 4 = all)")
    p.add_argument("--q_centroid", type=int, default=int(cfg.q_centroid), choices=(0, 1),
                   help="1 = query 2D points are the centroids of the mask pixels of each patch instead of the patch centres")
    p.add_argument("--n_hyp", type=int, default=cfg.n_hyp,
                   help="PnP hypotheses passed to the render-and-compare scoring / refinement")
    p.add_argument("--oracle", action="store_true",
                   help="diagnostic: only use the template nearest to the ground-truth view")
    p.add_argument("--image_stride", type=int, default=1, help="keep every k-th target image")
    p.add_argument("--max_images", type=int, default=0, help="0 = all (applied after the stride)")
    p.add_argument("--out_dir", default=None, help="default: <root>/results/pose_exps")
    p.add_argument("--method", default="free6d",
                   help="alphanumeric only (bop_toolkit splits the file name on '.' and '_')")
    p.add_argument("--tag", default="",
                   help="appended to the method name for ablation runs, alphanumeric only; required for subset / diagnostic runs")
    p.add_argument("--cache", action="store_true",
                   help="keep the template banks on disk (about 67 MB per object), default: rebuild them at every run "
                        "(a few seconds per object)")
    p.add_argument("--reset_cache", action="store_true", help="rebuild the template bank cache (only with --cache)")
    p.add_argument("--no_eval", action="store_true", help="skip the quick local MSSD / MSPD evaluation")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if args.dinov3_weights is None:
        args.dinov3_weights = osp.join(args.dinov3_repo, "weights", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    if re.fullmatch(r"[A-Za-z0-9]+", args.method) is None:
        p.error(f"--method '{args.method}' must match [A-Za-z0-9]+ (bop_toolkit splits the file name on '.' and '_')")
    if re.fullmatch(r"[A-Za-z0-9]*", args.tag) is None:
        p.error(f"--tag '{args.tag}' must match [A-Za-z0-9]* (bop_toolkit splits the file name on '.' and '_')")
    if args.tag == "" and (args.oracle or args.image_stride > 1 or args.max_images > 0):
        p.error("subset / diagnostic runs need --tag so that they do not overwrite the full result")
    return args


def mean_or_none(values):
    return sum(values) / len(values) if values else None


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def fmt_pct(x):
    return "n/a" if x is None else f"{100.0 * x:.2f}%"


def fmt_mm(x):
    return "n/a" if x is None else f"{x:.1f}mm"


def match_gt(mask, obj_id, K, gts):
    """GT instance of the same object whose projected origin is closest to the mask box centre, else None.

    Nearest-instance association, not one-to-one: two detections of an object may be associated with the same
    ground-truth instance. Used for the diagnostics and for --oracle only, never for the estimates.
    """
    box = mask_to_box(mask)
    cands = [g for g in gts if g["obj_id"] == obj_id]
    if box is None or not cands:
        return None
    centre = np.array([0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])])

    def dist(g):
        p = K @ g["t"]
        return float(np.linalg.norm(p[:2] / p[2] - centre))

    return min(cands, key=dist)


def summarize_diag(records):
    """Pose / view error summary over the records with a GT association, None if there are none."""
    recs = [r for r in records if r["rot_err"] is not None]
    if not recs:
        return None
    col = {
        k: np.array([r[k] for r in recs], dtype=np.float64)
        for k in ("rot_err", "trans_err", "view_err", "view_err_best")
    }
    diag = {"n": len(recs)}
    for k, v in col.items():
        diag[f"{k}_mean"] = float(v.mean())
        diag[f"{k}_median"] = float(np.median(v))
    diag["frac_rot_lt_15"] = float((col["rot_err"] < 15.0).mean())
    diag["frac_rot_lt_30"] = float((col["rot_err"] < 30.0).mean())
    diag["frac_view_nearest"] = float((col["view_err"] <= col["view_err_best"] + 1e-3).mean())
    diag["frac_view_lt_40"] = float((col["view_err"] < 40.0).mean())
    # translation error split: relative depth error (signed) and lateral error, all / rot_err < 15 deg
    z_rel = np.array([r["z_err_rel"] for r in recs], dtype=np.float64)
    lat = np.array([r["lat_err"] for r in recs], dtype=np.float64)
    for suffix, keep in (("", np.ones(len(recs), dtype=bool)), ("_rot15", col["rot_err"] < 15.0)):
        ok = keep.any()
        diag[f"z_err_rel_median{suffix}"] = float(np.median(z_rel[keep])) if ok else None
        diag[f"z_err_rel_abs_median{suffix}"] = float(np.median(np.abs(z_rel[keep]))) if ok else None
        diag[f"lat_err_median{suffix}"] = float(np.median(lat[keep])) if ok else None
    return diag


@torch.no_grad()
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    dataset = args.dataset
    split = split_name(dataset)
    dataset_dir = osp.join(args.root, "datasets", dataset)
    template_dir = osp.join(args.root, "datasets", "templates_pyrender", dataset)
    detections_path = args.detections or osp.join(
        args.root, "results", "seg_exps",
        f"CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_{dataset}.json",
    )
    out_dir = args.out_dir or osp.join(args.root, "results", "pose_exps")
    stem = f"{args.method}{args.tag}_{dataset}-{split.replace('_', '-')}"
    csv_path = osp.join(out_dir, f"{stem}.csv")
    stats_path = osp.join(out_dir, f"{stem}_stats.json")
    on_cuda = args.device.startswith("cuda")

    cad = cad_dir(dataset_dir, dataset)
    models_info = load_models_info(dataset_dir, dataset)
    targets = load_targets(dataset_dir)
    detections = load_detections(detections_path)
    logging.info(f"{dataset}: {len(targets)} target images, {len(detections)} images with detections ({detections_path})")

    layers = tuple(int(x) for x in args.layers.split(",") if x.strip())
    extractor = DenseDINOv3(
        repo_dir=args.dinov3_repo, weights=args.dinov3_weights, layers=layers, device=args.device,
    )
    renderer = ObjectRenderer(cad, dataset)
    try:
        cache_dir = None
        if args.cache:
            cache_dir = osp.join(
                template_dir, "pose_bank",
                f"{extractor.name}_{args.pose_set}_S{args.crop_size}_pad{args.crop_pad}_v{BANK_VERSION}",
            )
        bank = TemplateBank(
            dataset, cad, template_dir, models_info, args.pose_set, extractor, renderer,
            crop_size=args.crop_size, crop_pad=args.crop_pad, cache_dir=cache_dir,
            device=args.device, reset=args.reset_cache,
        )
        cfg = PoseConfig(
            crop_size=args.crop_size, crop_pad=args.crop_pad, q_min_cover=args.q_min_cover,
            topk=args.topk, pnp_thr=args.pnp_thr, pnp_iters=args.pnp_iters,
            min_inliers=args.min_inliers, refine_iters=args.refine_iters,
            refine_radius=args.refine_radius, q_rolls=args.q_rolls, q_centroid=bool(args.q_centroid),
            n_hyp=args.n_hyp, oracle=args.oracle,
        )
        estimator = PoseEstimator(extractor, bank, renderer, cfg, device=args.device)

        reader = SceneReader(dataset_dir, dataset)
        quick = None if args.no_eval else QuickEval(dataset_dir, dataset)
        n_eval_images = 0

        images = sorted(targets)[:: max(1, args.image_stride)]
        if args.max_images > 0:
            images = images[: args.max_images]
        logging.info(f"{len(images)} images to process")

        # onboarding: template banks of all objects of the selected images, excluded from the pose times
        obj_ids = sorted({obj_id for im in images for obj_id in targets[im]})
        if on_cuda:
            torch.cuda.synchronize()
        t_onboard = time.perf_counter()
        bank_info = {}
        warm_render = cfg.refine_iters > 0 or cfg.n_hyp > 1
        for obj_id in obj_ids:
            entry = bank.get(obj_id)
            bank_info[str(obj_id)] = {
                "shared": bool(entry["shared"]),
                "seg_iou": None if math.isnan(float(entry["seg_iou"])) else float(entry["seg_iou"]),
                "n_truncated": int(entry["n_truncated"]),
                "mean_valid_patches": float(entry["valid"].sum(1).float().mean()),
            }
            if warm_render:
                # load the mesh and upload it to GL now, not in the per-image pose time
                renderer.render(
                    obj_id, entry["poses"][0].cpu().numpy(), entry["K"][0].cpu().numpy(),
                    (args.crop_size, args.crop_size),
                )
        if on_cuda:
            torch.cuda.synchronize()
        onboarding_time = time.perf_counter() - t_onboard
        n_shared = sum(info["shared"] for info in bank_info.values())
        logging.info(f"onboarding: {len(obj_ids)} objects in {onboarding_time:.1f}s")
        estimator.timers.clear()
        tpl_views = {}  # obj_id -> (T, 3) viewing directions of the template poses

        rows, records = [], []
        n_targets = n_instances = n_fallbacks = 0  # targets: (image, object) pairs, instances: sum of inst_count
        pose_times, total_times = [], []
        for n, (scene_id, im_id) in enumerate(images, 1):
            image = load_rgb_uint8(reader.image_path(scene_id, im_id))
            K = reader.K(scene_id, im_id)
            gts = reader.gt(scene_id, im_id)
            targets_img = targets[(scene_id, im_id)]
            dets_img = detections.get((scene_id, im_id), [])
            dets = []
            for obj_id, inst_count in targets_img.items():
                cand = sorted(
                    (d for d in dets_img if int(d["category_id"]) == obj_id),
                    key=lambda d: -float(d["score"]),
                )
                for d in cand[:inst_count]:
                    det = {"obj_id": obj_id, "mask": det_mask(d), "score": float(d["score"])}
                    g = match_gt(det["mask"], obj_id, K, gts) if gts is not None else None
                    if g is not None:
                        det["gt_R"], det["gt_t"] = g["R"], g["t"]
                    dets.append(det)
            n_targets += len(targets_img)
            n_instances += sum(targets_img.values())

            if on_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = estimator.estimate_image(image, K, dets)
            if on_cuda:
                torch.cuda.synchronize()
            pose_time = time.perf_counter() - t0
            det_time = float(dets_img[0].get("time", 0.0)) if dets_img else 0.0
            pose_times.append(pose_time)
            total_times.append(det_time + pose_time)

            ests = []
            for det, res in zip(dets, results):
                if res is None:
                    continue
                rows.append({
                    "scene_id": scene_id, "im_id": im_id, "obj_id": res["obj_id"],
                    "score": res["score"], "R": res["R"], "t": res["t"],
                    "time": det_time + pose_time,
                })
                rec = {
                    "scene_id": scene_id, "im_id": im_id, "obj_id": res["obj_id"],
                    "score": res["score"], "template": res["template"],
                    "n_matches": res["n_matches"], "n_inliers": res["n_inliers"],
                    "fallback": res["fallback"], "sim": res["sim"],
                    "refine_steps": res["refine_steps"], "fm_score": res["fm_score"],
                    "roll": res["roll"], "hyp_rank": res["hyp_rank"],
                    "rot_err": None, "trans_err": None, "view_err": None, "view_err_best": None,
                    "z_err_rel": None, "lat_err": None,
                }
                if "gt_R" in det:
                    R_e, t_e, R_g, t_g = res["R"], res["t"], det["gt_R"], det["gt_t"]
                    cos = (np.trace(R_e @ R_g.T) - 1.0) / 2.0
                    rec["rot_err"] = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                    rec["trans_err"] = float(np.linalg.norm(t_e - t_g))
                    rec["z_err_rel"] = float((t_e[2] - t_g[2]) / t_g[2])  # signed, dimensionless
                    rec["lat_err"] = float(np.linalg.norm((t_e - t_g)[:2]))  # mm
                    if res["obj_id"] not in tpl_views:
                        P = bank.get(res["obj_id"])["poses"].double().cpu().numpy()
                        tpl_views[res["obj_id"]] = view_direction(P[:, :3, :3], P[:, :3, 3])
                    c_tpl = tpl_views[res["obj_id"]]
                    c_gt = view_direction(R_g, t_g)
                    angles = np.degrees(np.arccos(np.clip(c_tpl @ c_gt, -1.0, 1.0)))  # (T,)
                    rec["view_err"] = float(angles[res["template"]])
                    rec["view_err_best"] = float(angles.min())
                records.append(rec)
                n_fallbacks += int(res["fallback"])
                ests.append({"obj_id": res["obj_id"], "R": res["R"], "t": res["t"]})

            if quick is not None and gts is not None:
                quick.add_image(scene_id, im_id, K, image.shape[1], targets_img, ests, gts)
                n_eval_images += 1

            if n % 50 == 0 or n == len(images):
                logging.info(
                    f"[{n}/{len(images)}] estimates {len(rows)}, fallbacks {n_fallbacks}, "
                    f"mean pose time {sum(pose_times) / len(pose_times):.3f}s/image"
                )

        save_bop_csv(csv_path, rows)
        quick_summary = quick.summary() if quick is not None and n_eval_images > 0 else None
        mean_matches = mean_or_none([r["n_matches"] for r in records])
        mean_inliers = mean_or_none([r["n_inliers"] for r in records])
        mean_pose_time = mean_or_none(pose_times)
        mean_total_time = mean_or_none(total_times)
        diag = summarize_diag(records)
        n_img = max(len(images), 1)
        timers = {name: estimator.timers.get(name, 0.0) / n_img for name in TIMER_STAGES}
        stats = {
            "args": vars(args),
            "counts": {
                "images": len(images), "targets": n_targets, "instances": n_instances,
                "estimates": len(rows), "fallbacks": n_fallbacks,
            },
            "onboarding_time": onboarding_time,
            "bank": bank_info,
            "mean_n_matches": mean_matches,
            "mean_n_inliers": mean_inliers,
            "mean_pose_time_per_image": mean_pose_time,
            "mean_total_time_per_image": mean_total_time,
            "timers": timers,
            "diag": diag,
            "quick_eval": quick_summary,
            "estimates": records,
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=1)

        line = (
            f"done: {csv_path} | images {len(images)}, targets {n_targets}, instances {n_instances}, "
            f"estimates {len(rows)}, fallbacks {n_fallbacks} | mean matches {fmt(mean_matches)}, "
            f"inliers {fmt(mean_inliers)} | pose time {fmt(mean_pose_time)}s, total time {fmt(mean_total_time)}s/image"
        )
        if quick_summary is not None:
            line += (
                f" | quick eval: AR_mssd {quick_summary['AR_mssd']:.3f}, AR_mspd {quick_summary['AR_mspd']:.3f}, "
                f"AR_mean2 {quick_summary['AR_mean2']:.3f}, recall_mssd_0.1 {quick_summary['recall_mssd_0.1']:.3f}, "
                f"recall_mspd_10 {quick_summary['recall_mspd_10']:.3f}"
            )
        else:
            line += " | quick eval: none"
        if diag is not None:
            line += (
                f" | diag (n={diag['n']}): rot_err {diag['rot_err_mean']:.1f}/{diag['rot_err_median']:.1f} deg, "
                f"trans_err {diag['trans_err_mean']:.1f}/{diag['trans_err_median']:.1f} mm, "
                f"view_err {diag['view_err_mean']:.1f}/{diag['view_err_median']:.1f} deg, "
                f"view_err_best {diag['view_err_best_mean']:.1f}/{diag['view_err_best_median']:.1f} deg (mean/median), "
                f"rot<15 {diag['frac_rot_lt_15']:.3f}, rot<30 {diag['frac_rot_lt_30']:.3f}, "
                f"view_nearest {diag['frac_view_nearest']:.3f}, view<40 {diag['frac_view_lt_40']:.3f}, "
                f"z_err_rel median {fmt_pct(diag['z_err_rel_median'])} (rot<15: {fmt_pct(diag['z_err_rel_median_rot15'])}), "
                f"lat_err median {fmt_mm(diag['lat_err_median'])}"
            )
        else:
            line += " | diag: none"
        line += " | timers (s/image): " + ", ".join(f"{k} {v:.3f}" for k, v in timers.items())
        line += f" | onboarding {onboarding_time:.1f}s"
        line += f" | templates shared {n_shared}/{len(obj_ids)} objects"
        logging.info(line)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
