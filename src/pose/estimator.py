"""Training-free 6D pose of segmented objects: DINOv3 patch matching against rendered templates, PnP, optional render-and-match refinement."""
import collections
import contextlib
import time
from dataclasses import dataclass

import numpy as np
import torch

from src.pose.geometry import (
    backproject_to_model,
    box_to_square_crop,
    crop_intrinsics,
    crop_resize,
    mask_to_box,
    normalize_rgb,
    patch_centers,
    patch_pool_centers,
    patch_pool_mask,
    patch_pool_xyz,
    rot90_points_back,
)
from src.pose.matching import match_templates, mutual_nn
from src.pose.pnp import pose_from_template, solve_pnp_ransac


@dataclass
class PoseConfig:
    crop_size: int = 448
    crop_pad: float = 1.1
    q_min_cover: float = 0.5  # min mask coverage of a query patch
    topk: int = 5  # templates passed to PnP
    pnp_thr: float = 20.0  # RANSAC reprojection threshold (px in the query crop)
    pnp_iters: int = 400
    min_inliers: int = 8
    refine_iters: int = 0  # render-and-match refinement steps, 0 = off
    refine_radius: float = 0.35  # match gate as a fraction of crop_size, <= 0 disables it
    q_rolls: int = 4  # query rotations by multiples of 90 deg: 1 (none), 2 (0/180) or 4
    q_centroid: bool = False  # query 2D points = centroid of the mask pixels of each patch instead of the patch centre
    n_hyp: int = 1  # PnP hypotheses passed to the render-and-compare scoring / refinement
    oracle: bool = False  # diagnostic: only use the template nearest to the ground-truth view


def _pose_matrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _np64(x):
    return x.detach().double().cpu().numpy()


def view_direction(R, t):
    """Camera-centre direction in the object frame, c = -R^T t / |t|. R (..., 3, 3), t (..., 3) numpy."""
    c = -np.einsum("...ji,...j->...i", R, t)
    return c / np.linalg.norm(t, axis=-1, keepdims=True)


class PoseEstimator:
    def __init__(self, extractor, bank, renderer, cfg, device="cuda"):
        assert cfg.q_rolls in (1, 2, 4), f"q_rolls must be 1, 2 or 4, got {cfg.q_rolls}"
        self.extractor = extractor
        self.bank = bank
        self.renderer = renderer
        self.cfg = cfg
        self.device = device
        self.ks = [i * (4 // cfg.q_rolls) for i in range(cfg.q_rolls)]  # query rot90 steps, contains 0
        self.timers = collections.defaultdict(float)  # seconds per stage

    @contextlib.contextmanager
    def _timed(self, name):
        on_cuda = torch.device(self.device).type == "cuda"
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if on_cuda:
                torch.cuda.synchronize()
            self.timers[name] += time.perf_counter() - t0

    @torch.no_grad()
    def estimate_image(self, image, K, dets):
        """
        `image`: (H, W, 3) uint8 RGB, `K`: 3x3, `dets`: [{"obj_id", "mask" (H, W) bool, "score"}],
        optionally with "gt_R" (3, 3) and "gt_t" (3,) (only used when cfg.oracle).
        Returns one entry per det (None if its mask is empty), else a dict with R (3, 3),
        t (3,) in mm, obj_id, score, template, n_matches, n_inliers, fallback, sim,
        refine_steps, fm_score, roll (query rot90 step of the chosen hypothesis) and
        hyp_rank (index of the chosen hypothesis among the candidates).
        """
        cfg, S, dev, ks = self.cfg, self.cfg.crop_size, self.device, self.ks
        K = np.asarray(K, dtype=np.float64)
        results = [None] * len(dets)
        queries, q_imgs = [], []
        with self._timed("prep"):
            img = normalize_rgb(torch.from_numpy(np.ascontiguousarray(image)).to(dev))
            for i, det in enumerate(dets):
                mask = det["mask"]
                box = mask_to_box(mask)
                if box is None:
                    continue
                crop = box_to_square_crop(box, cfg.crop_pad)
                mask_t = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
                m_c = crop_resize(mask_t[None].float(), *crop, S)[0] > 0.5
                q_img = crop_resize(img, *crop, S) * m_c
                q_imgs.extend(torch.rot90(q_img, k, dims=(1, 2)) for k in ks)
                queries.append((i, crop, m_c, crop_intrinsics(K, *crop, S)))
        if not queries:
            return results
        with self._timed("extract"):
            feats = self.extractor.extract(torch.stack(q_imgs))  # (B * len(ks), N, C)
        nk = len(ks)
        for b, (i, crop, m_c, K_c) in enumerate(queries):
            det = dets[i]
            results[i] = self._estimate_one(det, feats[b * nk:(b + 1) * nk], m_c, K, K_c, crop)
            results[i]["score"] = float(det["score"])
        return results

    def _estimate_one(self, det, feats_k, m_c, K, K_c, crop):
        """`feats_k`: (len(ks), N, C) features of the query crop rotated by every k in self.ks."""
        cfg, S, dev, ks = self.cfg, self.cfg.crop_size, self.device, self.ks
        obj_id = int(det["obj_id"])
        centers = patch_centers(S, dev)
        entry = self.bank.get(obj_id)

        # match every query rotation against all templates
        q_valid, M, q_xy = [], [], []
        for ki, k in enumerate(ks):
            with self._timed("prep"):
                m_k = torch.rot90(m_c, k, dims=(0, 1))
                q_valid.append(patch_pool_mask(m_k) >= cfg.q_min_cover)
                q_xy.append(patch_pool_centers(m_k) if cfg.q_centroid else centers)  # 2D points in the rotated crop
            with self._timed("match"):
                M.append(match_templates(feats_k[ki][q_valid[ki]], entry["feats"], entry["valid"]))

        # candidate (k, template) pairs ranked by score_sum, the first topk are kept
        scores = torch.stack([m["score_sum"] for m in M])  # (len(ks), T)
        T = scores.shape[1]
        t_star = None
        if cfg.oracle and "gt_R" in det and "gt_t" in det:
            c_gt = view_direction(
                np.asarray(det["gt_R"], dtype=np.float64), np.asarray(det["gt_t"], dtype=np.float64)
            )
            P = _np64(entry["poses"])
            t_star = int(np.argmax(view_direction(P[:, :3, :3], P[:, :3, 3]) @ c_gt))
            allowed = torch.full_like(scores, float("-inf"))
            allowed[:, t_star] = scores[:, t_star]
            scores = allowed
        top = torch.topk(scores.flatten(), min(cfg.topk, scores.numel()))
        pairs = [
            (idx // T, idx % T)
            for val, idx in zip(top.values.tolist(), top.indices.tolist())
            if val > float("-inf")
        ]

        # PnP on the mutual matches of every pair, hypotheses sorted by (inliers, score_sum)
        hyps = []
        with self._timed("pnp"):
            for ki, ti in pairs:
                sel = M[ki]["mutual"][ti]
                n_matches = int(sel.sum())
                if n_matches < 4:
                    continue
                pts2d = _np64(rot90_points_back(q_xy[ki][q_valid[ki]][sel], ks[ki], S))
                pts3d = _np64(entry["xyz"][ti][M[ki]["nn_qt"][ti][sel]])
                res = solve_pnp_ransac(pts3d, pts2d, K_c, cfg.pnp_thr, cfg.pnp_iters)
                if not res["ok"]:
                    continue
                hyps.append({
                    "R": res["R"], "t": res["t"], "template": ti, "roll": ks[ki],
                    "n_matches": n_matches, "n_inliers": len(res["inliers"]),
                    "sim": float(M[ki]["score_sum"][ti]), "fallback": False,
                })
        hyps.sort(key=lambda h: (h["n_inliers"], h["sim"]), reverse=True)

        if hyps and hyps[0]["n_inliers"] >= cfg.min_inliers:
            cands = [h for h in hyps if h["n_inliers"] >= cfg.min_inliers][: cfg.n_hyp]
        else:
            ti = t_star if t_star is not None else int(torch.argmax(M[0]["score_sum"]))  # ks[0] == 0
            T_t = _np64(entry["poses"][ti])
            R, t = pose_from_template(
                T_t[:3, :3], T_t[:3, 3], _np64(entry["K"][ti]), _np64(entry["crops"][ti]), K, crop,
            )
            cands = [{
                "R": R, "t": t, "template": ti, "roll": 0,
                "n_matches": int(M[0]["count"][ti]), "n_inliers": 0,
                "sim": float(M[0]["score_sum"][ti]), "fallback": True,
            }]

        # render-and-compare: keep the candidate with the best featuremetric score
        hyp_rank, chosen = 0, cands[0]
        R, t, refine_steps, fm_score = chosen["R"], chosen["t"], 0, None
        if cfg.refine_iters > 0 or cfg.n_hyp > 1:
            best_fm = float("-inf")
            for rank, cand in enumerate(cands):
                with self._timed("refine"):
                    R_r, t_r, steps, fm = self._refine(
                        obj_id, feats_k[0], q_valid[0], q_xy[0], K_c, cand["R"], cand["t"]
                    )
                if fm is not None and fm > best_fm:
                    best_fm = fm
                    hyp_rank, chosen = rank, cand
                    R, t, refine_steps, fm_score = R_r, t_r, steps, fm
        return {
            "R": np.asarray(R, dtype=np.float64),
            "t": np.asarray(t, dtype=np.float64),
            "obj_id": obj_id,
            "template": int(chosen["template"]),
            "n_matches": int(chosen["n_matches"]),
            "n_inliers": int(chosen["n_inliers"]),
            "fallback": chosen["fallback"],
            "sim": float(chosen["sim"]),
            "refine_steps": refine_steps,
            "fm_score": fm_score,
            "roll": int(chosen["roll"]),
            "hyp_rank": hyp_rank,
        }

    def _refine(self, obj_id, Fq, q_valid, q_pts, K_c, R, t):
        """
        Render-and-match refinement in the query-crop camera. Every step renders the current
        pose, scores it by the featuremetric agreement with the query at identical patch
        positions, and updates the pose by PnP on query/render mutual matches. The best
        scoring rendered pose is returned with the number of PnP updates it results from.
        `q_pts`: (N, 2) 2D points of the query patches (patch centres or mask centroids) used by PnP;
        the radius gate uses the plain patch centres.
        """
        cfg, S, dev = self.cfg, self.cfg.crop_size, self.device
        centers = patch_centers(S, dev)
        cur = (R, t)
        best = None  # (pose, fm, number of PnP updates the pose results from)
        for it in range(cfg.refine_iters + 1):
            rgb, depth = self.renderer.render(obj_id, _pose_matrix(*cur), K_c, (S, S))
            depth_t = torch.from_numpy(depth).to(dev)
            r_mask = depth_t > 0
            if int(r_mask.sum()) < 1024:
                break
            Fr = self.extractor.extract(
                (normalize_rgb(torch.from_numpy(rgb).to(dev)) * r_mask)[None]
            )[0]
            xyz_map, _ = backproject_to_model(depth_t, K_c, cur[0], cur[1])
            r_xyz, r_cover = patch_pool_xyz(xyz_map, r_mask)
            r_valid = r_cover >= 0.5
            both = q_valid & r_valid
            union = q_valid | r_valid
            fm = float((Fq * Fr).sum(-1)[both].sum() / union.sum().clamp(min=1))
            if best is None or fm > best[1]:
                best = (cur, fm, it)  # the pose scored at loop index `it` results from `it` updates
            else:
                break  # score dropped: keep the previous best
            if it == cfg.refine_iters:
                break
            if not bool(q_valid.any()) or not bool(r_valid.any()):
                break
            sim = Fq[q_valid] @ Fr[r_valid].T
            if cfg.refine_radius > 0:
                dist = torch.cdist(centers[q_valid], centers[r_valid])
                sim = sim.masked_fill(dist > cfg.refine_radius * S, float("-inf"))
            ia, ib = mutual_nn(sim)
            res = solve_pnp_ransac(
                _np64(r_xyz[r_valid][ib]), _np64(q_pts[q_valid][ia]), K_c, cfg.pnp_thr, cfg.pnp_iters
            )
            if not res["ok"] or len(res["inliers"]) < cfg.min_inliers:
                break
            cur = (res["R"], res["t"])
        if best is None:
            return R, t, 0, None
        return best[0][0], best[0][1], best[2], best[1]
