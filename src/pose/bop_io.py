"""BOP dataset I/O for the pose stage: targets, detections, scene files, result csv, quick local recalls."""
import json
import os
import os.path as osp

import numpy as np

from bop_toolkit_lib import inout, misc, pose_error
from src.dataloader.base_bop import BaseBOP
from src.model.utils import rle_to_mask

TARGETS_FILE = "test_targets_bop19.json"


def split_name(dataset_name):
    """Test split folder of a dataset (T-LESS and HB are evaluated on the PrimeSense images)."""
    return "test_primesense" if dataset_name in ("tless", "hb") else "test"


def cad_dir(dataset_dir, dataset_name):
    """Folder with the CAD models (T-LESS uses the CAD ones, not the reconstructed ones)."""
    return osp.join(dataset_dir, "models_cad" if dataset_name == "tless" else "models")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_models_info(dataset_dir, dataset_name):
    """{obj_id: models_info entry} of the CAD models."""
    info = _load_json(osp.join(cad_dir(dataset_dir, dataset_name), "models_info.json"))
    return {int(k): v for k, v in info.items()}


def load_targets(dataset_dir):
    """{(scene_id, im_id): {obj_id: inst_count}} of the BOP'19 test targets."""
    targets = {}
    for item in _load_json(osp.join(dataset_dir, TARGETS_FILE)):
        per_image = targets.setdefault((int(item["scene_id"]), int(item["im_id"])), {})
        obj_id = int(item["obj_id"])
        per_image[obj_id] = per_image.get(obj_id, 0) + int(item["inst_count"])
    return targets


def load_detections(json_path):
    """{(scene_id, image_id): [detections as stored in the file]} of a run_inference.py result."""
    detections = {}
    for det in _load_json(json_path):
        detections.setdefault((int(det["scene_id"]), int(det["image_id"])), []).append(det)
    return detections


def det_mask(det):
    """Boolean (H, W) mask of a detection (uncompressed RLE in det["segmentation"])."""
    return rle_to_mask(det["segmentation"])


class SceneReader:
    """Images, intrinsics and ground truth of a BOP test split; per-scene json files are cached."""

    def __init__(self, dataset_dir, dataset_name):
        self.split_dir = osp.join(dataset_dir, split_name(dataset_name))
        self._json = {}
        self._image_dirs = {}

    def _scene_json(self, scene_id, name):
        """Content of <scene>/<name>.json, None if the file does not exist."""
        key = (scene_id, name)
        if key not in self._json:
            path = osp.join(self.split_dir, f"{scene_id:06d}", f"{name}.json")
            self._json[key] = _load_json(path) if osp.isfile(path) else None
        return self._json[key]

    def image_path(self, scene_id, im_id):
        if scene_id not in self._image_dirs:
            scene_path = osp.join(self.split_dir, f"{scene_id:06d}")
            self._image_dirs[scene_id] = BaseBOP.find_image_dir(scene_path)
        image_dir, ext = self._image_dirs[scene_id]
        return osp.join(image_dir, f"{im_id:06d}{ext}")

    def K(self, scene_id, im_id):
        cams = self._scene_json(scene_id, "scene_camera")
        return np.asarray(cams[str(im_id)]["cam_K"], dtype=np.float64).reshape(3, 3)

    def gt(self, scene_id, im_id):
        """[{"obj_id", "R", "t", "visib_fract"}] of an image, None if the split has no scene_gt.json."""
        gts = self._scene_json(scene_id, "scene_gt")
        if gts is None:
            return None
        infos = self._scene_json(scene_id, "scene_gt_info")
        out = []
        for i, g in enumerate(gts.get(str(im_id), [])):
            visib = 1.0
            if infos is not None:
                visib = float(infos[str(im_id)][i]["visib_fract"])
            out.append(
                {
                    "obj_id": int(g["obj_id"]),
                    "R": np.asarray(g["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
                    "t": np.asarray(g["cam_t_m2c"], dtype=np.float64).reshape(3),
                    "visib_fract": visib,
                }
            )
        return out


def save_bop_csv(path, rows):
    """Write pose estimates (dicts with scene_id, im_id, obj_id, score, R, t, time) as a BOP'19 csv."""
    results = [
        {
            "scene_id": int(r["scene_id"]),
            "im_id": int(r["im_id"]),
            "obj_id": int(r["obj_id"]),
            "score": float(r["score"]),
            "R": np.asarray(r["R"], dtype=np.float64).reshape(3, 3),
            "t": np.asarray(r["t"], dtype=np.float64).reshape(3, 1),
            "time": float(r["time"]),
        }
        for r in rows
    ]
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    inout.save_bop_results(path, results, version="bop19")


class QuickEval:
    """Fast local proxy of the BOP MSSD / MSPD recalls (no rendering, so no VSD); for ablations only, official numbers come from bop_toolkit."""

    def __init__(self, dataset_dir, dataset_name, max_sym_disc_step=0.01):
        self.eval_dir = osp.join(dataset_dir, "models_eval")
        info = _load_json(osp.join(self.eval_dir, "models_info.json"))
        self.models_info = {int(k): v for k, v in info.items()}
        self.max_sym_disc_step = max_sym_disc_step
        self._pts = {}
        self._syms = {}
        self.n_targets = 0
        self.n_est = 0
        self.err_mssd = []  # errors of the matched (est, gt) pairs, normalized by the diameter
        self.err_mspd = []  # errors of the matched pairs, in pixels at a 640 px image width

    def _model(self, obj_id):
        if obj_id not in self._pts:
            path = osp.join(self.eval_dir, f"obj_{obj_id:06d}.ply")
            self._pts[obj_id] = inout.load_ply(path)["pts"]
            self._syms[obj_id] = misc.get_symmetry_transformations(
                self.models_info[obj_id], self.max_sym_disc_step
            )
        return self._pts[obj_id], self._syms[obj_id], self.models_info[obj_id]["diameter"]

    @staticmethod
    def _greedy(err):
        """One-to-one assignment by ascending error, errors of the assigned pairs."""
        used_e, used_g, out = set(), set(), []
        for flat in np.argsort(err, axis=None):
            e, g = np.unravel_index(flat, err.shape)
            if e in used_e or g in used_g:
                continue
            used_e.add(e)
            used_g.add(g)
            out.append(float(err[e, g]))
        return out

    def add_image(self, scene_id, im_id, K, im_width, targets, ests, gts):
        """
        `targets`: {obj_id: inst_count} of the image, `ests`: dicts with obj_id, R, t,
        `gts`: dicts as returned by SceneReader.gt.
        """
        K = np.asarray(K, dtype=np.float64)
        for obj_id, inst_count in targets.items():
            self.n_targets += int(inst_count)
            est_list = [e for e in ests if e["obj_id"] == obj_id]
            gt_list = [g for g in gts if g["obj_id"] == obj_id and g["visib_fract"] >= 0.1]
            self.n_est += len(est_list)
            if not est_list or not gt_list:
                continue
            pts, syms, diameter = self._model(obj_id)
            e_mssd = np.zeros((len(est_list), len(gt_list)))
            e_mspd = np.zeros_like(e_mssd)
            for i, est in enumerate(est_list):
                R_e = np.asarray(est["R"], dtype=np.float64).reshape(3, 3)
                t_e = np.asarray(est["t"], dtype=np.float64).reshape(3, 1)
                for j, gt in enumerate(gt_list):
                    R_g = np.asarray(gt["R"], dtype=np.float64).reshape(3, 3)
                    t_g = np.asarray(gt["t"], dtype=np.float64).reshape(3, 1)
                    e_mssd[i, j] = pose_error.mssd(R_e, t_e, R_g, t_g, pts, syms) / diameter
                    e_mspd[i, j] = pose_error.mspd(R_e, t_e, R_g, t_g, K, pts, syms) * 640.0 / im_width
            self.err_mssd += self._greedy(e_mssd)
            self.err_mspd += self._greedy(e_mspd)

    def summary(self):
        n = max(self.n_targets, 1)
        mssd = np.asarray(self.err_mssd, dtype=np.float64)
        mspd = np.asarray(self.err_mspd, dtype=np.float64)
        th_mssd = np.arange(1, 11) * 0.05
        th_mspd = np.arange(1, 11) * 5.0
        ar_mssd = float(np.mean([(mssd < th).sum() / n for th in th_mssd]))
        ar_mspd = float(np.mean([(mspd < th).sum() / n for th in th_mspd]))
        return {
            "n_targets": int(self.n_targets),
            "n_est": int(self.n_est),
            "AR_mssd": ar_mssd,
            "AR_mspd": ar_mspd,
            "AR_mean2": 0.5 * (ar_mssd + ar_mspd),
            "recall_mssd_0.1": float((mssd < 0.1).sum() / n),
            "recall_mspd_10": float((mspd < 10.0).sum() / n),
        }
