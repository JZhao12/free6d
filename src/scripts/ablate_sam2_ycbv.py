"""Subset ablation of SAM2 automatic-mask-generator settings for the localisation stage on YCB-V.

usage: python -m src.scripts.ablate_sam2_ycbv [stride] [json list of run() kwargs]
   e.g. python -m src.scripts.ablate_sam2_ycbv 12
        python -m src.scripts.ablate_sam2_ycbv 12 '[{"name":"stab0.88","stab":0.88}]'
Requires the pyrender templates, their cached level-0 descriptors and scene_gt_coco.json
(see README, "Testing on BOP datasets").
Reports raw proposals / image, proposal recall vs GT (IoU>=0.5 / 0.75) and the
segm AP of the full localisation pipeline on the subset (pycocotools, same GT files as bop_toolkit)."""
import json, os.path as osp, sys, logging, time, io, contextlib
from collections import defaultdict
import numpy as np, torch, torchvision
from PIL import Image
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from src.model.utils import Detections
from src.model.sam2 import CustomSam2AutomaticMaskGenerator
logging.getLogger().setLevel(logging.ERROR)
ROOT = "datasets/bop23_challenge/datasets/ycbv"
STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 12
CONFIGS = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None

from hydra.core.global_hydra import GlobalHydra; GlobalHydra.instance().clear()
with initialize(version_base=None, config_path="../../configs"):
    cfg = compose(config_name="run_inference", overrides=["dataset_name=ycbv"])
OmegaConf.set_struct(cfg, False)
model = instantiate(cfg.model); model.move_to_device("cuda"); model.eval()
desc = torch.load("datasets/bop23_challenge/datasets/templates_pyrender/ycbv/descriptors_dinov3_vitl16_pyrender_level0.pth", map_location="cuda")
model.set_reference_descriptors(desc)
sam = model.segmentor_model.predictor.model
base_kwargs = dict(points_per_batch=64, box_nms_thresh=0.7, min_mask_region_area=0, output_mode="binary_mask", segmentor_width_size=640)

# subset: every STRIDE-th target image (deterministic, spans all scenes)
targets = json.load(open(f"{ROOT}/test_targets_bop19.json"))
all_imgs = sorted({(t["scene_id"], t["im_id"]) for t in targets})
subset = all_imgs[::STRIDE]
# GT in COCO format from bop_toolkit's scene_gt_coco.json, image ids remapped to be unique
gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": str(i)} for i in range(1, 22)]}
scene_coco = {}
for sid, iid in subset:
    if sid not in scene_coco: scene_coco[sid] = json.load(open(f"{ROOT}/test/{sid:06d}/scene_gt_coco.json"))
    c = scene_coco[sid]; uid = sid * 10000 + iid
    im = next(i for i in c["images"] if i["id"] == iid); im = dict(im); im["id"] = uid; gt["images"].append(im)
    for a in c["annotations"]:
        if a["image_id"] == iid:
            a = dict(a); a["image_id"] = uid; a["id"] = len(gt["annotations"]) + 1; gt["annotations"].append(a)
n_gt_eval = sum(1 for a in gt["annotations"] if not a.get("ignore", 0))
print(f"subset: {len(subset)} images, {len(gt['annotations'])} GT instances ({n_gt_eval} evaluated)")
with contextlib.redirect_stdout(io.StringIO()):
    coco_gt = COCO(); coco_gt.dataset = gt; coco_gt.createIndex()
gt_masks = {}  # per image: list of (cat, bool mask) for non-ignored GT
for a in gt["annotations"]:
    if a.get("ignore", 0): continue
    seg = a["segmentation"]; rle = mu.frPyObjects(seg, seg["size"][0], seg["size"][1]) if isinstance(seg["counts"], list) else seg
    gt_masks.setdefault(a["image_id"], []).append((a["category_id"], torch.from_numpy(mu.decode(rle).astype(bool)).cuda()))

def coco_ap(results):
    if not results: return float("nan"), float("nan")
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(results); ev = COCOeval(coco_gt, coco_dt, "segm"); ev.params.imgIds = sorted(coco_gt.getImgIds())
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[8]  # AP, AR100

def run(name, stab=0.97, iou=0.8, pps=32, m2m=False, bf16=True, pred_json=None):
    amg = CustomSam2AutomaticMaskGenerator(sam=sam, points_per_side=pps, pred_iou_thresh=iou, stability_score_thresh=stab, use_m2m=m2m, **base_kwargs)
    results = []; n_raw = []; n_final = []; best_iou = []; t0 = time.time()
    pre = None
    if pred_json:  # replay an existing prediction file instead of running the model
        pre = defaultdict(list)
        for x in json.load(open(pred_json)): pre[(x["scene_id"], x["image_id"])].append(x)
    for sid, iid in subset:
        uid = sid * 10000 + iid
        rgb = np.array(Image.open(f"{ROOT}/test/{sid:06d}/rgb/{iid:06d}.png").convert("RGB"))
        if pre is not None:
            dets = pre[(sid, iid)]
            masks = [torch.from_numpy(mu.decode(mu.frPyObjects(x["segmentation"], 480, 640)).astype(bool)).cuda() for x in dets]
            raw_masks = torch.stack(masks) if masks else torch.zeros((0, 480, 640), dtype=torch.bool, device="cuda")
            n_raw.append(len(dets)); n_final.append(len(dets))
            for x in dets:
                seg = x["segmentation"]; rle = mu.frPyObjects(seg, seg["size"][0], seg["size"][1]); rle["counts"] = rle["counts"].decode("ascii")
                results.append({"image_id": uid, "category_id": x["category_id"], "score": x["score"], "segmentation": rle})
        else:
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
            with torch.no_grad(), ctx:
                props = amg.generate_masks(rgb); raw_masks = props["masks"]; n_raw.append(len(raw_masks))
                det = Detections(props); det.remove_very_small_detections(cfg.model.post_processing_config.mask_post_processing)
                if len(det) > 0:
                    q = model.descriptor_model(rgb, det)
                    sel, oid, sc, _ = model.find_matched_proposals(q)
                    det.filter(sel); det.add_attribute("scores", sc.float()); det.add_attribute("object_ids", oid)
                    if len(det) > 0: det.apply_nms_per_object_id(cfg.model.post_processing_config.nms_thresh)
            n_final.append(len(det))
            for k in range(len(det)):
                m = det.masks[k].cpu().numpy(); rle = mu.encode(np.asfortranarray(m.astype(np.uint8))); rle["counts"] = rle["counts"].decode("ascii")
                results.append({"image_id": uid, "category_id": int(det.object_ids[k]) + 1, "score": float(det.scores[k]), "segmentation": rle})
        # proposal recall: best IoU of any raw proposal with each GT instance
        for cat, gm in gt_masks.get(uid, []):
            if len(raw_masks) == 0: best_iou.append(0.0); continue
            inter = (raw_masks & gm).sum((1, 2)).float(); union = (raw_masks | gm).sum((1, 2)).float()
            best_iou.append(float((inter / union.clamp(min=1)).max()))
    ap, ar = coco_ap(results); bi = np.array(best_iou)
    print(f"{name:34s} raw/img {np.mean(n_raw):6.1f}  final/img {np.mean(n_final):5.1f}  recall@.5 {np.mean(bi>=.5):.3f}  recall@.75 {np.mean(bi>=.75):.3f}  subset segm AP {ap*100:5.1f}  AR100 {ar*100:5.1f}  ({(time.time()-t0)/len(subset):.2f}s/img)", flush=True)

PRED = "datasets/bop23_challenge/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json"
if CONFIGS is None:
    run("full-run json replayed (reference)", pred_json=PRED)
    run("stab0.97 iou0.80 pps32 bf16 (current)")
    run("stab0.97 iou0.80 pps32 fp32", bf16=False)
    run("stab0.95 (SAM2 default)", stab=0.95)
    run("stab0.92", stab=0.92)
    run("stab0.90", stab=0.90)
    run("stab0.85", stab=0.85)
    run("stab0.80 (custom script value)", stab=0.80)
    run("stab0.90 pps64", stab=0.90, pps=64)
    run("stab0.90 use_m2m", stab=0.90, m2m=True)
else:
    for c in CONFIGS: run(c.pop("name"), **c)
