import logging
import os
import argparse
import glob

import cv2
import distinctipy
import numpy as np
import torch
from PIL import Image
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from skimage.feature import canny
from skimage.morphology import binary_dilation
from torchvision.utils import save_image

from src.model.loss import Similarity
from src.model.utils import Detections, convert_npz_to_json, rle_to_mask
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import save_json_bop23

# set level logging
logging.basicConfig(level=logging.INFO)

def visualize(rgb, detections, save_path="./tmp/tmp.png"):
    """
    Draw the selected detections on top of the query image.

    `detections` is the flat list of BOP-style dicts produced by
    `convert_npz_to_json` (one entry per detection).
    """
    img = np.array(rgb).copy()
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    num_colors = max(len(detections), 1)
    colors = distinctipy.get_colors(num_colors)
    alpha = 0.33

    mask = np.zeros((height, width), dtype=bool)
    for det in detections:
        det_mask = rle_to_mask(det["segmentation"])
        mask |= det_mask
        edge = canny(det_mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        # category_id is 1-based; keep the index inside the palette for any value
        temp_id = int(det["category_id"]) % len(colors)

        r = int(255 * colors[temp_id][0])
        g = int(255 * colors[temp_id][1])
        b = int(255 * colors[temp_id][2])
        img[det_mask, 0] = alpha * r + (1 - alpha) * img[det_mask, 0]
        img[det_mask, 1] = alpha * g + (1 - alpha) * img[det_mask, 1]
        img[det_mask, 2] = alpha * b + (1 - alpha) * img[det_mask, 2]
        img[edge, :] = 255

    img = Image.fromarray(np.uint8(img))
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    img.save(save_path)
    prediction = Image.open(save_path)
    mask_pil = Image.fromarray(mask)

    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new("RGB", (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat, mask_pil


def run_inference(template_dir, rgb_path, mask_path, num_max_dets, conf_threshold, stability_score_thresh):
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="inference_custom.yaml")
    # only the SAM2 segmentor is supported
    cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    metric = Similarity()
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    # DINOv3 randomly rescales its RoPE coordinates in train mode, which makes the
    # descriptors non-deterministic; switch every submodule to inference behaviour
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.move_to_device(device)

    logging.info("Initializing template")
    # render_custom.sh writes the templates as "%06d.png"; restrict the
    # glob to digit-prefixed names so masks or visualisations written into
    # template_dir by earlier runs can never be picked up as templates
    mask_abs = os.path.abspath(mask_path)
    template_paths = sorted(
        path
        for path in glob.glob(f"{template_dir}/[0-9]*.png")
        if os.path.abspath(path) != mask_abs
    )
    if len(template_paths) == 0:
        raise FileNotFoundError(f"No template (*.png) found in {template_dir}")
    boxes, templates = [], []
    for path in template_paths:
        image = Image.open(path)
        boxes.append(image.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        templates.append(image)

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    save_image(templates, f"{template_dir}/free6d_results/templates.png", nrow=7)
    ref_feats = model.descriptor_model.compute_features(
        templates, token_name="x_norm_clstoken"
    )
    logging.info(f"Ref feats: {ref_feats.shape}")

    # run inference
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    decriptors = model.descriptor_model.forward(np.array(rgb), detections)

    save_path = f"{template_dir}/free6d_results/detection"

    # get scores per proposal
    if len(detections) > 0:
        scores = metric(decriptors[:, None, :], ref_feats[None, :, :])
        score_per_detection = torch.topk(scores, k=5, dim=-1)[0]
        score_per_detection = torch.mean(score_per_detection, dim=-1)

        # get top-k detections (k cannot exceed the number of proposals)
        k = min(num_max_dets, len(score_per_detection))
        scores, index = torch.topk(score_per_detection, k=k, dim=-1)

        # keep only the top-k proposals whose score passes the threshold;
        # filter scores and boxes with the same mask so they stay aligned
        keep = scores > conf_threshold
        detections.filter(index[keep])
        scores = scores[keep]
        detections.add_attribute("scores", scores)
        detections.add_attribute("object_ids", torch.zeros_like(scores))
        detections.to_numpy()
    else:
        logging.info("No proposal was generated, writing empty predictions")
        detections = Detections(
            {
                "masks": np.zeros((0, np.array(rgb).shape[0], np.array(rgb).shape[1]), dtype=bool),
                "boxes": np.zeros((0, 4), dtype=np.int64),
            }
        )
        detections.add_attribute("scores", np.zeros((0,), dtype=np.float32))
        detections.add_attribute("object_ids", np.zeros((0,), dtype=np.int64))

    detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
    # convert_npz_to_json returns (detections, detections_with_score_distribution)
    detections, _ = detections
    save_json_bop23(save_path + ".json", detections)
    vis_img, mask_img = visualize(rgb, detections)
    vis_img.save(f"{template_dir}/free6d_results/vis.png")

    mask_img.save(mask_path)
    logging.info(f"Saved results to {template_dir}/free6d_results")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_dir", nargs="?", help="Path to root directory of the template")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument(
        "--mask_path",
        nargs="?",
        default=None,
        help="Path where the predicted mask is written (default: <template_dir>/free6d_results/mask.png)",
    )
    parser.add_argument("--num_max_dets", nargs="?", default=1, type=int, help="Number of max detections")
    parser.add_argument("--confg_threshold", nargs="?", default=0.5, type=float, help="Confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.85, type=float, help="stability_score_thresh of SAM2 (0.97 of SAM v1 is too strict for SAM2)")
    args = parser.parse_args()

    os.makedirs(f"{args.template_dir}/free6d_results", exist_ok=True)
    mask_path = args.mask_path or f"{args.template_dir}/free6d_results/mask.png"
    mask_dir = os.path.dirname(os.path.abspath(mask_path))
    os.makedirs(mask_dir, exist_ok=True)
    run_inference(
        args.template_dir,
        args.rgb_path,
        mask_path,
        num_max_dets=args.num_max_dets,
        conf_threshold=args.confg_threshold,
        stability_score_thresh=args.stability_score_thresh,
    )
