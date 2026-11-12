import logging
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sam2
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def _resolve_model_cfg(model_cfg: str) -> str:
    """
    Resolve a SAM2 hydra config path.

    SAM2 ships its configs inside the `sam2` python package (sam2/configs/...).
    Paths are resolved relative to the SAM2 package so they work no matter which
    directory the script is launched from, and so that they never depend on
    this project's own Hydra search path (which would otherwise shadow them).
    """
    if os.path.isabs(model_cfg):
        return model_cfg
    # configs ship inside the sam2 package itself (sam2/configs/...), so anchor the
    # lookup on the imported package rather than on this file's location
    sam2_pkg_dir = os.path.dirname(os.path.abspath(sam2.__file__))
    candidates = [
        os.path.join(sam2_pkg_dir, model_cfg),
        model_cfg,
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Cannot find SAM2 model config '{model_cfg}'. Tried: {candidates}"
    )


def load_sam2(
    model_cfg: str,
    checkpoint_path: str,
    device: str = "cuda",
    apply_postprocessing: bool = False,
):
    """
    Build a SAM2 model from a config + checkpoint without going through
    `sam2.build_sam2`.

    `build_sam2` uses `hydra.compose`, which reuses the GlobalHydra singleton. When
    the pipeline is launched through `@hydra.main`, GlobalHydra is already initialized with
    this project's own search path, so SAM2 configs are never found and building raises
    `MissingConfigException`. Loading the YAML with `OmegaConf` and instantiating it
    directly avoids GlobalHydra entirely. This mirrors what `sam2.build_sam2` does
    internally (compose + instantiate + load_state_dict).
    """
    config_path = _resolve_model_cfg(model_cfg)
    logging.info(
        f"Loading SAM2: cfg={config_path}, ckpt={checkpoint_path}, device={device}"
    )

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    if apply_postprocessing:
        # same overrides as sam2.build_sam2(apply_postprocessing=True)
        OmegaConf.update(
            cfg,
            "model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability",
            True,
            merge=True,
        )
        OmegaConf.update(
            cfg,
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta",
            0.05,
            merge=True,
        )
        OmegaConf.update(
            cfg,
            "model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh",
            0.98,
            merge=True,
        )

    model = instantiate(cfg.model, _recursive_=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Failed to load SAM2 checkpoint.\n"
            f"Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}"
        )

    model = model.to(device)
    model.eval()
    logging.info("Loaded SAM2 checkpoint successfully")
    return model


class CustomSam2AutomaticMaskGenerator(SAM2AutomaticMaskGenerator):
    """
    Wrapper returning the proposal format of the detector:
      input : RGB uint8 HWC
      output: {"masks": torch.BoolTensor[N,H,W], "boxes": torch.FloatTensor[N,4] (xyxy)}
    """

    def __init__(
        self,
        sam,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.8,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        mask_threshold: float = 0.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids=None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",  # avoid pycocotools dependency
        use_m2m: bool = False,
        multimask_output: bool = True,
        segmentor_width_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            model=sam,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            mask_threshold=mask_threshold,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
            use_m2m=use_m2m,
            multimask_output=multimask_output,
            **kwargs,
        )
        self.segmentor_width_size = segmentor_width_size
        logging.info("Init CustomSam2AutomaticMaskGenerator done!")

    def preprocess_resize(self, image: np.ndarray) -> np.ndarray:
        orig_h, orig_w = image.shape[:2]
        new_w = int(self.segmentor_width_size)
        new_h = int(new_w * orig_h / orig_w)
        resized = cv2.resize(image, (new_w, new_h))  # (width, height)
        return resized

    def postprocess_resize(self, detections: Dict[str, torch.Tensor], orig_size) -> Dict[str, torch.Tensor]:
        orig_h, orig_w = orig_size
        masks = detections["masks"].unsqueeze(1).float()  # [N,1,h,w]
        masks = F.interpolate(masks, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[:, 0]
        detections["masks"] = masks > 0.5

        scale = orig_w / float(self.segmentor_width_size)
        boxes = detections["boxes"].float() * scale
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, orig_h - 1)
        detections["boxes"] = boxes
        return detections

    @torch.no_grad()
    def generate_masks(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        # image: RGB uint8 HWC
        if self.segmentor_width_size is not None:
            orig_size = image.shape[:2]
            image = self.preprocess_resize(image)

        anns: List[Dict[str, Any]] = super().generate(image)

        if len(anns) == 0:
            h, w = image.shape[:2]
            device = next(self.predictor.model.parameters()).device
            out = {
                "masks": torch.zeros((0, h, w), dtype=torch.bool, device=device),
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            }
            if self.segmentor_width_size is not None:
                out = self.postprocess_resize(out, orig_size)
            return out

        # Convert to the proposal format of the detector
        device = next(self.predictor.model.parameters()).device
        masks = [torch.from_numpy(a["segmentation"]).to(device=device) for a in anns]  # HW np -> torch
        masks = torch.stack(masks, dim=0).bool()  # [N,H,W]

        # SAM2 bbox is XYWH
        boxes_xywh = torch.tensor([a["bbox"] for a in anns], dtype=torch.float32, device=device)
        boxes_xyxy = boxes_xywh.clone()
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2]
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3]

        out = {"masks": masks, "boxes": boxes_xyxy}

        if self.segmentor_width_size is not None:
            out = self.postprocess_resize(out, orig_size)

        return out
