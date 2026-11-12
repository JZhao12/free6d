import glob
import logging
import multiprocessing
import os
import os.path as osp
import time
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from tqdm import tqdm

from src.model.utils import BatchedData, Detections, convert_npz_to_json
from src.utils.inout import save_json_bop23


class Free6DDetector(pl.LightningModule):
    """
    Container for the two localisation stages:
      1. `segmentor_model`  (SAM2)   : RGB image -> class-agnostic mask proposals
      2. `descriptor_model` (DINOv3) : proposals -> CLS-token descriptors
    plus the template matching used to assign each proposal to a reference object.

    For BOP evaluation (run_inference.py), `ref_dataset`, `dataset_name` and
    `name_prediction_file` are set before `Trainer.test`: `test_step` then writes one
    npz per image and `on_test_epoch_end` merges them into a BOP-format json.
    """

    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        matching_config,
        post_processing_config,
        onboarding_config=None,
        log_interval=5,
        log_dir=None,
        save_mask=True,
        **kwargs,
    ):
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.onboarding_config = onboarding_config
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.save_mask = save_mask
        self.ref_data = {"descriptors": None}

        # set by run_inference.py for BOP evaluation
        self.ref_dataset = None
        self.dataset_name = None
        self.name_prediction_file = None

        if self.log_dir is not None:
            os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)

        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info("Init Free6DDetector done!")

    def move_to_device(self, device=None):
        """Move both backbones to `device` (defaults to this module's device)."""
        device = torch.device(device) if device is not None else self.device
        self.descriptor_model.model = self.descriptor_model.model.to(device)
        self.descriptor_model.model.device = device
        self.segmentor_model.predictor.model = self.segmentor_model.predictor.model.to(
            device
        )
        logging.info(f"Moving models to {device} done!")

    def set_reference_descriptors(self, descriptors):
        """
        descriptors: N_objects x N_templates x D tensor of template descriptors.
        """
        self.ref_data["descriptors"] = descriptors

    def find_matched_proposals(self, proposal_descriptors):
        """
        Match proposal descriptors against the reference descriptors.

        Returns (idx_selected_proposals, pred_idx_objects, pred_scores,
        pred_score_distribution) following the matching rule of CNOS (Nguyen et al., 2023).
        """
        assert (
            self.ref_data["descriptors"] is not None
        ), "call set_reference_descriptors() first"
        scores = self.matching_config.metric(
            proposal_descriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            k = min(5, scores.shape[-1])
            score_per_proposal_and_object = torch.topk(scores, k=k, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest score
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_proposals

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        if len(idx_selected_proposals) > self.matching_config.max_num_instances:
            logging.info(
                f"Selecting top {self.matching_config.max_num_instances} instances ..."
            )
            _, idx = torch.topk(
                score_per_proposal[idx_selected_proposals],
                k=self.matching_config.max_num_instances,
            )
            idx_selected_proposals = idx_selected_proposals[idx]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        pred_scores = score_per_proposal[idx_selected_proposals]
        pred_score_distribution = score_per_proposal_and_object[idx_selected_proposals]
        return (
            idx_selected_proposals,
            pred_idx_objects,
            pred_scores,
            pred_score_distribution,
        )

    # ------------------------------------------------------------------ #
    # BOP evaluation (pytorch_lightning.Trainer.test, see run_inference.py)
    # ------------------------------------------------------------------ #
    @property
    def prediction_dir(self):
        return osp.join(
            self.log_dir, "predictions", self.dataset_name, self.name_prediction_file
        )

    def _templates_newer_than(self, cache_path):
        """True if a pyrender template of the reference set was written after the descriptor cache (re-rendered)."""
        if self.onboarding_config.rendering_type != "pyrender":
            return False  # other template sources are not files of template_dir
        cache_mtime = osp.getmtime(cache_path)
        for obj_id in self.ref_dataset.obj_ids:
            obj_dir = osp.join(self.ref_dataset.template_dir, f"obj_{obj_id:06d}")
            for idx in self.ref_dataset.index_templates:
                path = osp.join(obj_dir, f"{idx:06d}.png")
                if osp.exists(path) and osp.getmtime(path) > cache_mtime:
                    return True
        return False

    def set_reference_objects(self):
        """Compute (or load from cache) the descriptors of all templates of `ref_dataset`."""
        assert self.ref_dataset is not None, "set model.ref_dataset before testing"
        logging.info("Initializing reference objects ...")
        start_time = time.time()
        # cache keyed by descriptor model, rendering type and template level so that
        # switching one of them never reuses stale descriptors
        descriptors_path = osp.join(
            self.ref_dataset.template_dir,
            f"descriptors_{self.descriptor_model.model_name}"
            f"_{self.onboarding_config.rendering_type}"
            f"_level{self.onboarding_config.level_templates}.pth",
        )
        use_cache = (
            osp.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        )
        if use_cache and self._templates_newer_than(descriptors_path):
            logging.info(
                f"Templates newer than {descriptors_path}: recomputing the descriptors"
            )
            use_cache = False
        if use_cache:
            descriptors = torch.load(descriptors_path, map_location=self.device)
            logging.info(f"Loaded cached descriptors from {descriptors_path}")
        else:
            descriptors = BatchedData(None)
            for idx in tqdm(
                range(len(self.ref_dataset)), desc="Computing descriptors ..."
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                descriptors.append(ref_feats.float())
            descriptors.stack()  # N_objects x N_templates x descriptor_size
            descriptors = descriptors.data
            torch.save(descriptors, descriptors_path)
            logging.info(f"Saved descriptors to {descriptors_path}")
        self.set_reference_descriptors(descriptors)
        logging.info(
            f"Runtime: {time.time() - start_time:.02f}s, Descriptors shape: {descriptors.shape}"
        )

    def on_test_start(self):
        assert (
            self.log_dir is not None
            and self.dataset_name is not None
            and self.name_prediction_file is not None
        ), "log_dir, dataset_name and name_prediction_file must be set for BOP testing"
        os.makedirs(self.prediction_dir, exist_ok=True)
        self.move_to_device()
        self.set_reference_objects()

    @staticmethod
    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def test_step(self, batch, idx):
        assert batch["image"].shape[0] == 1, "Batch size must be 1"
        scene_id = int(batch["scene_id"][0])
        frame_id = int(batch["frame_id"][0])
        image_np = batch["image"][0].cpu().numpy()  # H x W x 3, uint8 RGB

        # stage 1: proposals + descriptors
        self._sync()
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )
        query_descriptors = self.descriptor_model(image_np, detections)
        self._sync()
        proposal_stage_end_time = time.time()

        # stage 2: matching
        matching_stage_start_time = time.time()
        if len(detections) == 0:
            logging.warning(f"No proposal for scene {scene_id} frame {frame_id}")
            num_objects = self.ref_data["descriptors"].shape[0]
            device = detections.boxes.device
            detections.add_attribute("scores", torch.zeros((0,), device=device))
            detections.add_attribute(
                "score_distribution", torch.zeros((0, num_objects), device=device)
            )
            detections.add_attribute(
                "object_ids", torch.zeros((0,), dtype=torch.long, device=device)
            )
        else:
            (
                idx_selected_proposals,
                pred_idx_objects,
                pred_scores,
                pred_score_distribution,
            ) = self.find_matched_proposals(query_descriptors)
            detections.filter(idx_selected_proposals)
            detections.add_attribute("scores", pred_scores.float())
            detections.add_attribute(
                "score_distribution", pred_score_distribution.float()
            )
            detections.add_attribute("object_ids", pred_idx_objects)
            if len(detections) > 0:
                detections.apply_nms_per_object_id(
                    nms_thresh=self.post_processing_config.nms_thresh
                )
        self._sync()
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        file_path = osp.join(self.prediction_dir, f"scene{scene_id}_frame{frame_id}")
        detections.save_to_file(
            scene_id=scene_id,
            frame_id=frame_id,
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=False,
            save_mask=self.save_mask,
            save_score_distribution=True,
            # BOP object ids of the templates (e.g. [1, 5, 6, ...] for LM-O)
            obj_ids=list(self.ref_dataset.obj_ids),
        )
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def on_test_epoch_end(self):
        if self.global_rank != 0:  # only rank 0 process
            return
        # simpler than all_gather: load the per-image files so nothing is missing
        result_paths = sorted(
            path
            for path in glob.glob(osp.join(self.prediction_dir, "*.npz"))
            if "runtime" not in path
        )
        num_workers = min(10, os.cpu_count() or 1)
        logging.info(
            f"Converting {len(result_paths)} npz files to json with {num_workers} workers ..."
        )
        convert_npz_to_json_with_idx = partial(
            convert_npz_to_json, list_npz_paths=result_paths
        )
        with multiprocessing.Pool(processes=num_workers) as pool:
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
        formatted_detections = []
        formatted_detections_with_score_distribution = []
        for detection in detections:
            formatted_detections.extend(detection[0])
            formatted_detections_with_score_distribution.extend(detection[1])

        detections_path = osp.join(self.log_dir, f"{self.name_prediction_file}.json")
        save_json_bop23(detections_path, formatted_detections)
        logging.info(
            f"Saved {len(formatted_detections)} predictions (BOP format) to {detections_path}"
        )

        detections_path = osp.join(
            self.log_dir, f"{self.name_prediction_file}_with_score_distribution.json"
        )
        save_json_bop23(detections_path, formatted_detections_with_score_distribution)
        logging.info(
            f"Saved predictions (BOP format + score distribution) to {detections_path}"
        )
