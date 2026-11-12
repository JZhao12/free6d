"""
Reference templates picked from the BlenderProc4BOP `train_pbr` split of a BOP dataset,
replicating the template selection of the CNOS baseline with BlenderProc templates (row 7
of Table 1 of its paper). Selected with `model.onboarding_config.rendering_type=pbr` in
run_inference.py.

Visual check (one grid of templates per object):
    python -m src.dataloader.bop_pbr ycbv [--root datasets/bop23_challenge] [--level 0] [--out logs/pbr_templates]
"""
import argparse
import logging
import os
import os.path as osp
import time

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.dataloader.base_bop import BaseBOP
from src.poses.utils import (
    NearestTemplateFinder,
    combine_R_and_T,
    get_obj_poses_from_template_level,
    opencv2opengl,
)
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import load_json, save_json


class BOPTemplatePBR(BaseBOP):
    """
    For every viewpoint of the template level, the template of an object is the instance
    of the first `max_num_scenes` scenes (x `max_num_frames` frames) of `<root_dir>/train_pbr`
    whose viewing direction is the closest, among the instances that are more than
    `min_visib_fract` visible. The template image is the frame with the background blacked
    out by `mask_visib` and cropped to the mask, exactly like a masked proposal.

    Difference with the original implementation (src/dataloader/bop_pbr.py upstream): the
    nearest instance is searched among all candidates; the original drew 5000 of them at
    random (with replacement, no seed) before the search, so its templates changed from
    run to run and were not always the closest ones.

    `template_dir` is the pyrender template folder of the dataset; it is only used to find
    the object ids and to store the descriptor cache next to the pyrender descriptors.
    """

    def __init__(
        self,
        root_dir,
        template_dir,
        obj_ids,
        processing_config,
        level_templates,
        pose_distribution,
        split="train_pbr",
        max_num_scenes=10,
        max_num_frames=1000,
        min_visib_fract=0.8,
        **kwargs,
    ):
        super().__init__(root_dir=root_dir, split=split)
        self.template_dir = template_dir
        self.obj_ids = list(obj_ids) if obj_ids is not None else self.find_obj_ids()
        assert len(self.obj_ids) > 0, f"No object found for {root_dir}"
        self.level_templates = level_templates
        self.pose_distribution = pose_distribution
        self.max_num_scenes = max_num_scenes
        self.max_num_frames = max_num_frames
        self.min_visib_fract = min_visib_fract
        self.processing_config = processing_config
        self.rgb_transform = T.Compose(
            [T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size)
        self.load_template_poses(level_templates, pose_distribution)

        split_dir = osp.join(root_dir, split)
        assert osp.isdir(split_dir), (
            f"{split_dir} not found: download the BlenderProc4BOP split with "
            f"src/scripts/download_train_pbr.sh"
        )
        self.load_list_scene(split)
        self.list_scenes = self.list_scenes[: self.max_num_scenes]
        logging.info(
            f"Picking {len(self.template_poses)} PBR templates per object for "
            f"{len(self.obj_ids)} objects from the first {len(self.list_scenes)} scenes of {split_dir}"
        )
        self.metaData = self.select_templates()

    def find_obj_ids(self):
        """Object ids from the pyrender template folders, else from models_info.json."""
        if osp.isdir(self.template_dir):
            obj_ids = sorted(
                int(name[4:10])
                for name in os.listdir(self.template_dir)
                if name.startswith("obj_") and osp.isdir(osp.join(self.template_dir, name))
            )
            if len(obj_ids) > 0:
                return obj_ids
        for cad_dir in ["models", "models_cad"]:
            models_info = osp.join(self.root_dir, cad_dir, "models_info.json")
            if osp.exists(models_info):
                return sorted(int(obj_id) for obj_id in load_json(models_info))
        raise FileNotFoundError(
            f"Neither obj_XXXXXX folders in {self.template_dir} nor models_info.json in {self.root_dir}"
        )

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution != "all":
            raise NotImplementedError
        self.template_poses = get_obj_poses_from_template_level(
            level_templates, pose_distribution
        )

    def load_candidates(self):
        """
        Every instance of the selected scenes that is more than `min_visib_fract` visible:
        a DataFrame (scene_id, frame_id, idx_obj, obj_id, visib_fract, rgb_path) and the
        matching N x 4 x 4 object poses (cam_R_m2c, cam_t_m2c).
        """
        obj_ids = set(self.obj_ids)
        rows, poses = [], []
        for scene_path in tqdm(self.list_scenes, desc="Reading train_pbr annotations"):
            scene_id = int(osp.basename(scene_path))
            image_dir, ext = self.find_image_dir(scene_path)
            scene_gt = load_json(osp.join(scene_path, "scene_gt.json"))
            scene_gt_info = load_json(osp.join(scene_path, "scene_gt_info.json"))
            frame_ids = sorted(int(frame_id) for frame_id in scene_gt)[: self.max_num_frames]
            for frame_id in frame_ids:
                rgb_path = osp.relpath(
                    osp.join(image_dir, f"{frame_id:06d}{ext}"), self.root_dir
                )
                frame_gt, frame_info = scene_gt[str(frame_id)], scene_gt_info[str(frame_id)]
                for idx_obj, (gt, info) in enumerate(zip(frame_gt, frame_info)):
                    if (
                        int(gt["obj_id"]) not in obj_ids
                        or float(info["visib_fract"]) <= self.min_visib_fract
                    ):
                        continue
                    rows.append(
                        (scene_id, frame_id, idx_obj, int(gt["obj_id"]), float(info["visib_fract"]), rgb_path)
                    )
                    poses.append(combine_R_and_T(gt["cam_R_m2c"], gt["cam_t_m2c"]))
        candidates = pd.DataFrame(
            rows, columns=["scene_id", "frame_id", "idx_obj", "obj_id", "visib_fract", "rgb_path"]
        )
        return candidates, np.stack(poses) if len(poses) else np.zeros((0, 4, 4))

    def select_templates(self):
        """
        One candidate per template viewpoint and object: the one whose viewing direction
        (camera z axis in object coordinates, as in NearestTemplateFinder) is the nearest.
        The selection is written to <root_dir>/<split>_templates_level<L>.json for inspection.
        """
        start_time = time.time()
        candidates, poses = self.load_candidates()
        finder = NearestTemplateFinder(
            level_templates=self.level_templates,
            pose_distribution=self.pose_distribution,
            return_inplane=False,
        )
        template_dirs = opencv2opengl(finder.obj_template_poses)[:, 2, :3]
        selected = []
        for obj_id in self.obj_ids:
            index_obj = np.flatnonzero(candidates["obj_id"].values == obj_id)
            assert len(index_obj) > 0, (
                f"obj {obj_id}: no instance with visib_fract > {self.min_visib_fract} in the "
                f"first {len(self.list_scenes)} scenes of {self.split}, increase max_num_scenes"
            )
            obj_poses = poses[index_obj]
            # only the rotation enters the search, so the translation normalisation of the
            # original implementation is not needed
            nearest = finder.search_nearest_query(obj_poses)  # one index per template pose
            candidate_dirs = opencv2opengl(obj_poses)[:, 2, :3]
            cosines = (candidate_dirs[nearest] * template_dirs).sum(axis=1)
            angles = np.degrees(np.arccos(np.clip(cosines, -1, 1)))
            for template_id, (idx, angle) in enumerate(zip(nearest, angles)):
                row = candidates.iloc[index_obj[idx]]
                selected.append(
                    {
                        "obj_id": int(obj_id),
                        "template_id": int(template_id),
                        "scene_id": int(row.scene_id),
                        "frame_id": int(row.frame_id),
                        "idx_obj": int(row.idx_obj),
                        "visib_fract": float(row.visib_fract),
                        "rgb_path": str(row.rgb_path),
                        "angle_deg": float(angle),
                    }
                )
            logging.info(
                f"obj {obj_id:3d}: {len(index_obj):6d} candidates, viewing-direction error "
                f"mean {angles.mean():.1f} deg, max {angles.max():.1f} deg, "
                f"{len(np.unique(nearest))} distinct frames"
            )
        metaData = pd.DataFrame(selected)
        selection_path = osp.join(
            self.root_dir, f"{self.split}_templates_level{self.level_templates}.json"
        )
        save_json(selection_path, selected)
        logging.info(
            f"Selected {len(metaData)} PBR templates from {len(candidates)} candidates in "
            f"{time.time() - start_time:.1f}s, written to {selection_path}"
        )
        return metaData

    def __getitem__(self, idx):
        obj_id = self.obj_ids[idx]
        rows = self.metaData[self.metaData["obj_id"] == obj_id].sort_values("template_id")
        assert len(rows) == len(self.template_poses), f"obj {obj_id}: {len(rows)} templates"
        templates, masks, boxes = [], [], []
        for row in rows.itertuples():
            mask_path = osp.join(
                self.root_dir,
                self.split,
                f"{row.scene_id:06d}",
                "mask_visib",
                f"{row.frame_id:06d}_{row.idx_obj:06d}.png",
            )
            mask = Image.open(mask_path).convert("L")
            bbox = mask.getbbox()
            if bbox is None:
                logging.warning(f"Empty mask {mask_path}")
                bbox = (0, 0, mask.size[0], mask.size[1])
            boxes.append(bbox)

            rgb = Image.open(osp.join(self.root_dir, row.rgb_path)).convert("RGB")
            # black background, as for the masked proposals and the pyrender templates
            masked_rgb = Image.composite(rgb, Image.new("RGB", rgb.size, (0, 0, 0)), mask)
            templates.append(torch.from_numpy(np.array(masked_rgb) / 255).float())
            masks.append(torch.from_numpy(np.array(mask) / 255).float().unsqueeze(-1))

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_cropped = self.proposal_processor(images=templates, boxes=boxes)
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_cropped),
            "template_masks": masks_cropped[:, 0, :, :],
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import OmegaConf
    from torchvision.utils import save_image

    parser = argparse.ArgumentParser(description="save a grid of the selected PBR templates per object")
    parser.add_argument("dataset_name")
    parser.add_argument("--root", default="./datasets/bop23_challenge")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--out", default="./logs/pbr_templates")
    args = parser.parse_args()

    dataset = BOPTemplatePBR(
        root_dir=osp.join(args.root, "datasets", args.dataset_name),
        template_dir=osp.join(args.root, "datasets", "templates_pyrender", args.dataset_name),
        obj_ids=None,
        processing_config=OmegaConf.create({"image_size": 224}),
        level_templates=args.level,
        pose_distribution="all",
    )
    inv_rgb_transform = T.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )
    out_dir = osp.join(args.out, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    for idx in tqdm(range(len(dataset)), desc="Saving template grids"):
        templates = inv_rgb_transform(dataset[idx]["templates"]).clamp(0, 1)
        save_image(templates, osp.join(out_dir, f"obj_{dataset.obj_ids[idx]:06d}.png"), nrow=7)
    angles = dataset.metaData["angle_deg"]
    print(
        f"{args.dataset_name}: {len(dataset)} objects x {len(dataset.template_poses)} templates, "
        f"viewing-direction error mean {angles.mean():.1f} deg, max {angles.max():.1f} deg; grids in {out_dir}"
    )
