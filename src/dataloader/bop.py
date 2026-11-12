import logging
import os
import os.path as osp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataloader.base_bop import BaseBOP
from src.poses.utils import load_index_level_in_level2
from src.utils.bbox_utils import CropResizePad
from src.utils.inout import load_json, save_json


def load_rgb_uint8(path):
    """
    Read a BOP test image as uint8 H x W x 3.

    PIL's convert("RGB") clips 16-bit images (ITODD's gray/*.tif, mode I;16) to 255
    instead of rescaling them, so grayscale images are rescaled explicitly and then
    replicated to three channels.
    """
    image = Image.open(path)
    if image.mode in ("RGB", "RGBA", "P", "L"):
        return np.array(image.convert("RGB"))
    array = np.array(image).astype(np.float64)
    if array.ndim == 2:
        if array.max() > 255:
            array = array / 65535.0 * 255.0
        array = np.repeat(np.clip(array, 0, 255)[..., None].astype(np.uint8), 3, axis=-1)
        return array
    raise ValueError(f"Unsupported image {path}: mode={image.mode}, shape={array.shape}")


class BOPTemplate(Dataset):
    """
    Templates rendered with pyrender (src/scripts/render_template_with_pyrender.py):
    `<template_dir>/obj_XXXXXX/YYYYYY.png`, RGBA images on a transparent background,
    one file per viewpoint of the dense level-2 sphere (642 views).

    `level_templates` selects the subset of views used as references
    (0: 42 views, 1: 162 views, 2: all 642 views).
    """

    def __init__(
        self,
        template_dir,
        obj_ids,
        processing_config,
        level_templates,
        pose_distribution,
        num_imgs_per_obj=50,
        **kwargs,
    ):
        self.template_dir = template_dir
        if obj_ids is None:
            obj_ids = sorted(
                {
                    int(name[4:10])
                    for name in os.listdir(template_dir)
                    if name.startswith("obj_")
                    and osp.isdir(osp.join(template_dir, name))
                }
            )
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        assert len(obj_ids) > 0, f"No obj_XXXXXX folder found in {template_dir}"
        self.obj_ids = list(obj_ids)
        self.num_imgs_per_obj = num_imgs_per_obj  # only used by model-free onboarding
        self.processing_config = processing_config
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size)
        self.load_template_poses(level_templates, pose_distribution)

    def __len__(self):
        return len(self.obj_ids)

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all")
        else:
            raise NotImplementedError
        logging.info(
            f"Using {len(self.index_templates)} templates per object (level {level_templates})"
        )

    def __getitem__(self, idx):
        templates, masks, boxes = [], [], []
        obj_dir = osp.join(self.template_dir, f"obj_{self.obj_ids[idx]:06d}")
        for id_template in self.index_templates:
            image = Image.open(osp.join(obj_dir, f"{id_template:06d}.png"))
            bbox = image.getbbox()  # None if the rendering is fully transparent
            if bbox is None:
                logging.warning(f"Empty template {obj_dir}/{id_template:06d}.png")
                bbox = (0, 0, image.size[0], image.size[1])
            boxes.append(bbox)

            mask = image.getchannel("A")
            mask = torch.from_numpy(np.array(mask) / 255).float()
            masks.append(mask.unsqueeze(-1))

            # transparent background -> black background, as for the masked proposals
            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            templates.append(image)

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        templates_cropped = self.proposal_processor(images=templates, boxes=boxes)
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)
        return {
            "templates": self.rgb_transform(templates_cropped),
            "template_masks": masks_cropped[:, 0, :, :],
        }


class BaseBOPTest(BaseBOP):
    """
    Query images of a BOP test split, restricted to the images listed in the
    `test_targets_bop*.json` file (the ones used by the BOP evaluation).
    """

    def __init__(
        self,
        root_dir,
        split,
        dataset_name=None,
        reset_metaData=True,
        **kwargs,
    ):
        super().__init__(root_dir=root_dir, split=split, dataset_name=dataset_name)
        self.load_required_test_images_from_target_file()
        self.load_list_scene(split=split)
        self.load_metaData(reset_metaData=reset_metaData, split=split)

    def load_required_test_images_from_target_file(self) -> None:
        dataset_dir = Path(self.root_dir)
        target_files = sorted(dataset_dir.glob("test_targets_bop*.json"))
        # If multiple files are found, use the bop_version to select the correct one.
        if len(target_files) > 1:
            bop_version = "bop19"
            if self.dataset_name in ["hot3d", "hopev2", "handal"]:
                bop_version = "bop24"
            target_files = [f for f in target_files if bop_version in f.name]
        assert (
            len(target_files) == 1
        ), f"Expected one target file in {dataset_dir}, found {target_files}"
        logging.info(f"Loading target file: {target_files[0]}")
        targets = load_json(str(target_files[0]))
        # the target file has one row per (image, object): keep each image once
        images_per_scene = {}
        for item in targets:
            scene_id, im_id = int(item["scene_id"]), int(item["im_id"])
            images_per_scene.setdefault(scene_id, set()).add(im_id)
        self.target_images_per_scene = {
            scene_id: sorted(im_ids) for scene_id, im_ids in images_per_scene.items()
        }
        num_images = sum(len(v) for v in self.target_images_per_scene.values())
        logging.info(
            f"{num_images} target images in {len(self.target_images_per_scene)} scenes"
        )

    def load_metaData(self, reset_metaData=True, split="test"):
        start_time = time.time()
        metaData_path = osp.join(self.root_dir, f"{split}_metaData.json")
        if reset_metaData or not osp.exists(metaData_path):
            metaData = {"scene_id": [], "frame_id": [], "rgb_path": []}
            for scene_path in tqdm(self.list_scenes, desc="Loading metaData"):
                scene_id = int(osp.basename(scene_path))
                im_ids = self.target_images_per_scene.get(scene_id, [])
                if len(im_ids) == 0:
                    logging.warning(f"Scene {scene_id} has no target image, skipped")
                    continue
                image_dir, ext = self.find_image_dir(scene_path)
                for im_id in im_ids:
                    metaData["scene_id"].append(scene_id)
                    metaData["frame_id"].append(im_id)
                    metaData["rgb_path"].append(osp.join(image_dir, f"{im_id:06d}{ext}"))
            save_json(metaData_path, metaData)
        else:
            metaData = load_json(metaData_path)
        self.metaData = pd.DataFrame.from_dict(metaData)
        # shuffle to spread the scenes over the run (fixed seed)
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {time.time() - start_time:.2f} seconds"
        )

    def __getitem__(self, idx):
        row = self.metaData.iloc[idx]
        image = load_rgb_uint8(row["rgb_path"])
        # uint8 H x W x 3: the segmentor and the descriptor model do their own
        # normalization, so the image is passed through unchanged
        image = torch.from_numpy(image)
        return dict(
            image=image,
            scene_id=int(row["scene_id"]),
            frame_id=int(row["frame_id"]),
        )
