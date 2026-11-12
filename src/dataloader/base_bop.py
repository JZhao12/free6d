import logging
import os
import os.path as osp
from pathlib import Path

from torch.utils.data import Dataset


class BaseBOP(Dataset):
    def __init__(self, root_dir, split, dataset_name=None, **kwargs):
        """
        Read a dataset in the BOP format.
        See https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
        """
        self.root_dir = root_dir
        self.split = split
        self.dataset_name = dataset_name

    def load_list_scene(self, split=None):
        if isinstance(split, str):
            split_folder = osp.join(self.root_dir, split)
            self.list_scenes = sorted(
                [
                    osp.join(split_folder, scene)
                    for scene in os.listdir(split_folder)
                    if os.path.isdir(osp.join(split_folder, scene))
                    and scene != "models"
                ]
            )
        elif isinstance(split, list):
            self.list_scenes = []
            for scene in split:
                if not isinstance(scene, str):
                    scene = f"{scene:06d}"
                if os.path.isdir(osp.join(self.root_dir, scene)):
                    self.list_scenes.append(osp.join(self.root_dir, scene))
            self.list_scenes = sorted(self.list_scenes)
        else:
            raise NotImplementedError
        logging.info(f"Found {len(self.list_scenes)} scenes")

    @staticmethod
    def find_image_dir(scene_path):
        """
        Return (image_dir, extension) of the color images of a BOP scene.

        BOP stores them in `rgb/*.png` for most datasets, `rgb/*.jpg` for some PBR
        splits and `gray/*.tif` for ITODD. Resolving it from the folder content avoids
        depending on bop_toolkit_lib at inference time.
        """
        for sub_dir in ["rgb", "gray"]:
            image_dir = Path(scene_path) / sub_dir
            if image_dir.is_dir():
                break
        else:
            raise FileNotFoundError(f"No rgb/ or gray/ folder found in {scene_path}")
        extensions = sorted({p.suffix for p in image_dir.iterdir() if p.is_file()})
        if len(extensions) == 0:
            raise FileNotFoundError(f"No image found in {image_dir}")
        if len(extensions) > 1:
            logging.warning(f"Multiple image extensions {extensions} in {image_dir}")
        return str(image_dir), extensions[0]

    def __len__(self):
        return len(self.metaData)
