"""
Render the templates of BOP datasets with pyrender.

For every CAD model of a dataset, the object is rendered from the 642 viewpoints of
the dense level-2 sphere (scaled to a camera distance of 0.4 of the predefined
poses). Output:

    <root_dir>/datasets/templates_pyrender/<dataset>/obj_XXXXXX/YYYYYY.png

Usage: python -m src.scripts.render_template_with_pyrender dataset_name=ycbv
"""
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
from functools import partial

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.poses.utils import get_obj_poses_from_template_level

logging.basicConfig(level=logging.INFO)

BOP23_DATASETS = ["lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv"]


def call_pyrender(
    idx_obj,
    list_cad_path,
    list_output_dir,
    obj_pose_path,
    disable_output,
    gpus_devices,
):
    output_dir = list_output_dir[idx_obj]
    cad_path = list_cad_path[idx_obj]
    if os.path.exists(output_dir):  # start from a clean folder
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    command = [
        sys.executable,
        "-m",
        "src.poses.pyrender",
        cad_path,
        obj_pose_path,
        output_dir,
        str(gpus_devices),
        "true" if disable_output else "false",
        # BOP models are in mm (the size heuristic of src/poses/pyrender.py fails for small parts)
        "--unit",
        "mm",
    ]
    log_path = osp.join(output_dir, "render.log")
    with open(log_path, "w") as log_file:
        if disable_output:
            returncode = subprocess.call(command, stdout=log_file, stderr=subprocess.STDOUT)
        else:
            print(f"executing command:\n{' '.join(command)}")
            returncode = subprocess.call(command)
    return idx_obj, returncode


def resolve_cad_dir(root_dir, dataset_name):
    if dataset_name == "tless":
        cad_file_ext = ".ply"
        candidates = ["models/models_cad", "models_cad"]
    elif dataset_name == "hot3d":
        cad_file_ext = ".glb"
        candidates = ["object_models"]
    else:
        cad_file_ext = ".ply"
        candidates = ["models/models", "models"]
    for candidate in candidates:
        cad_dir = osp.join(root_dir, dataset_name, candidate)
        if osp.isdir(cad_dir):
            return cad_dir, cad_file_ext
    raise FileNotFoundError(
        f"CAD folder not found for {dataset_name}, tried {candidates} in {osp.join(root_dir, dataset_name)}"
    )


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="render",
)
def render(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    root_save_dir = osp.join(cfg.data.root_dir, "templates_pyrender")
    template_poses = get_obj_poses_from_template_level(
        level=cfg.level, pose_distribution="all"
    )
    template_poses[:, :3, 3] *= 0.4  # zoom to object

    if cfg.dataset_name is None or cfg.dataset_name == "all":
        datasets = BOP23_DATASETS
    else:
        datasets = [cfg.dataset_name]

    for dataset_name in datasets:
        dataset_save_dir = osp.join(root_save_dir, dataset_name)
        logging.info(f"Rendering templates for {dataset_name} to {dataset_save_dir}")
        os.makedirs(dataset_save_dir, exist_ok=True)
        obj_pose_path = osp.join(dataset_save_dir, "template_poses.npy")
        np.save(obj_pose_path, template_poses)

        cad_dir, cad_file_ext = resolve_cad_dir(cfg.data.root_dir, dataset_name)
        object_ids = sorted(
            int(name[4:][: -len(cad_file_ext)])
            for name in os.listdir(cad_dir)
            if name.startswith("obj_") and name.endswith(cad_file_ext)
        )
        logging.info(f"{len(object_ids)} objects found in {cad_dir}: {object_ids}")
        cad_paths = [
            osp.join(cad_dir, f"obj_{object_id:06d}{cad_file_ext}")
            for object_id in object_ids
        ]
        output_dirs = [
            osp.join(dataset_save_dir, f"obj_{object_id:06d}")
            for object_id in object_ids
        ]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus)
        start_time = time.time()
        call_pyrender_with_index = partial(
            call_pyrender,
            list_cad_path=cad_paths,
            list_output_dir=output_dirs,
            obj_pose_path=obj_pose_path,
            disable_output=cfg.disable_output,
            gpus_devices=cfg.gpus,
        )
        with multiprocessing.Pool(processes=int(cfg.num_workers)) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(call_pyrender_with_index, range(len(object_ids))),
                    total=len(object_ids),
                )
            )

        # check that every object got all its templates
        failed = []
        for idx_obj, returncode in sorted(results):
            num_rendered = len(
                [f for f in os.listdir(output_dirs[idx_obj]) if f.endswith(".png")]
            )
            if returncode != 0 or num_rendered != len(template_poses):
                failed.append((object_ids[idx_obj], returncode, num_rendered))
        logging.info(
            f"Total time to render templates for {dataset_name}: {time.time() - start_time:.1f}s"
        )
        if failed:
            raise RuntimeError(
                f"Rendering failed for (obj_id, returncode, num_rendered): {failed}. "
                f"See obj_XXXXXX/render.log in {dataset_save_dir}"
            )
        logging.info(
            f"{len(object_ids)} objects x {len(template_poses)} templates rendered in {dataset_save_dir}"
        )


if __name__ == "__main__":
    render()
