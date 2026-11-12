"""
Run the Free6D localisation stage (SAM 2.1 + DINOv3) on a BOP test split and write the predictions in the
BOP challenge format (COCO results + uncompressed RLE masks).

usage: python run_inference.py dataset_name=ycbv
       python run_inference.py dataset_name=ycbv model.onboarding_config.level_templates=1
       python run_inference.py dataset_name=ycbv machine.trainer.limit_test_batches=5   # smoke test
       python run_inference.py dataset_name=ycbv model.onboarding_config.rendering_type=pbr  # BlenderProc4BOP templates
"""
import logging
import os.path as osp

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="run_inference")
def run_inference(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_path = hydra_cfg["runtime"]["output_dir"]
    logging.info(f"Inference script. The outputs of hydra will be stored in: {output_path}")

    dataset_name = cfg.dataset_name
    assert (
        dataset_name in cfg.data.datasets
    ), f"Unknown dataset {dataset_name}, available: {list(cfg.data.datasets.keys())}"

    logging.info("Initializing trainer")
    trainer = instantiate(cfg.machine.trainer)

    query_dataloader_config = cfg.data.query_dataloader.copy()
    ref_dataloader_config = cfg.data.reference_dataloader.copy()

    # query images
    query_dataloader_config.dataset_name = dataset_name
    if dataset_name in ["hb", "tless"]:
        query_dataloader_config.split = "test_primesense"
    else:
        query_dataloader_config.split = "test"
    query_dataloader_config.root_dir = osp.join(
        query_dataloader_config.root_dir, dataset_name
    )
    query_dataset = instantiate(query_dataloader_config)
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=1,  # only support a single image for now
        num_workers=cfg.machine.num_workers,
        shuffle=False,
    )

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    model.ref_obj_names = cfg.data.datasets[dataset_name].obj_names
    model.dataset_name = dataset_name

    # reference templates
    rendering_type = cfg.model.onboarding_config.rendering_type
    pbr_config = ref_dataloader_config.pop("pbr", {})
    ref_dataloader_config.template_dir = osp.join(
        ref_dataloader_config.template_dir, "templates_pyrender", dataset_name
    )
    if rendering_type == "pyrender":
        ref_dataset = instantiate(ref_dataloader_config)
    elif rendering_type == "pbr":
        # templates picked from the BlenderProc4BOP train_pbr frames of the dataset; the
        # descriptor cache is stored next to the pyrender templates (template_dir)
        ref_dataloader_config._target_ = "src.dataloader.bop_pbr.BOPTemplatePBR"
        ref_dataloader_config.root_dir = query_dataloader_config.root_dir
        ref_dataloader_config.update(pbr_config)
        ref_dataset = instantiate(ref_dataloader_config)
    else:
        raise NotImplementedError(f"Unknown rendering_type {rendering_type}")
    model.ref_dataset = ref_dataset

    segmentation_name = cfg.model.segmentor_model._target_.split(".")[-1]
    agg_function = cfg.model.matching_config.aggregation_function
    level_template = cfg.model.onboarding_config.level_templates
    model.name_prediction_file = f"{segmentation_name}_template_{rendering_type}{level_template}_agg{agg_function}_{dataset_name}"
    logging.info(f"Loading dataloader for {dataset_name} done!")

    trainer.test(model, dataloaders=query_dataloader)
    logging.info("---" * 20)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_inference()
