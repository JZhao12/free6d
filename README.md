# Free6D: training-free object localisation and 6D pose estimation with vision foundation models

Free6D localises unseen objects in a single RGB image and estimates their 6D pose from a CAD model
alone. It runs on two frozen vision foundation models, **SAM 2.1** (Hiera-L) [2] and **DINOv3**
(ViT-L/16) [3]: nothing is trained or fine-tuned, no depth image is used, and a new object is onboarded
by rendering 42 views of its CAD model.

| Stage | Output | How |
|---|---|---|
| Onboarding (once per object) | 42 templates | pyrender views of the CAD model; DINOv3 class token and patch features of every view |
| Localisation | instance masks, boxes, object identities | SAM 2.1 mask proposals, matched to the templates with the DINOv3 class token |
| 6D pose | rotation and translation of every instance | DINOv3 patch features matched to the same templates, 2D-3D correspondences, PnP-RANSAC |
| Refinement (optional) | refined poses, still training-free | render-and-compare with DINOv3 features |

One DINOv3 backbone tells which object a mask shows (class token) and how it is posed (patch tokens),
and both stages share the same 42 templates.

## Results at a glance

Evaluated on the seven BOP'23 core datasets [6] (LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V), BOP'19
test targets, single RTX 5070. ITODD and HB have no public test ground truth and are scored by the
BOP website, the other five locally with `bop_toolkit`.

- **Localisation.** Mean 2D segmentation AP **45.6** over the seven datasets with pyrender templates,
  against 36.1 (pyrender templates) and 40.4 (BlenderProc templates) for CNOS [1]. With BlenderProc
  templates Free6D reaches 44.9; see [Localisation results](#localisation-results).
- **6D pose.** Mean average recall (BOP AR) over the seven datasets: **44.4** with PnP alone and
  **49.6** with the training-free refinement, against 37.2 for FoundPose [4] and 39.5 for Pos3R [5]
  (both training-free, without refinement); see [Pose results](#pose-results).
- **Speed.** About 1.2 s per image for the localisation, plus about 0.1 s per object for the pose
  (0.3 s with refinement).

## Method

Input: an RGB image, the camera intrinsics and the CAD models of the objects to find. Output: for
every instance a mask, a box, the object identity, a confidence and the 6D pose.

```
CAD model ──pyrender──▶ 42 template views ──DINOv3──▶ class tokens + patch features with 3D coordinates
                                                          │
RGB image ──SAM 2.1 auto-masks──▶ proposals ──DINOv3 class token──▶ matching ──▶ masks + identities
                                                          │
masked crop (4 rotations) ──DINOv3 patch tokens──▶ mutual nearest neighbours ──▶ 2D-3D matches
                                                          │
                 EPnP-RANSAC ──▶ pose ──(optional)──▶ render, compare DINOv3 features, re-match
```

### Stage 0: onboarding (once per object)

1. **Templates.** The CAD model is rendered with pyrender from 42 viewpoints (level-0 icosphere, about
   32 degrees between neighbours, camera at 0.4 m, black background, one in-plane orientation per
   viewpoint).
2. **Descriptors for localisation.** Each template is cropped to its foreground, padded to a square and
   resized to 224x224; the DINOv3 class token (1024-d) is its descriptor.
3. **Template bank for pose**, from the same templates. Each one is cropped to a square (1.1 x the
   longer side of its foreground box) and resized to 448x448 with the background zeroed; the patch
   tokens of block 19 (0-based, of 24) of DINOv3 ViT-L/16 give 28x28 = 784 L2-normalised features. A
   depth map rendered at the template pose is back-projected to model coordinates and averaged over the
   foreground pixels of each patch, so every patch carries a 3D model coordinate. Patches with less
   than half of their pixels on the object are not used.
4. **Sharing rule.** The pose stage reuses the localisation templates. It renders its own (the same 42
   viewpoints, camera fitted to the object) only for objects whose templates are cut by the image
   border (objects too large for the fixed 0.4 m distance) or do not agree with a re-rendering of the
   CAD model (mean mask IoU below 0.9). On the BOP'23 core datasets 103 of the 108 objects share
   their templates.

### Stage 1: localisation

1. **Proposals.** The image is resized to 640 px width and SAM 2.1 generates class-agnostic masks
   (32x32 point grid, `pred_iou_thresh` 0.8, `stability_score_thresh` 0.85, box NMS 0.7); proposals
   smaller than 5% of the image side or 0.03% of the image area are dropped.
2. **Descriptors.** Each proposal is masked, cropped, padded to a square and resized to 224x224; its
   DINOv3 class token is compared with those of the templates.
3. **Matching.** Cosine similarity with the 42 templates of every object, aggregated as the mean of the
   5 best templates (the scoring rule of CNOS [1]). A proposal is assigned to the object with the
   highest score and rejected below 0.15; per-object NMS at IoU 0.25, at most 100 instances per image.

### Stage 2: 6D pose (per detected instance)

1. **Query features.** The mask is cropped to a square (1.1 x the longer box side), resized to 448x448
   with the background zeroed, and fed to DINOv3 in four orientations (0, 90, 180 and 270 degrees):
   DINOv3 patch features are not invariant to in-plane rotation and the templates hold a single
   in-plane orientation per viewpoint. Patches with at least half of their pixels on the mask are kept.
2. **Matching.** For each of the 4 orientations and each of the 42 templates, mutual nearest neighbours
   (cosine similarity) between query and template patches. An (orientation, template) pair is scored
   by the sum of the similarities of its mutual matches; the 5 best pairs are kept.
3. **PnP.** Each kept pair gives 2D-3D correspondences (query patch centre mapped back to the
   un-rotated crop, model coordinate of the matched template patch): EPnP + RANSAC (400 iterations,
   20 px reprojection threshold in the 448 px crop), then Levenberg-Marquardt on the inliers. The
   hypothesis with the most inliers is kept. With fewer than 8 inliers the pose falls back to the
   rotation of the best template, placed from the size and position of the crop.
4. **Optional refinement, still training-free** (`--n_hyp 3 --refine_iters 2`). The 3 hypotheses with
   the most inliers are rendered in the camera of the query crop and scored by the agreement of the
   DINOv3 features of rendering and query at identical patch positions. Each hypothesis is refined up
   to twice: query and rendering are re-matched (mutual nearest neighbours within 0.35 of the crop
   size), PnP is solved with the 3D coordinates of the rendering, and an update is kept only if the
   score increases. The best-scoring pose is returned.

The confidence of a pose is the localisation score of its mask; the reported time per image is the
localisation time plus the pose time.

Three design choices matter most (see [Ablations](#ablations-ycb-v-subset)): the query is fed in four
orientations instead of adding in-plane rotated templates, the patch features come from an
intermediate block (19) instead of the last one, and both stages share the 42 templates.

## BOP method description

Text used for the method entry on the [BOP website](https://bop.felk.cvut.cz/) (its description template), 2D detection and 2D segmentation tasks. For the 6D localization task the pose stage of [Method](#method) is added; that text is written to `datasets/bop23_challenge/results/bop_submission_pose/README.txt`.

**Submitted to:** BOP Challenge 2025, model-based 2D detection and 2D segmentation of unseen objects,
BOP-Classic-Core (LM-O, T-LESS, TUD-L, IC-BIN, ITODD, HB, YCB-V), BOP'19 test targets.

**Training data:** none. Free6D is zero-shot: SAM 2.1 Hiera-L and DINOv3 ViT-L/16 are used with their
public pretrained weights (`sam2.1_hiera_large.pt`, `dinov3_vitl16_pretrain_lvd1689m`), no fine-tuning
on BOP data.

**Onboarding data:** 42 templates per object rendered with pyrender from the CAD model (viewpoints of
the level-0 icosphere, camera distance 0.4 of the predefined poses, black background, uniform grey
material for T-LESS). No real onboarding images.

**Used 3D models:** the default BOP CAD models (`models/`, `models_cad/` for T-LESS).

**Notes:** three steps.
(1) Proposals: SAM 2.1 automatic mask generator on the RGB image resized to 640 px width,
`points_per_side` 32, `pred_iou_thresh` 0.8, `stability_score_thresh` 0.85 (the SAM v1 value 0.97 is too
strict for SAM2 and discards most proposals), `box_nms_thresh` 0.7; proposals smaller than 5% of the
image side or 0.03% of the image area are dropped. (2) Descriptors: each proposal is masked, cropped,
padded to a square and resized to 224x224; the DINOv3 CLS token (1024-d) is the descriptor, the same
for the templates. (3) Matching: cosine similarity between a proposal and the 42 templates of every
object, aggregated as the mean of the top 5; the object with the highest score is assigned, proposals
below 0.15 are rejected, per-object NMS at IoU 0.25, at most 100 instances per image. The same
prediction file serves both tasks (modal mask as RLE for segmentation, amodal box for detection);
`time` is the wall time of stages 1 to 3 per image. Inference runs in `bf16-mixed` with
pytorch-lightning. Code: https://github.com/JZhao12/free6d.

### Computer specifications

| Component | Value |
|---|---|
| GPU | 1x NVIDIA GeForce RTX 5070, 12 GB |
| CPU | Intel Core Ultra 9 285K, 24 cores |
| RAM | 62 GB |
| Storage | Samsung 990 EVO Plus NVMe SSD, 2 TB |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-42-generic |
| Driver / CUDA | NVIDIA driver 595.99.02, CUDA 12.8, cuDNN 9.10 |
| Software | Python 3.10.22, PyTorch 2.8.0+cu128, pytorch-lightning 2.5.4, SAM 2.1, DINOv3 (torch.hub, local), bop_toolkit |
| Runtime | about 1.2 s per image end to end (SAM2 proposals ~1.1 s, DINOv3 descriptors + matching ~0.1 s); template rendering (642 views per object) under a minute per dataset; GPU memory 5 to 10 GB depending on image size (640x480 to 1280x960) |

## Layout

```
configs/
  inference_custom.yaml            # custom inference entry point: composes `user` and `model`
  run_inference.yaml               # BOP inference entry point (+ machine, data)
  render.yaml                      # BOP template rendering
  user/default.yaml                # paths to sam2_repo / dinov3_repo / BOP root dir
  machine/local.yaml, machine/trainer/local.yaml  # pytorch-lightning 2.x Trainer (bf16-mixed)
  data/bop.yaml                    # BOP datasets table, query / template dataloaders
  model/free6d.yaml                # stage composition, post-processing, onboarding (templates)
  model/segmentor_model/sam2.yaml  # SAM2 config, checkpoint, proposal thresholds
  model/descriptor_model/dinov3.yaml
run_inference.py                   # BOP: SAM2 + DINOv3 on a test split -> BOP-format json
run_pose.py                        # BOP: 6D pose of the detections (DINOv3 patch matching + PnP) -> BOP'19 csv
src/
  dataloader/bop.py                # BOP test images (test_targets) and pyrender templates
  model/sam2.py                    # SAM2 automatic mask generator (wrapper)
  model/dinov3.py                  # DINOv3 descriptor wrapper
  model/detector.py                # two-stage container, template matching, BOP test_step
  model/utils.py, model/loss.py    # Detections structure, similarity
  poses/                           # template viewpoints and pyrender rendering
  pose/features.py                 # 6D pose: dense DINOv3 patch features (intermediate block)
  pose/templates.py                # 6D pose: template bank, patch features + 3D coordinates of the 42 views
  pose/render.py                   # 6D pose: pyrender renderer for arbitrary poses / intrinsics
  pose/matching.py, pose/pnp.py    # 6D pose: mutual nearest neighbours, EPnP-RANSAC, fallback pose
  pose/estimator.py                # 6D pose: per-image estimation, optional render-and-match refinement
  pose/geometry.py, pose/bop_io.py # 6D pose: crop / patch geometry, BOP targets / csv / quick recalls
  scripts/render_custom.sh         # custom step 1: render templates
  scripts/run_inference_custom.sh  # custom step 2: run inference
  scripts/inference_custom.py
  scripts/render_template_with_pyrender.py  # BOP: render templates of a whole dataset
  scripts/download_bop_classic.sh  # BOP: download + unzip the BOP'23 core datasets
  scripts/run_bop_dataset.sh       # BOP: templates -> inference -> scores for one dataset (smoke|full)
  scripts/eval_bop_coco.sh         # BOP: COCO AP/AR with bop_toolkit, locally
  scripts/summarize_bop_results.py # BOP: localisation results table
  scripts/run_bop_pose.sh          # BOP: 6D pose -> csv -> scores for one dataset
  scripts/eval_bop_pose.sh         # BOP: VSD / MSSD / MSPD average recalls with bop_toolkit, locally
  scripts/summarize_pose_results.py # BOP: 6D pose results table
  scripts/ablate_sam2_ycbv.py      # SAM2 proposal-parameter ablation on a YCB-V subset
  utils/                           # bbox / IO / trimesh helpers
media/teabox/                      # example CAD model and RGB image
datasets/bop23_challenge/          # BOP datasets, templates and results (not tracked)
sam2_repo/                         # SAM2 upstream repo (cloned during setup, not tracked)
dinov3_repo/                       # DINOv3 upstream repo + weights/ (cloned during setup, not tracked)
bop_toolkit_repo/                  # bop_toolkit (cloned during setup, not tracked)
```

## Installation

The environment is managed with mamba, named `free6d`, Python 3.10, PyTorch 2.8 with CUDA 12.8.

```bash
git clone https://github.com/JZhao12/free6d.git
cd free6d
mamba create -n free6d python=3.10
mamba activate free6d
pip install -r requirements.txt
```

Alternatively use `mamba env create -f environment.yml` from the repository root; it reads
`requirements.txt` next to it.

### SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git sam2_repo
pip install -e sam2_repo
wget -P sam2_repo/checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

> Clone into `sam2_repo`, never into a directory named `sam2/` at the repository root: that would
> shadow the pip-installed `sam2` package.

### DINOv3

```bash
git clone https://github.com/facebookresearch/dinov3.git dinov3_repo
pip install -r dinov3_repo/requirements.txt
mkdir -p dinov3_repo/weights
```

DINOv3 weights must be requested from Meta as described in `dinov3_repo/README.md`; place
`dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` in `dinov3_repo/weights/`.

Both repository paths are set in `configs/user/default.yaml`, relative to the repository root:

```yaml
dinov3_repo: ./dinov3_repo
sam2_repo: ./sam2_repo
local_root_dir: ./datasets/bop23_challenge   # BOP datasets / templates / results
```

### bop_toolkit (BOP evaluation only)

`bop_toolkit` pins `numpy<2`, which would downgrade numpy and break `opencv-python` 5 in this
environment, so it is installed without its dependencies (only `pypng` is actually missing). The
BOP evaluation uses a `pycocotools` fork that keeps the `ignore` flag of the ground truth:

```bash
git clone --depth 1 https://github.com/thodan/bop_toolkit.git bop_toolkit_repo
pip install --no-deps -e ./bop_toolkit_repo && pip install pypng
pip install --no-deps "git+https://github.com/MartinSmeyer/cocoapi.git#subdirectory=PythonAPI"
```

The 6D pose evaluation (`eval_bop19_pose.py`, VSD error) additionally needs the `vispy` renderer of
`bop_toolkit`: `pip install vispy` (adds `vispy` and `hsluv`, no other package changes). It renders
offscreen through EGL; `src/scripts/run_bop_pose.sh` unsets `DISPLAY` for it.

## Usage

Run every command from the repository root with the `free6d` environment active.

```bash
export CAD_PATH=./media/teabox/mesh.obj
export RGB_PATH=./media/teabox/000000.png
export OUTPUT_DIR=./tmp/teabox
# Optional: where the mask is written, defaults to $OUTPUT_DIR/free6d_results/mask.png
# export MASK_PATH=./tmp/teabox/free6d_results/mask.png

bash ./src/scripts/render_custom.sh         # 1. render templates into $OUTPUT_DIR/*.png
bash ./src/scripts/run_inference_custom.sh  # 2. run inference, results in $OUTPUT_DIR/free6d_results/
```

Output files:

| File | Content |
|---|---|
| `free6d_results/templates.png` | Grid of templates fed to DINOv3 |
| `free6d_results/detection.npz` | Detections (boxes, scores, segmentation, ...) |
| `free6d_results/detection.json` | BOP-style flat list, `segmentation` as uncompressed RLE |
| `free6d_results/vis.png` | Input image next to the detection overlay |
| `free6d_results/mask.png` | Merged binary mask of the selected detections (1-bit) |

Calling the Python entry point exposes more options:

```bash
python -m src.scripts.inference_custom \
  --template_dir ./tmp/teabox --rgb_path ./media/teabox/000000.png \
  --num_max_dets 3 --confg_threshold 0.3 --stability_score_thresh 0.8
```

| Option | Default | Meaning |
|---|---|---|
| `--num_max_dets` | 1 | Maximum number of detections to keep |
| `--confg_threshold` | 0.5 | Similarity threshold (historical spelling of the option) |
| `--stability_score_thresh` | 0.85 | SAM2 proposal stability; the shell script passes 0.8, lower it to 0.5 for small objects. The SAM v1 value 0.97 drops most SAM2 proposals |

Rendering parameters live in `src/scripts/render_custom.sh`: `LIGHTING_ITENSITY` (light intensity)
and `RADIUS` (camera-to-object distance).

## Testing on BOP datasets

Everything lives under `configs/user/default.yaml:local_root_dir`
(`./datasets/bop23_challenge`).

### Quick path: one dataset end to end

```bash
bash src/scripts/download_bop_classic.sh lmo          # no argument = all seven BOP'23 core datasets
bash src/scripts/run_bop_dataset.sh lmo smoke         # 5 images: checks templates -> inference -> scoring
bash src/scripts/run_bop_dataset.sh lmo full          # all BOP'19 test targets
bash src/scripts/download_train_pbr.sh lmo            # BlenderProc4BOP frames (22 GB), then:
RENDERING=pbr bash src/scripts/run_bop_dataset.sh lmo full
python -m src.scripts.summarize_bop_results           # localisation results table
```

`run_bop_dataset.sh` renders the templates if they are missing, runs the inference, validates the
prediction file with `bop_toolkit` and scores it locally. Concurrent invocations serialise their GPU
stages through a file lock (`GPU_LOCK`, default `/tmp/free6d_gpu.lock`). The individual steps are
described below.

### 1. Dataset

`src/scripts/download_bop_classic.sh` fetches `<ds>_base.zip`, `<ds>_models.zip` and the BOP'19 test
archive from the Hugging Face `bop-benchmark` mirror (resumable, archives kept in `datasets/tmp/`)
and unzips them so that the folder looks like:

```
datasets/bop23_challenge/datasets/ycbv/
  test/000048/{rgb,depth,mask,mask_visib,scene_gt.json,scene_gt_info.json,scene_camera.json}   # ycbv_test_bop19.zip
  models/obj_000001.ply ...                                                                        # ycbv_models.zip
  test_targets_bop19.json, camera_*.json                                                           # ycbv_base.zip
```

Only the images listed in `test_targets_bop19.json` are processed (900 for YCB-V). Dataset
specifics are handled automatically: `test_primesense` split for T-LESS and HB, `models_cad` for
T-LESS, 8-bit grayscale `gray/*.tif` for ITODD, and the non-contiguous object ids of LM-O
(`[1, 5, 6, 8, 9, 10, 11, 12]`), which are taken from the template folder names.

### 2. Templates

Two kinds of templates, selected with `model.onboarding_config.rendering_type`:

**pyrender** (default):

```bash
python -m src.scripts.render_template_with_pyrender dataset_name=ycbv   # `all` = seven BOP'23 core datasets
```

Renders the 642 views of the dense sphere for every CAD model into
`datasets/bop23_challenge/datasets/templates_pyrender/ycbv/obj_XXXXXX/` (about 50 s and 470 MB for
YCB-V). The number of views actually used as references is `model.onboarding_config.level_templates`
(0: 42 views, 1: 162, 2: 642); their DINOv3 descriptors are cached next to the templates.

**pbr** (BlenderProc): no rendering; the templates are real frames of the photo-realistic
[BlenderProc4BOP](https://bop.felk.cvut.cz/datasets/) training split of the dataset.

```bash
bash src/scripts/download_train_pbr.sh ycbv   # no argument = all seven; 21-37 GB per dataset
```

Downloads `<dataset>_train_pbr.zip` from the Hugging Face mirror (8 parallel byte-range streams,
resumable) and extracts it to `datasets/bop23_challenge/datasets/<dataset>/train_pbr/` (LM-O uses the
LM archive). At inference time `src/dataloader/bop_pbr.py` picks, for each of the 42 template
viewpoints and each object, the instance of the first 10 scenes (10 x 1000 frames) whose viewing
direction is the closest among the instances that are more than 80% visible; the template is the
frame with the background blacked out by `mask_visib` and cropped to the mask, exactly like a
proposal. The selection is written to `<dataset>/train_pbr_templates_level0.json` and can be
inspected with `python -m src.dataloader.bop_pbr ycbv` (one image grid per object in
`logs/pbr_templates/`). This is the BlenderProc template selection of CNOS [1], except that the
nearest instance is searched among all candidates instead of a random subset of 5000 of them, which
makes the selection reproducible.

### 3. Inference

```bash
python run_inference.py dataset_name=ycbv
# denser templates
python run_inference.py dataset_name=ycbv model.onboarding_config.level_templates=1
# smoke test on 5 images
python run_inference.py dataset_name=ycbv machine.trainer.limit_test_batches=5
# BlenderProc templates (after download_train_pbr.sh ycbv)
python run_inference.py dataset_name=ycbv model.onboarding_config.rendering_type=pbr
```

Writes one `.npz` per image and the merged BOP-format predictions (COCO results with uncompressed
RLE masks) to
`datasets/bop23_challenge/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json`
(`_template_pbr0_` with BlenderProc templates). That file can be uploaded to the
[BOP website](https://bop.felk.cvut.cz/) or scored locally:

### 4. Evaluation

```bash
DATASET_NAME=ycbv bash src/scripts/eval_bop_coco.sh \
  datasets/bop23_challenge/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json
# BlenderProc templates: the method name has to differ (bop_toolkit parses METHOD_DATASET-SPLIT.json)
DATASET_NAME=ycbv bash src/scripts/eval_bop_coco.sh \
  datasets/bop23_challenge/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pbr0_aggavg_5_ycbv.json free6dpbr
```

The script computes `scene_gt_coco.json` once (`bop_toolkit/scripts/calc_gt_coco.py`), then runs
`eval_bop22_coco.py` for `segm` (BOP 2D segmentation task) and amodal `bbox` (2D detection task).
Scores land in `datasets/bop23_challenge/results/bop_eval/free6d_ycbv-test/scores_bop22_coco_*.json`.

The BOP'19 test archives of ITODD and HB ship no `scene_gt.json`, so these two cannot be scored
locally: `run_bop_dataset.sh` stops after writing the prediction file, which has to be uploaded to
the [BOP website](https://bop.felk.cvut.cz/) for scoring.

### Localisation results

All numbers: BOP'19 test targets, SAM 2.1 Hiera-L proposals (`stability_score_thresh` 0.85), DINOv3
ViT-L/16 descriptors, 42 templates per object (level 0) either rendered with pyrender or picked from
the BlenderProc4BOP frames (see [2. Templates](#2-templates)), `bf16-mixed`, single RTX 5070
(see [BOP method description](#bop-method-description)). LM-O, T-LESS, TUD-L, IC-BIN and YCB-V are scored
locally with `bop_toolkit`; ITODD and HB by the BOP website. Tables are generated with
`python -m src.scripts.summarize_bop_results`.

#### 2D segmentation AP (BOP'19 test targets, 42 template views)

| Method | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean |
|---|---|---|---|---|---|---|---|---|
| CNOS (SAM + DINOv2 + Pyrender) | 33.3 | 38.3 | 35.8 | 27.2 | 14.8 | 45.9 | 57.6 | 36.1 |
| CNOS (SAM + DINOv2 + BlenderProc) | 39.6 | 39.7 | 39.1 | 28.4 | 28.2 | 48.0 | 59.5 | 40.4 |
| CNOS (FastSAM + DINOv2 + BlenderProc) | 39.7 | 37.4 | 48.0 | 27.0 | 25.4 | 51.1 | 59.9 | 41.2 |
| Free6D (SAM2 + DINOv3 + Pyrender) | 41.7 | 46.3 | 46.6 | 38.6 | 24.2 | 56.3 | 65.6 | 45.6 |
| Free6D (SAM2 + DINOv3 + BlenderProc) | 42.9 | 43.3 | 37.9 | 36.3 | 32.6 | 55.8 | 65.5 | 44.9 |

The first 3 rows are Table 1 of the CNOS paper [1].

#### 2D detection AP (amodal bbox), ours only

| Method | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean |
|---|---|---|---|---|---|---|---|---|
| Free6D (SAM2 + DINOv3 + Pyrender) | 41.6 | 36.5 | 40.0 | 26.0 | - | - | 57.3 | 40.3 |
| Free6D (SAM2 + DINOv3 + BlenderProc) | 42.3 | 31.5 | 33.8 | 24.3 | - | - | 51.0 | 36.6 |

No bbox score for ITODD, HB (only the segmentation task was scored for them), so the Mean column is over the other 5 datasets and is not comparable to the 7-dataset segmentation mean.

#### Per-dataset details: pyrender templates

| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image | scored |
|---|---|---|---|---|---|---|---|---|---|
| LM-O | 200 | 10839 | 41.7 | 69.8 | 46.4 | 50.7 | 41.6 | 1.27 s | local |
| T-LESS | 1000 | 49646 | 46.3 | 57.6 | 53.1 | 55.7 | 36.5 | 1.28 s | local |
| TUD-L | 600 | 17544 | 46.6 | 75.9 | 52.6 | 52.3 | 40.0 | 1.17 s | local |
| IC-BIN | 150 | 4358 | 38.6 | 66.1 | 41.7 | 44.9 | 26.0 | 1.25 s | local |
| ITODD | 721 | 5861 | 24.2 | 36.5 | 27.9 | 38.5 | - | 1.17 s | BOP website |
| HB | 300 | 11206 | 56.3 | 70.6 | 65.6 | 59.8 | - | 1.24 s | BOP website |
| YCB-V | 900 | 23503 | 65.6 | 87.0 | 76.4 | 72.8 | 57.3 | 1.24 s | local |

ITODD, HB: the BOP'19 test archives ship no scene_gt.json, so these scores come from the BOP website (method name `free6d`, same eval_bop22_coco.py); prediction files in `datasets/bop23_challenge/results/bop_submission/`, raw score files in `docs/bop_website_scores/`. Their bbox AP is listed only if the 2D detection task was also scored there.

The files uploaded to the BOP website (one per dataset, `free6d_<dataset>-test*.json[.gz]`) are written by `run_bop_dataset.sh` and collected in `datasets/bop23_challenge/results/bop_submission/`.

#### Per-dataset details: BlenderProc templates

| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image | scored |
|---|---|---|---|---|---|---|---|---|---|
| LM-O | 200 | 11526 | 42.9 | 72.7 | 48.2 | 51.3 | 42.3 | 1.26 s | local |
| T-LESS | 1000 | 50377 | 43.3 | 53.3 | 49.7 | 54.0 | 31.5 | 1.28 s | local |
| TUD-L | 600 | 16853 | 37.9 | 63.4 | 42.9 | 52.4 | 33.8 | 1.15 s | local |
| IC-BIN | 150 | 4633 | 36.3 | 62.8 | 39.4 | 44.3 | 24.3 | 1.21 s | local |
| ITODD | 721 | 5827 | 32.6 | 48.3 | 37.4 | 46.0 | - | 1.12 s | BOP website |
| HB | 300 | 11469 | 55.8 | 70.7 | 65.2 | 61.7 | - | 1.21 s | BOP website |
| YCB-V | 900 | 23704 | 65.5 | 87.0 | 76.6 | 71.2 | 51.0 | 1.18 s | local |

ITODD, HB: the BOP'19 test archives ship no scene_gt.json, so these scores come from the BOP website (method name `free6dpbr`, same eval_bop22_coco.py); prediction files in `datasets/bop23_challenge/results/bop_submission_pbr/`, raw score files in `docs/bop_website_scores/`. Their bbox AP is listed only if the 2D detection task was also scored there.

**Pyrender vs BlenderProc templates.** CNOS [1] gains 4.3 AP (36.1 to 40.4) from the realistic
templates with SAM + DINOv2. With SAM2 + DINOv3 the two template kinds are on par on average
(44.9 against 45.6): the realistic ones win clearly on ITODD (+8.4, the dataset where CNOS also
gains the most: its grey industrial parts under industrial lighting are far from the flat grey
pyrender renderings), slightly on LM-O (+1.2), are even on YCB-V (-0.1) and HB (-0.5), and lose on
IC-BIN (-2.3) and on the two datasets with untextured objects: T-LESS (-3.0) and TUD-L (-8.7). The
selected templates were inspected
(`python -m src.dataloader.bop_pbr <dataset>`, viewing-direction error 0.4 to 3.2 degrees on
average, 17.6 on LM-O whose PBR scenes only show the objects from above). What differs from the
pyrender renderings: the BlenderProc4BOP frames light the untextured objects with random coloured
illumination (T-LESS parts come out green, pink or grey; the white TUD-L ape pink or blue), the
objects are small (median mask side 88 px on TUD-L, 95 px on LM-O, against 406 px for the pyrender
renderings) and up to 20% occluded (`visib_fract` > 0.8). On TUD-L the realistic templates of the
white ape then match almost every white blob: 80% of all detections are labelled as that object
(a third with pyrender templates). The 2D detection AP drops more than the segmentation AP with
BlenderProc templates (YCB-V 57.3 -> 51.0 for the same proposals): the boxes of identical masks are
identical in both runs, the realistic templates simply rank and label the SAM2 proposals differently.

#### YCB-V: effect of the SAM2 stability threshold

`stab` is `stability_score_thresh` in `configs/model/segmentor_model/sam2.yaml` (default 0.85):

| SAM2 stab | Task | AP | AP50 | AP75 | AR100 | time / image |
|---|---|---|---|---|---|---|
| 0.85 (default) | 2D segmentation | 65.6 | 87.0 | 76.4 | 72.8 | 1.24 s |
| 0.85 (default) | 2D detection (amodal bbox) | 57.3 | 81.5 | 64.3 | 66.4 | 1.24 s |
| 0.90 | 2D segmentation | 65.1 | 86.0 | 75.8 | 72.2 | 1.19 s |
| 0.90 | 2D detection (amodal bbox) | 57.5 | 80.9 | 64.5 | 66.3 | 1.19 s |
| 0.97 (SAM v1 value) | 2D segmentation | 51.7 | 63.4 | 60.9 | 55.1 | 1.02 s |
| 0.97 (SAM v1 value) | 2D detection (amodal bbox) | 45.3 | 60.1 | 50.5 | 49.4 | 1.02 s |

The stability threshold matters: 0.97 was inherited from the original SAM v1 config, but SAM2's
stability scores are distributed differently and that value discards most of its proposals
(proposal recall at IoU 0.5 drops from 98% to 71% on YCB-V; dark objects such as the clamps and the
marker get no proposal at all). DINOv3 matching is not the bottleneck: when a proposal overlaps a
ground-truth instance, its label is correct 99.9% of the time. Denser point grids
(`points_per_side=64`) and `use_m2m` did not help in a subset ablation (`python -m src.scripts.ablate_sam2_ycbv`).

## 6D pose estimation (training-free)

`run_pose.py` implements stage 2 of the [Method](#method): the 6D pose of every instance found by the
localisation stage, from the RGB image alone, with the same frozen DINOv3 ViT-L/16 and the same 42
templates. It follows the template matching + PnP recipe of training-free methods such as
FoundPose [4] (DINOv2 features) and Pos3R [5] (MASt3R features), with DINOv3 patch features as the
only matcher.

### Usage

The segmentation run of the dataset has to exist (it provides the detections and the templates).

```bash
bash src/scripts/run_bop_dataset.sh ycbv full                                 # segmentation (once)
bash src/scripts/run_bop_pose.sh ycbv                                         # coarse pose -> csv -> scores
TAG=refine bash src/scripts/run_bop_pose.sh ycbv --n_hyp 3 --refine_iters 2   # with refinement
python -m src.scripts.summarize_pose_results                                  # results table (add --tag refine)
```

`run_bop_pose.sh` calls `run_pose.py` and then `eval_bop_pose.sh`. `run_pose.py` writes
`METHOD[TAG]_DATASET-SPLIT.csv` (the naming required by the BOP toolkit and website) and a
`_stats.json` with counts, timings and diagnostics to `datasets/bop23_challenge/results/pose_exps/`.
`eval_bop_pose.sh` runs `eval_bop19_pose.py` of `bop_toolkit` (VSD, MSSD and MSPD average recalls,
scores in `results/bop_eval_pose/`). ITODD and HB have no public test ground truth: their csv files
have to be uploaded to the [BOP website](https://bop.felk.cvut.cz/) (6D localization of unseen objects).

The method name and the tag of a run go through the environment variables `METHOD` and `TAG` of
`run_bop_pose.sh` (it needs them to find the csv it evaluates); every other argument is passed on to
`run_pose.py`.

Main options of `run_pose.py`, defaults in brackets:

| Option | Meaning |
|---|---|
| `--layers` [19] | DINOv3 block(s) the patch features are taken from (comma separated, 0-based) |
| `--crop_size` [448] | side of the query / template crops in px (multiple of 16) |
| `--q_rolls` [4] | query rotations by multiples of 90 degrees (1, 2 or 4) |
| `--topk` [5] | (rotation, template) pairs passed to PnP |
| `--pnp_thr` [20] | RANSAC reprojection threshold in px of the crop |
| `--pose_set` [auto42] | templates: `auto42`, `seg42` (always the segmentation templates) or own renderings (`level0x1`, `cube8x5`, ...) |
| `--dinov3_repo` [./dinov3_repo], `--dinov3_weights` | DINOv3 repository and ViT-L/16 checkpoint (default: the LVD-1689M weights in `<repo>/weights/`) |
| `--n_hyp` [1], `--refine_iters` [0] | render-and-compare hypothesis selection and refinement |
| `--image_stride`, `--max_images`, `--tag` | subset runs; a tag is mandatory for them so that they never overwrite a full result |
| `--oracle` | diagnostic: the template nearest to the ground-truth view is used |
| `--cache` | keep the template features on disk (about 67 MB per object); by default they are rebuilt at every run, about one second per object |

When test ground truth is available, `run_pose.py` also logs a quick MSSD / MSPD recall and error
statistics for ablations. The ground truth is never used for the estimates unless `--oracle` is set.

### Pose results

All numbers: BOP'19 test targets, Free6D localisation with pyrender templates (see [Localisation results](#localisation-results)), DINOv3 ViT-L/16 patch features, 42 templates per object,
single RTX 5070. LM-O, T-LESS, TUD-L, IC-BIN and YCB-V are scored locally with `bop_toolkit`
(`eval_bop19_pose.py`); ITODD and HB have no public test ground truth and are scored by the BOP
website with the same script (uploaded csv files in
`datasets/bop23_challenge/results/bop_submission_pose/`, returned scores in `docs/bop_website_scores/`).
The tables are generated with `python -m src.scripts.summarize_pose_results [--tag refine]`.

#### Average recall (AR = mean of AR_VSD, AR_MSSD, AR_MSPD)

| Method | Training-free | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Mean |
|---|---|---|---|---|---|---|---|---|---|
| FoundPose [4], coarse (DINOv2 patches) | yes | 39.6 | 33.8 | 46.7 | 23.9 | 20.4 | 50.8 | 45.2 | 37.2 |
| Pos3R [5], coarse (MASt3R, 40 templates) | yes | 32.3 | 31.5 | 47.3 | 33.1 | 25.1 | 53.7 | 53.9 | 39.5 |
| Free6D, coarse (`free6d`) | yes | 43.5 | 44.3 | 52.2 | 34.2 | 24.1 | 56.4 | 56.4 | 44.4 |
| Free6D, refined (`free6drefine`) | yes | 48.8 | 48.4 | 59.4 | 38.6 | 28.6 | 64.6 | 58.7 | 49.6 |

The first two rows are Table 1 of the Pos3R paper [5]. Without refinement Free6D is ahead on six of
the seven datasets; the exception is ITODD (24.1 against 25.1 for Pos3R), the dataset on which the
localisation misses the most target instances (35%). The comparison is end to end, not of the pose
stage alone: those methods start from the CNOS [1] detections (segmentation AP 40.4 to 41.2 in the
segmentation table above), Free6D from its own localisation (45.6). Refiners trained on pose data
remain ahead of the training-free refinement: Pos3R with the MegaPose refiner reaches 57.3 on the
seven datasets.

#### Per-dataset details: coarse (`free6d`)

| | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | mean |
|---|---|---|---|---|---|---|---|---|
| AR_VSD | 30.6 | 37.3 | 39.1 | 29.2 | 18.2* | 49.1* | 47.1 | 35.8 |
| AR_MSSD | 39.5 | 39.6 | 50.0 | 29.8 | 19.0* | 54.0* | 53.0 | 40.7 |
| AR_MSPD | 60.5 | 56.1 | 67.4 | 43.7 | 34.9* | 66.2* | 69.1 | 56.8 |
| AR | 43.5 | 44.3 | 52.2 | 34.2 | 24.1* | 56.4* | 56.4 | 44.4 |
| time/img (s) | 1.96 | 1.77 | 1.26 | 2.48 | 1.49* | 1.70* | 1.66 | 1.76 |
| pose time/img (s) | 0.69 | 0.49 | 0.09 | 1.23 | 0.28 | 0.46 | 0.42 | 0.53 |
| estimates / instances | 1342 / 1445 | 5045 / 6423 | 600 / 600 | 1732 / 1786 | 1982 / 3041 | 1377 / 1630 | 4013 / 4123 | - |
| fallbacks | 1 | 18 | 2 | 10 | 1 | 0 | 4 | - |

\* scores returned by the BOP website for ITODD, HB (copied from `docs/bop_website_scores/`), not computed locally.

#### Per-dataset details: refined (`free6drefine`, `--n_hyp 3 --refine_iters 2`)

| | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | mean |
|---|---|---|---|---|---|---|---|---|
| AR_VSD | 36.2 | 42.0 | 47.4 | 34.5 | 22.5* | 59.8* | 48.8 | 41.6 |
| AR_MSSD | 46.3 | 43.6 | 57.5 | 34.6 | 23.4* | 62.8* | 54.0 | 46.0 |
| AR_MSPD | 64.0 | 59.6 | 73.4 | 46.6 | 39.9* | 71.3* | 73.3 | 61.2 |
| AR | 48.8 | 48.4 | 59.4 | 38.6 | 28.6* | 64.6* | 58.7 | 49.6 |
| time/img (s) | 3.27 | 2.72 | 1.45 | 4.67 | 2.07* | 2.60* | 2.52 | 2.76 |
| pose time/img (s) | 2.00 | 1.44 | 0.28 | 3.43 | 0.80 | 1.36 | 1.28 | 1.51 |
| estimates / instances | 1342 / 1445 | 5045 / 6423 | 600 / 600 | 1732 / 1786 | 1982 / 3041 | 1377 / 1630 | 4013 / 4123 | - |
| fallbacks | 1 | 18 | 2 | 10 | 1 | 0 | 4 | - |

\* scores returned by the BOP website for ITODD, HB (copied from `docs/bop_website_scores/`), not computed locally.

- **Time.** `time/img` is what the BOP evaluation reports: localisation (about 1.2 s) plus the pose
  stage, which costs about 0.1 s per object (coarse) and 0.3 s per object (refined); the per-image pose
  time ranges from 0.09 s (TUD-L, one object per image) to 1.23 s (IC-BIN, about 12 objects per image).
  GPU memory peaks at about 7 GB with the default 448 px crops.
- **Onboarding.** No additional templates for 103 of the 108 objects of the test targets: the 42
  localisation templates are reused, their patch features are computed in about one second per object
  at the start of a run. Five objects get own renderings of the same 42 viewpoints: TUD-L 1 and 3 and
  T-LESS 7 and 8 (templates cut by the image border) and ITODD 28 (a 24 mm washer whose templates are
  35 px wide: their mask IoU with the re-rendering is 0.88, below the 0.9 threshold).
- **Missing detections.** A pose is estimated only for detected instances (`estimates / instances`):
  100% of the target instances on TUD-L, 97% on YCB-V and IC-BIN, 93% on LM-O, 84% on HB, 79% on T-LESS
  and 65% on ITODD. The others count as failures in the AR.
- **Reproducibility.** Two runs give bit-identical csv files. This required a single-sampled
  renderer: the 4x multisampled colour images of pyrender differ by one grey level in a few pixels from
  call to call, enough to change matches and RANSAC inlier sets (`src/pose/render.py::disable_msaa`).
  With multisampling two refined runs differed by up to 0.4 AR on a dataset (IC-BIN: 39.1 and 39.5);
  the means over the five locally scored datasets were then 51.2 refined and 46.2 coarse, against
  50.8 and 46.1 single-sampled.

#### Shared or own templates

Coarse AR on the five locally scored datasets with the three template options (run with the
multisampled renderer, before it was made deterministic; the corresponding final `auto42` numbers
are the coarse row above):

| Templates (`--pose_set`) | LM-O | T-LESS | TUD-L | IC-BIN | YCB-V | Mean |
|---|---|---|---|---|---|---|
| segmentation templates for every object (`seg42`) | 44.0 | 44.3 | 49.4 | 34.1 | 56.3 | 45.6 |
| own renderings of the 42 viewpoints for every object (`level0x1`) | 42.6 | 44.0 | 52.1 | 33.3 | 52.9 | 45.0 |
| shared unless cut by the image border (`auto42`, default) | 44.0 | 44.4 | 52.6 | 34.1 | 56.3 | 46.3 |

The templates of the segmentation stage are at least as good for pose as renderings made for it,
except for objects too large for the fixed 0.4 m rendering distance (two of the three TUD-L objects).

#### Ablations (YCB-V subset)

Every 6th target image of YCB-V (150 images, 686 instances), one change at a time from the default
coarse configuration. The recalls are the quick local ones of `run_pose.py` (MSSD and MSPD average
recalls, no VSD); differences below about 0.02 are within the noise of this subset.

| Configuration | AR_MSSD | AR_MSPD | Mean | Pose time / image |
|---|---|---|---|---|
| default coarse (block 19, 4 query rotations, 448 px crops, `auto42`) | 0.537 | 0.698 | 0.617 | 0.42 s |
| block 23 (last) | 0.525 | 0.653 | 0.589 | 0.38 s |
| block 15 | 0.534 | 0.658 | 0.596 | 0.42 s |
| block 11 | 0.458 | 0.572 | 0.515 | 0.43 s |
| 1 query rotation (`--q_rolls 1`) | 0.481 | 0.620 | 0.550 | 0.18 s |
| 2 query rotations | 0.513 | 0.665 | 0.589 | 0.26 s |
| 1 pair passed to PnP (`--topk 1`) | 0.532 | 0.679 | 0.606 | 0.35 s |
| RANSAC threshold 12 px | 0.536 | 0.685 | 0.610 | 0.50 s |
| 560 px crops (threshold 25 px) | 0.544 | 0.721 | 0.633 | 0.64 s |
| own renderings of the 42 viewpoints (`level0x1`) | 0.472 | 0.688 | 0.580 | 0.42 s |
| 8 cube-corner views x 5 in-plane rotations (`cube8x5`, the Pos3R layout) | 0.439 | 0.619 | 0.529 | 0.43 s |
| oracle: template nearest to the ground-truth view | 0.550 | 0.713 | 0.631 | 0.46 s |
| refinement of the best hypothesis (`--refine_iters 2`) | 0.540 | 0.720 | 0.630 | 0.70 s |
| selection among 3 rendered hypotheses (`--n_hyp 3`) | 0.554 | 0.716 | 0.635 | 0.75 s |
| default refined (`--n_hyp 3 --refine_iters 2`) | 0.541 | 0.737 | 0.639 | 1.30 s |
| 5 hypotheses, refined | 0.538 | 0.737 | 0.637 | 1.89 s |
| first working version (block 23, 1 rotation, threshold 12 px) | 0.439 | 0.552 | 0.495 | 0.21 s |

- **In-plane rotation is the main weakness of DINOv3 patch features, and query rotations fix most of
  it.** Feeding the query crop in 4 orientations is the largest single gain (0.550 to 0.617) and keeps
  the template set at 42 views.
- **An intermediate block matches better than the last one** (block 19: 0.617, block 23: 0.589).
- **Template selection is not the bottleneck any more**: picking the template nearest to the
  ground-truth view adds 0.014.
- **Viewpoint coverage matters more than in-plane copies.** The 40-template layout of Pos3R (8 views x 5
  in-plane rotations) is clearly worse with DINOv3 features (0.529), in line with Table 3 of the Pos3R
  paper where DINOv2 fails in that layout; 42 views every 32 degrees plus query rotations work.
- **Refinement improves the 2D alignment (MSPD), much less the 3D one (MSSD).** The remaining error is
  mostly along the viewing direction: on this subset the median translation error of the coarse poses
  is 25 mm, of which only 5.4 mm lateral; the median relative depth error is 2.8%, with a bias of +2.3%
  on YCB-V (between -0.6% and +1.0% on the other four datasets). The 16 px patch grid is a likely
  cause: it limits how well the object scale, hence its distance, is resolved.

### Known issues and next steps

- **Templates of small parts rendered before the unit fix.** The unit of a CAD model used to be
  guessed from its size, and ITODD object 28 (a 24 x 24 x 2 mm washer) was rendered 1000 times too
  large: it was never detected (0 of its 249 target instances, 101 after the fix; ITODD segmentation
  AP 23.7 before, 24.2 after). BOP models are now always read as millimetres (`--unit mm` of
  `src/poses/pyrender.py`) and the ITODD results above use the fixed templates. Templates rendered
  before the fix are not re-rendered automatically: delete `templates_pyrender/itodd/obj_000028/` and
  run `run_bop_dataset.sh itodd full` again.
- **Sensitivity of RANSAC.** Scores are stable, individual poses less so: switching the renderer
  from multisampled to single-sampled changed almost every pose slightly (median 13 mm in translation
  on YCB-V) while the coarse AR moved by at most 0.5 per dataset (refined: at most 0.9, IC-BIN). A more
  stable final fit (iterating inlier selection and least squares to convergence) is a direct candidate
  to reduce the depth error.
- **Depth accuracy.** Sub-patch localisation of the matches (feature upsampling or a featuremetric
  alignment) would address the scale / distance error that the refinement does not remove.
- **Occlusion and symmetry.** LM-O and IC-BIN (occlusion, clutter) and the symmetric T-LESS parts remain
  the hardest cases, as for the methods compared above.
- **Own templates.** Renderings fitted to the object are worse than the segmentation templates on YCB-V
  (0.580 against 0.617); their camera distance (5 object diameters) and lighting were not tuned.
- **Edge cases of the pose stage.** A mask so thin that no 16 px patch is at least half covered has no
  query feature: its pose is a fallback from an arbitrary template (it is still written, with the
  score of the detection). The template features of all objects of a dataset are kept on the GPU
  (about 67 MB per object with 42 templates), which bounds the number of templates per object on a
  12 GB card. The error statistics that `run_pose.py` logs associate each estimate with the nearest
  ground-truth instance of its object (not one-to-one); the scores come from `bop_toolkit`.

## Configuration

- `configs/model/segmentor_model/sam2.yaml`: `model_cfg` / `checkpoint_path` select the SAM2 size.
  Switching to `sam2.1_hiera_s.yaml` + `sam2.1_hiera_small.pt` cuts memory use substantially.
  `points_per_batch`, `box_nms_thresh` and `min_mask_region_area` map to the automatic mask generator.
  `stability_score_thresh` (0.85) controls how many proposals SAM2 keeps; see [Localisation results](#localisation-results).
- `configs/model/descriptor_model/dinov3.yaml`: `model_name` and `weights` select the DINOv3 size;
  `image_size: 224` and `patch_size: 16` must stay divisible.
- `configs/model/free6d.yaml`: `segmentor_width_size` is the width SAM2 runs at; `post_processing_config`
  filters small boxes and masks and applies NMS; `matching_config` defines score aggregation;
  `onboarding_config.level_templates` picks the template density for BOP runs.
- `configs/machine/trainer/local.yaml`: pytorch-lightning 2.x `Trainer` used by `run_inference.py`
  (`precision: bf16-mixed`, single GPU, `limit_test_batches` for smoke tests).
- Template descriptor caches (`descriptors_<model>_<pyrender|pbr>_level<k>.pth` next to the templates)
  are recomputed when a pyrender template is newer than the cache. The cache of the BlenderProc
  templates does not depend on the selection parameters (`reference_dataloader.pbr` in
  `configs/data/bop.yaml`): after changing them, run once with
  `model.onboarding_config.reset_descriptors=True`.

## Troubleshooting

- `python: command not found`: the `free6d` environment is not active.
- Rendering fails with an EGL error: `src/poses/pyrender.py` uses EGL offscreen rendering, which needs
  a working GPU driver; without one, switch `PYOPENGL_PLATFORM` to `osmesa`.
- Memory: SAM2-L with DINOv3-L takes about 12 s end to end and roughly 4 GB of VRAM on a 12 GB card.
- `No template (*.png) found` during inference: run `render_custom.sh` first, or check `OUTPUT_DIR`.
- The 6D pose evaluation cannot create an OpenGL context: `bop_toolkit` renders with vispy through
  EGL; run it with `DISPLAY` unset and `PYOPENGL_PLATFORM=egl` (`src/scripts/run_bop_pose.sh` does both).

## References

Free6D uses [SAM 2](https://github.com/facebookresearch/sam2) and
[DINOv3](https://github.com/facebookresearch/dinov3) with their public weights; their licenses apply
(the DINOv3 weights are governed by the DINOv3 License). The code of this repository is released under
the MIT license, see `LICENSE`.

1. V. N. Nguyen, T. Groueix, G. Ponimatkin, V. Lepetit, T. Hodan. *CNOS: A Strong Baseline for
   CAD-based Novel Object Segmentation.* ICCV Workshops, 2023.
2. N. Ravi et al. *SAM 2: Segment Anything in Images and Videos.* 2024.
3. O. Siméoni et al. *DINOv3.* 2025.
4. E. P. Örnek, Y. Labbé, B. Tekin, L. Ma, C. Keskin, C. Forster, T. Hodan. *FoundPose: Unseen Object
   Pose Estimation with Foundation Features.* ECCV, 2024.
5. W. Deng, D. Campbell, C. Sun, J. Zhang, S. Kanitkar, M. E. Shaffer, S. Gould. *Pos3R: 6D Pose
   Estimation for Unseen Objects Made Easy.* CVPR, 2025.
6. T. Hodan et al. *BOP Challenge 2023 on Detection, Segmentation and Pose Estimation of Seen and
   Unseen Rigid Objects.* CVPR Workshops, 2024.

BibTeX of [1]:

```latex
@inproceedings{nguyen2023cnos,
title={CNOS: A Strong Baseline for CAD-based Novel Object Segmentation},
author={Nguyen, Van Nguyen and Groueix, Thibault and Ponimatkin, Georgy and Lepetit, Vincent and Hodan, Tomas},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={2134--2140},
year={2023}
}
```
