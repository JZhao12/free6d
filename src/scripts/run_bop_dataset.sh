#!/usr/bin/env bash
# End-to-end localisation run (2D segmentation / detection) on one BOP dataset: templates -> inference -> bop_toolkit scores.
#
# usage: PYTHON=<python> bash src/scripts/run_bop_dataset.sh <dataset> [smoke|full]
#   smoke : 5 test images; predictions under results/smoke_<dataset>/, scores under
#           results/bop_eval/smoke_<dataset>-test*/ (pipeline check only, scores are meaningless)
#   full  : every BOP'19 test target; predictions under results/seg_exps/, scores under
#           results/bop_eval/free6d_<dataset>-test*/
#   RENDERING=pbr : use templates picked from the BlenderProc4BOP train_pbr frames instead of
#           the pyrender ones (needs src/scripts/download_train_pbr.sh <dataset> first);
#           predictions *_template_pbr0_*.json, scores under results/bop_eval/free6dpbr_<dataset>-test*/
# GPU stages of concurrent invocations are serialised with flock on $GPU_LOCK
# (default /tmp/free6d_gpu.lock); a 12 GB card fits one run at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
DS=${1:?usage: $0 <dataset> [smoke|full]}
MODE=${2:-full}
RENDERING=${RENDERING:-pyrender}
case $RENDERING in pyrender|pbr) ;; *) echo "ERROR: RENDERING must be pyrender or pbr, got '$RENDERING'" >&2; exit 1;; esac
ROOT=${ROOT_DIR:-./datasets/bop23_challenge}
LOCK=${GPU_LOCK:-/tmp/free6d_gpu.lock}
mkdir -p logs

case $DS in tless|hb) SPLIT=test_primesense; SPLIT_SUFFIX=-primesense;; *) SPLIT=test; SPLIT_SUFFIX="";; esac
case $DS in tless) CAD_DIR=$ROOT/datasets/$DS/models_cad;; *) CAD_DIR=$ROOT/datasets/$DS/models;; esac
DS_DIR=$ROOT/datasets/$DS
[[ -d $DS_DIR/$SPLIT ]] || { echo "ERROR: $DS_DIR/$SPLIT not found (run src/scripts/download_bop_classic.sh $DS)" >&2; exit 1; }
[[ -f $DS_DIR/test_targets_bop19.json ]] || { echo "ERROR: $DS_DIR/test_targets_bop19.json not found" >&2; exit 1; }
[[ -d $CAD_DIR ]] || { echo "ERROR: CAD folder $CAD_DIR not found" >&2; exit 1; }
N_OBJ=$(find "$CAD_DIR" -maxdepth 1 -name 'obj_*.ply' | wc -l)
N_IMG=$($PYTHON -c "import json;print(len({(t['scene_id'],t['im_id']) for t in json.load(open('$DS_DIR/test_targets_bop19.json'))}))")
echo "== $DS: $N_OBJ CAD models in $CAD_DIR, $N_IMG target images in $SPLIT, mode=$MODE, templates=$RENDERING"

# 1. templates
TPL=$ROOT/datasets/templates_pyrender/$DS
if [[ $RENDERING == pbr ]]; then
  # frames of the BlenderProc4BOP split, picked at inference time by src/dataloader/bop_pbr.py
  N_SCENES=$(ls -d "$DS_DIR"/train_pbr/*/ 2>/dev/null | wc -l)
  [[ $N_SCENES -ge 10 ]] || { echo "ERROR: $DS_DIR/train_pbr has $N_SCENES scenes, need 10 (run src/scripts/download_train_pbr.sh $DS)" >&2; exit 1; }
  echo "== PBR templates picked from $DS_DIR/train_pbr ($N_SCENES scenes)"
else
  # 642 pyrender views per object, rendered once
  N_DONE=$($PYTHON -c "import glob;print(sum(1 for d in glob.glob('$TPL/obj_*/') if len(glob.glob(d+'*.png'))==642))")
  if [[ $N_DONE -ne $N_OBJ ]]; then
    echo "== rendering templates ($N_DONE/$N_OBJ objects complete) -> logs/render_$DS.log"
    flock -w 7200 "$LOCK" $PYTHON -m src.scripts.render_template_with_pyrender dataset_name=$DS > logs/render_$DS.log 2>&1 \
      || { echo "ERROR: rendering failed" >&2; tail -5 logs/render_$DS.log >&2; exit 1; }
    echo "== rendered $(find "$TPL" -name '*.png' | wc -l) templates"
  else
    echo "== templates already rendered: $TPL ($N_OBJ objects x 642 views)"
  fi
fi

# 2. inference
if [[ $MODE == smoke ]]; then NAME_EXP=smoke_$DS; EXTRA="machine.trainer.limit_test_batches=5"; METHOD=smoke
else NAME_EXP=seg_exps; EXTRA=""; METHOD=free6d; fi
# bop_toolkit wants METHOD_DATASET-SPLIT.json with no "_" in METHOD, so pbr results use free6dpbr / smokepbr
if [[ $RENDERING == pbr ]]; then METHOD=${METHOD}pbr; TAG=${MODE}_pbr; else TAG=$MODE; fi
EXTRA="$EXTRA model.onboarding_config.rendering_type=$RENDERING"
echo "== inference -> $ROOT/results/$NAME_EXP (log: logs/infer_${DS}_$TAG.log)"
flock -w 7200 "$LOCK" $PYTHON run_inference.py dataset_name=$DS name_exp=$NAME_EXP $EXTRA > logs/infer_${DS}_$TAG.log 2>&1 \
  || { echo "ERROR: inference failed" >&2; grep -E 'Error|Traceback' -A3 logs/infer_${DS}_$TAG.log | tail -20 >&2; exit 1; }
PRED=$(ls "$ROOT/results/$NAME_EXP/"*"_template_${RENDERING}"[0-9]"_agg"*"_${DS}.json" | grep -v with_score_distribution)
N_NPZ=$(find "$ROOT/results/$NAME_EXP/predictions/$DS/$(basename "$PRED" .json)" -name '*_runtime.npz' | wc -l)
echo "== $N_NPZ images processed, predictions: $PRED"

# 3. format check (bop_toolkit) + per-image sanity: GT object ids vs predicted ids
$PYTHON - "$PRED" "$DS_DIR/$SPLIT" "$MODE" <<'EOF'
import json, sys, os.path as osp
from collections import defaultdict
from bop_toolkit_lib import inout
pred_path, split_dir, mode = sys.argv[1:4]
inout.check_coco_results(pred_path, ann_type="segm"); inout.check_coco_results(pred_path, ann_type="bbox")
d = json.load(open(pred_path)); by = defaultdict(list)
for x in d: by[(x["scene_id"], x["image_id"])].append(x)
print(f"== format OK: {len(d)} detections on {len(by)} images ({len(d)/max(len(by),1):.1f}/image), "
      f"score {min(x['score'] for x in d):.2f}-{max(x['score'] for x in d):.2f}, category ids {sorted({x['category_id'] for x in d})}")
if mode == "smoke":
    for (s, i), P in sorted(by.items()):
        hi = sorted({x["category_id"] for x in P if x["score"] > 0.4})
        gt_path = osp.join(split_dir, f"{s:06d}", "scene_gt.json")
        if osp.exists(gt_path):
            gt = sorted({o["obj_id"] for o in json.load(open(gt_path))[str(i)]})
            print(f"   scene {s} im {i}: GT {gt} | pred(score>0.4) {hi} | hit {len(set(gt) & set(hi))}/{len(gt)}")
        else:  # ITODD ships no scene_gt.json for the test split
            print(f"   scene {s} im {i}: GT n/a (no scene_gt.json) | pred(score>0.4) {hi} | {len(P)} dets")
EOF

# 4. evaluation (CPU only). ITODD ships no test ground truth (no scene_gt.json): its
#    predictions can only be scored by uploading $PRED to https://bop.felk.cvut.cz/
if ! ls "$DS_DIR/$SPLIT"/*/scene_gt.json >/dev/null 2>&1; then
  echo "== no scene_gt.json in $DS_DIR/$SPLIT: local evaluation impossible, upload $PRED to the BOP website"
  echo "== DONE $DS $MODE ($RENDERING templates, predictions only)"
  exit 0
fi
echo "== evaluation (log: logs/eval_${DS}_$TAG.log)"
DATASET_NAME=$DS PYTHON=$PYTHON ROOT_DIR=$ROOT bash src/scripts/eval_bop_coco.sh "$PRED" $METHOD > logs/eval_${DS}_$TAG.log 2>&1 \
  || { echo "ERROR: evaluation failed" >&2; tail -20 logs/eval_${DS}_$TAG.log >&2; exit 1; }
SCORE_DIR=$ROOT/results/bop_eval/${METHOD}_${DS}-test${SPLIT_SUFFIX}
$PYTHON - "$SCORE_DIR" "$MODE" <<'EOF'
import json, sys
score_dir, mode = sys.argv[1:3]
note = "  (smoke: 5 images, numbers are not meaningful)" if mode == "smoke" else ""
for t in ["segm", "bbox"]:
    s = json.load(open(f"{score_dir}/scores_bop22_coco_{t}.json"))
    print(f"== {t:4s} AP {s['AP']*100:5.1f}  AP50 {s['AP50']*100:5.1f}  AP75 {s['AP75']*100:5.1f}  "
          f"AR100 {s['AR100']*100:5.1f}  time/img {s['average_time_per_image']:.2f}s{note}")
EOF
echo "== DONE $DS $MODE ($RENDERING templates)"
