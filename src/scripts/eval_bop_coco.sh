#!/usr/bin/env bash
# Evaluate the localisation predictions on a BOP dataset with bop_toolkit (COCO AP/AR of the
# BOP 2D detection / segmentation tasks), locally instead of on the BOP website.
#
# usage: DATASET_NAME=ycbv bash src/scripts/eval_bop_coco.sh <prediction.json> [method_name]
#
#   prediction.json : file written by run_inference.py, e.g.
#                     datasets/bop23_challenge/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_ycbv.json
#   method_name     : bop_toolkit wants METHOD_DATASET-SPLIT.json; METHOD must not contain "_" (default: free6d)
#
# env: DATASET_NAME (default ycbv), ROOT_DIR (default ./datasets/bop23_challenge),
#      BOP_TOOLKIT_DIR (default ./bop_toolkit_repo), PYTHON (default python)
set -euo pipefail

PYTHON=${PYTHON:-python}
DATASET_NAME=${DATASET_NAME:-ycbv}
ROOT_DIR=${ROOT_DIR:-./datasets/bop23_challenge}
BOP_TOOLKIT_DIR=${BOP_TOOLKIT_DIR:-./bop_toolkit_repo}
RESULT_JSON=${1:?usage: DATASET_NAME=ycbv $0 <prediction.json> [method_name]}
METHOD=${2:-free6d}

if [[ "$METHOD" == *_* ]]; then
  echo "method_name '$METHOD' must not contain '_' (bop_toolkit parses METHOD_DATASET-SPLIT.json)" >&2
  exit 1
fi

SPLIT=test
SPLIT_TYPE=""
if [[ "$DATASET_NAME" == "tless" || "$DATASET_NAME" == "hb" ]]; then
  SPLIT_TYPE=primesense
fi

DATASETS_PATH=$(realpath "$ROOT_DIR/datasets")
EVAL_DIR=$(realpath -m "$ROOT_DIR/results/bop_eval")
mkdir -p "$EVAL_DIR"
# bop_toolkit_lib.config reads the dataset location from this variable
export BOP_PATH="$DATASETS_PATH"

SPLIT_DIR="$DATASETS_PATH/$DATASET_NAME/$SPLIT${SPLIT_TYPE:+_$SPLIT_TYPE}"

# 1. ground truth in COCO format (scene_gt_coco.json per scene), computed once
if ! ls "$SPLIT_DIR"/*/scene_gt_coco.json >/dev/null 2>&1; then
  echo "== computing scene_gt_coco.json for $DATASET_NAME ($SPLIT_DIR)"
  $PYTHON "$BOP_TOOLKIT_DIR/scripts/calc_gt_coco.py" \
    --dataset "$DATASET_NAME" --dataset_split "$SPLIT" \
    ${SPLIT_TYPE:+--dataset_split_type "$SPLIT_TYPE"} \
    --datasets_path "$DATASETS_PATH"
fi

# 2. copy predictions under the file name expected by bop_toolkit
RESULT_NAME="${METHOD}_${DATASET_NAME}-${SPLIT}${SPLIT_TYPE:+-$SPLIT_TYPE}.json"
cp "$RESULT_JSON" "$EVAL_DIR/$RESULT_NAME"
echo "== evaluating $EVAL_DIR/$RESULT_NAME"

# 3. segmentation AP (BOP 2D segmentation task) and amodal bbox AP (2D detection task)
for ANN_TYPE in segm bbox; do
  $PYTHON "$BOP_TOOLKIT_DIR/scripts/eval_bop22_coco.py" \
    --result_filenames "$RESULT_NAME" \
    --results_path "$EVAL_DIR" \
    --eval_path "$EVAL_DIR" \
    --ann_type "$ANN_TYPE"
done

echo "== scores written to $EVAL_DIR/${RESULT_NAME%.json}/"
ls "$EVAL_DIR/${RESULT_NAME%.json}/"
