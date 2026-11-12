#!/usr/bin/env bash
# Evaluate 6D pose estimates on a BOP dataset with bop_toolkit (BOP'19 / BOP'20 pose task:
# VSD, MSSD, MSPD average recalls), locally instead of on the BOP website.
#
# usage: DATASET_NAME=ycbv bash src/scripts/eval_bop_pose.sh <result.csv>
#
#   result.csv : file written by run_pose.py, e.g.
#                datasets/bop23_challenge/results/pose_exps/free6d_ycbv-test.csv
#                (it must already be named METHOD_DATASET-SPLIT[-SPLITTYPE].csv, METHOD without "_")
#
# env: DATASET_NAME (default ycbv, only used for the log), ROOT_DIR (default ./datasets/bop23_challenge),
#      BOP_TOOLKIT_DIR (default ./bop_toolkit_repo), PYTHON (default python),
#      NUM_WORKERS (default 8), RENDERER (default vispy)
set -euo pipefail

PYTHON=${PYTHON:-python}
DATASET_NAME=${DATASET_NAME:-ycbv}
ROOT_DIR=${ROOT_DIR:-./datasets/bop23_challenge}
BOP_TOOLKIT_DIR=${BOP_TOOLKIT_DIR:-./bop_toolkit_repo}
NUM_WORKERS=${NUM_WORKERS:-8}
RENDERER=${RENDERER:-vispy}
RESULT_CSV=${1:?usage: DATASET_NAME=ycbv $0 <result.csv>}

RESULT_NAME=$(basename "$RESULT_CSV")
if [[ "$RESULT_NAME" != *_*-*.csv ]]; then
  echo "result file name '$RESULT_NAME' must look like METHOD_DATASET-SPLIT[-SPLITTYPE].csv (bop_toolkit parses it)" >&2
  exit 1
fi

DATASETS_PATH=$(realpath "$ROOT_DIR/datasets")
EVAL_DIR=$(realpath -m "$ROOT_DIR/results/bop_eval_pose")
mkdir -p "$EVAL_DIR"
# bop_toolkit_lib.config reads the dataset location from this variable
export BOP_PATH="$DATASETS_PATH"
# bop_toolkit spawns its sub-scripts with a literal `python`: make it the interpreter of $PYTHON
PYTHON_BIN_DIR=$(dirname "$(command -v "$PYTHON")")
export PATH="$PYTHON_BIN_DIR:$PATH"

# copy the estimates under the results folder of bop_toolkit
cp "$RESULT_CSV" "$EVAL_DIR/$RESULT_NAME"
echo "== evaluating $EVAL_DIR/$RESULT_NAME ($DATASET_NAME, renderer=$RENDERER, workers=$NUM_WORKERS)"

$PYTHON "$BOP_TOOLKIT_DIR/scripts/eval_bop19_pose.py" \
  --renderer_type="$RENDERER" \
  --result_filenames="$RESULT_NAME" \
  --results_path="$EVAL_DIR" \
  --eval_path="$EVAL_DIR" \
  --targets_filename=test_targets_bop19.json \
  --num_workers="$NUM_WORKERS"

SCORES="$EVAL_DIR/${RESULT_NAME%.csv}/scores_bop19.json"
echo "== scores written to $EVAL_DIR/${RESULT_NAME%.csv}/"
if [[ -f "$SCORES" ]]; then
  cat "$SCORES"
  echo
fi
