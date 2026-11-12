#!/usr/bin/env bash
# One-shot 6D pose estimation of the Free6D detections on one BOP dataset: run_pose.py -> pose csv
# -> bop_toolkit scores (BOP'19 / BOP'20 pose task: VSD, MSSD, MSPD average recalls).
#
# usage: PYTHON=<python> [TAG=<tag>] [METHOD=<method>] bash src/scripts/run_bop_pose.sh <dataset> [extra run_pose.py args...]
#   METHOD : method name, no "_" (default free6d)
#   TAG    : appended to the method name for ablation runs, no "_" or "-" (default: empty)
#   (do not pass --tag / --method / --out_dir / --root as extra arguments: the script would evaluate another file)
#   csv    : results/pose_exps/<METHOD><TAG>_<dataset>-<split>.csv, scores under
#            results/bop_eval_pose/<METHOD><TAG>_<dataset>-<split>/
#   needs  : the segmentation result of src/scripts/run_bop_dataset.sh <dataset> full and the
#            42 level-0 pyrender templates of the segmentation stage are reused (objects whose
#            templates are cut by the image border get own renderings of the same viewpoints,
#            --pose_set auto42)
# The GPU stage of concurrent invocations is serialised with flock on $GPU_LOCK
# (default /tmp/free6d_gpu.lock); a 12 GB card fits one run at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-python}
DS=${1:?usage: $0 <dataset> [extra run_pose.py args...]}
shift
# the csv evaluated below is located from METHOD / TAG / ROOT_DIR: refuse the equivalent run_pose.py options
for arg in "$@"; do
  case $arg in
    --tag|--tag=*|--method|--method=*|--out_dir|--out_dir=*|--root|--root=*)
      echo "ERROR: '$arg' must not be passed as an extra argument: use the environment variables TAG=, METHOD= and ROOT_DIR= (e.g. TAG=fast bash $0 $DS --image_stride 10)" >&2
      exit 1;;
  esac
done
METHOD=${METHOD:-free6d}
TAG=${TAG:-}
ROOT=${ROOT_DIR:-./datasets/bop23_challenge}
LOCK=${GPU_LOCK:-/tmp/free6d_gpu.lock}
mkdir -p logs

case $DS in tless|hb) SPLIT=test_primesense;; *) SPLIT=test;; esac
DS_DIR=$ROOT/datasets/$DS
CSV=$ROOT/results/pose_exps/${METHOD}${TAG}_${DS}-${SPLIT//_/-}.csv
SEG=$ROOT/results/seg_exps/CustomSam2AutomaticMaskGenerator_template_pyrender0_aggavg_5_${DS}.json
RUN_LOG=logs/pose_${DS}${TAG:+_$TAG}.log
EVAL_LOG=logs/eval_pose_${DS}${TAG:+_$TAG}.log

[[ -d $DS_DIR/$SPLIT ]] || { echo "ERROR: $DS_DIR/$SPLIT not found (run src/scripts/download_bop_classic.sh $DS)" >&2; exit 1; }
[[ -f $DS_DIR/test_targets_bop19.json ]] || { echo "ERROR: $DS_DIR/test_targets_bop19.json not found" >&2; exit 1; }
[[ -f $SEG ]] || { echo "ERROR: segmentation result $SEG not found (run src/scripts/run_bop_dataset.sh $DS full first)" >&2; exit 1; }
compgen -G "$ROOT/datasets/templates_pyrender/$DS/obj_*/000000.png" >/dev/null \
  || { echo "ERROR: no level-0 templates in $ROOT/datasets/templates_pyrender/$DS (run src/scripts/run_bop_dataset.sh $DS full first)" >&2; exit 1; }
echo "== $DS: split $SPLIT, method ${METHOD}${TAG}, segmentation $SEG"

# 1. pose estimation (GPU)
export PYOPENGL_PLATFORM=egl
echo "== pose estimation -> $CSV (log: $RUN_LOG)"
flock -w 7200 "$LOCK" $PYTHON run_pose.py --dataset "$DS" --root "$ROOT" --method "$METHOD" ${TAG:+--tag "$TAG"} "$@" > "$RUN_LOG" 2>&1 \
  || { echo "ERROR: pose estimation failed" >&2; tail -20 "$RUN_LOG" >&2; exit 1; }
grep 'done:' "$RUN_LOG" || true

# 2. evaluation (CPU only). ITODD ships no test ground truth (no scene_gt.json): its estimates
#    can only be scored by uploading the csv to https://bop.felk.cvut.cz/
if ! ls "$DS_DIR/$SPLIT"/*/scene_gt.json >/dev/null 2>&1; then
  echo "== no scene_gt.json in $DS_DIR/$SPLIT: local evaluation impossible, upload $CSV to https://bop.felk.cvut.cz/ (6D localization of unseen objects, BOP'19-23 core datasets)"
  echo "== DONE $DS pose (${METHOD}${TAG})"
  exit 0
fi
echo "== evaluation (log: $EVAL_LOG)"
( unset DISPLAY; DATASET_NAME=$DS PYTHON=$PYTHON ROOT_DIR=$ROOT bash src/scripts/eval_bop_pose.sh "$CSV" ) > "$EVAL_LOG" 2>&1 \
  || { echo "ERROR: evaluation failed" >&2; tail -20 "$EVAL_LOG" >&2; exit 1; }
SCORES=$ROOT/results/bop_eval_pose/$(basename "$CSV" .csv)/scores_bop19.json
$PYTHON - "$SCORES" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"== AR_VSD {s['bop19_average_recall_vsd']*100:.1f}  AR_MSSD {s['bop19_average_recall_mssd']*100:.1f}  "
      f"AR_MSPD {s['bop19_average_recall_mspd']*100:.1f}  AR {s['bop19_average_recall']*100:.1f}  "
      f"time/img {s['bop19_average_time_per_image']:.2f}s")
EOF
echo "== DONE $DS pose (${METHOD}${TAG})"
