#!/usr/bin/env bash
# Download and extract the BOP'23 core datasets (base files, CAD models, BOP'19 test split)
# into datasets/bop23_challenge/datasets/<dataset>/, the layout used by run_inference.py.
#
# usage: bash src/scripts/download_bop_classic.sh [lmo tless tudl icbin itodd hb ycbv]
#        (no argument = all seven). Already extracted datasets are skipped, partial
#        downloads are resumed. Archives are kept in datasets/bop23_challenge/datasets/tmp/.
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(lmo tless tudl icbin itodd hb ycbv)
ROOT=${ROOT_DIR:-./datasets/bop23_challenge}/datasets
TMP=$ROOT/tmp
mkdir -p "$TMP"

# tried in order; {ds} is replaced by the dataset name
MIRRORS=(
  "https://huggingface.co/datasets/bop-benchmark/{ds}/resolve/main"
  "https://huggingface.co/datasets/bop-benchmark/datasets/resolve/main/{ds}"
  "https://bop.felk.cvut.cz/media/data/bop_datasets"
)

split_of()    { case $1 in tless|hb) echo test_primesense;; *) echo test;; esac; }
test_zip_of() { case $1 in tless|hb) echo "${1}_test_primesense_bop19.zip";; *) echo "${1}_test_bop19.zip";; esac; }
cad_dir_of()  { case $1 in tless) echo models_cad;; *) echo models;; esac; }

fetch() { # <dataset> <zip name>
  local ds=$1 f=$2 out=$TMP/$2 url
  [[ -f $out.ok ]] && { echo "  [cached] $f"; return 0; }
  for m in "${MIRRORS[@]}"; do
    url=${m//\{ds\}/$ds}/$f
    echo "  [get] $url"
    if wget -c -q --timeout=60 --tries=3 -O "$out" "$url" && unzip -tq "$out" >/dev/null 2>&1; then
      touch "$out.ok"; echo "  [ok]  $f ($(du -h "$out" | cut -f1))"; return 0
    fi
    rm -f "$out"  # do not resume a broken/HTML file from the next mirror
  done
  echo "  [FAILED] $f" >&2; return 1
}

extract() { # <dataset> <zip name>
  local ds=$1 f=$2 zip=$TMP/$2 dst=$ROOT/$1 x src
  [[ -f $TMP/$f.extracted ]] && { echo "  [already extracted] $f"; return 0; }
  x=$(mktemp -d "$TMP/x_${ds}_XXXX")
  unzip -q "$zip" -d "$x"
  # BOP archives usually wrap everything in a <dataset>/ folder; some mirrors do not
  if [[ -d $x/$ds ]]; then src=$x/$ds; else src=$x; fi
  mkdir -p "$dst"
  find "$src" -mindepth 1 -maxdepth 1 -exec mv -t "$dst" {} + 2>/dev/null \
    || { cp -a "$src"/. "$dst"/; }
  rm -rf "$x"
  touch "$TMP/$f.extracted"; echo "  [extracted] $f -> $dst"
}

for ds in "${DATASETS[@]}"; do
  split=$(split_of "$ds"); dst=$ROOT/$ds
  echo "== $ds"
  if [[ -d $dst/$split && -f $dst/test_targets_bop19.json && -d $dst/$(cad_dir_of "$ds") ]]; then
    echo "  [present] $dst already has $split/, $(cad_dir_of "$ds")/ and test_targets_bop19.json"
    continue
  fi
  for f in "${ds}_base.zip" "${ds}_models.zip" "$(test_zip_of "$ds")"; do
    fetch "$ds" "$f"
    extract "$ds" "$f"
  done
  echo "  [layout] $(ls "$dst" | tr '\n' ' ')"
  echo "  [scenes] $(ls -d "$dst/$split"/*/ 2>/dev/null | wc -l) in $split, CAD files: $(ls "$dst/$(cad_dir_of "$ds")"/obj_*.ply 2>/dev/null | wc -l)"
done
echo "== done: $(du -sh "$ROOT" | cut -f1) in $ROOT"
