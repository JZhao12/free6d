#!/usr/bin/env bash
# Download the BlenderProc4BOP `train_pbr` split of the BOP'23 core datasets and extract
# it into datasets/bop23_challenge/datasets/<dataset>/train_pbr/, the layout read by
# src/dataloader/bop_pbr.py (PBR templates, `model.onboarding_config.rendering_type=pbr`).
#
# usage: bash src/scripts/download_train_pbr.sh [ycbv lmo tless tudl icbin hb itodd]
#   (no argument = all seven, in that order). LM-O uses the LM archive (lm_train_pbr.zip).
#
# Each archive (21-37 GB) is fetched from the Hugging Face mirror as 256 MB byte-range
# parts, $PARTS of them in parallel (one stream is capped at ~5 MB/s, the aggregate at
# ~11 MB/s); an interrupted run resumes at byte granularity. The parts are concatenated,
# checked with `unzip -t` and extracted in the background while the next archive downloads.
# Archives stay in datasets/bop23_challenge/datasets/tmp/ (delete them once the extraction
# is verified); the .ok / .extracted markers make re-runs skip finished work.
set -euo pipefail
cd "$(dirname "$0")/../.."
DATASETS=("$@"); [[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(ycbv lmo tless tudl icbin hb itodd)
ROOT=${ROOT_DIR:-./datasets/bop23_challenge}/datasets
TMP=$ROOT/tmp
PARTS=${PARTS:-8}
CHUNK=$((256 * 1024 * 1024))
MIRROR=https://huggingface.co/datasets/bop-benchmark
mkdir -p "$TMP" logs

repo_of() { case $1 in lmo) echo lm;; *) echo "$1";; esac; }

# fetch_part <url> <part file> <first byte> <last byte>: appends the missing tail of the
# range, so a killed transfer resumes where it stopped (no curl --retry: a retry inside
# curl would re-send bytes that were already appended)
fetch_part() {
  local url=$1 out=$2 a=$3 b=$4 want=$(( $4 - $3 + 1 )) have try
  for try in $(seq 1 20); do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ $have -gt $want ]]; then
      echo "  [!] $(basename "$out") has $have > $want bytes, restarting it" >&2; rm -f "$out"; have=0
    fi
    [[ $have -eq $want ]] && { echo "  [part] $(basename "$out") ok"; return 0; }
    curl -sS -L --fail --speed-limit 20000 --speed-time 60 -r $(( a + have ))-"$b" "$url" >> "$out" \
      || { echo "  [retry $try] $(basename "$out") at $(stat -c %s "$out" 2>/dev/null || echo 0)/$want" >&2; sleep 20; }
  done
  have=$(stat -c %s "$out" 2>/dev/null || echo 0)
  [[ $have -eq $want ]] || { echo "  [FAILED] $out: $have/$want bytes" >&2; return 1; }
}
export -f fetch_part

# verify_and_extract <dataset> <zip> <expected size> <parts dir>; runs in the background
verify_and_extract() {
  local ds=$1 zip=$2 size=$3 pdir=$4 dst
  dst=$(dirname "$zip")/../$ds
  if [[ ! -f $zip.ok ]]; then
    cat "$pdir"/p* > "$zip"
    [[ $(stat -c %s "$zip") -eq $size ]] || { echo "ERROR: $zip is $(stat -c %s "$zip") bytes, expected $size" >&2; return 1; }
    unzip -tq "$zip" > /dev/null || { echo "ERROR: unzip -t failed on $zip" >&2; return 1; }
    rm -r "$pdir"; touch "$zip.ok"
    echo "== $ds: archive verified ($(du -h "$zip" | cut -f1), $(date '+%F %T'))"
  fi
  unzip -q -o "$zip" -d "$dst" || { echo "ERROR: extraction of $zip failed" >&2; return 1; }
  touch "$zip.extracted"
  echo "== $ds: extracted to $dst/train_pbr ($(ls -d "$dst"/train_pbr/*/ | wc -l) scenes, $(date '+%F %T'))"
}

pids=()
for ds in "${DATASETS[@]}"; do
  repo=$(repo_of "$ds"); f=${repo}_train_pbr.zip; zip=$TMP/$f; url=$MIRROR/$repo/resolve/main/$f
  [[ -f $zip.extracted ]] && { echo "== $ds: already extracted"; continue; }
  if [[ -f $zip.ok ]]; then
    size=$(stat -c %s "$zip"); pdir=""
  else
    size=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{s=$2} END{print s}')
    [[ ${size:-0} -gt 1000000000 ]] || { echo "ERROR: cannot resolve $url (content-length '${size:-}')" >&2; exit 1; }
    n=$(( (size + CHUNK - 1) / CHUNK )); pdir=$TMP/parts_$repo; mkdir -p "$pdir"
    echo "== $ds: $f $(( size / 1048576 )) MB in $n parts, $PARTS parallel ($(date '+%F %T'))"
    seq 0 $(( n - 1 )) | awk -v c="$CHUNK" -v s="$size" -v u="$url" -v d="$pdir" \
      '{a=$1*c; b=a+c-1; if (b>s-1) b=s-1; printf "%s %s/p%05d %d %d\n", u, d, $1, a, b}' \
      | xargs -P "$PARTS" -L 1 bash -c 'fetch_part "$@"' _
    echo "== $ds: all parts downloaded ($(date '+%F %T')), verifying and extracting in the background"
  fi
  verify_and_extract "$ds" "$zip" "$size" "$pdir" > "logs/extract_$ds.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
grep -h -E '^(==|ERROR)' logs/extract_*.log 2>/dev/null || true
echo "== download_train_pbr finished with status $status ($(date '+%F %T'))"
exit $status
