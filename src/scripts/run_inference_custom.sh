PYTHON=${PYTHON:-python3}

"$PYTHON" -m src.scripts.inference_custom \
  --template_dir "$OUTPUT_DIR" \
  --rgb_path "$RGB_PATH" \
  --mask_path "${MASK_PATH:-$OUTPUT_DIR/free6d_results/mask.png}" \
  --stability_score_thresh 0.8
