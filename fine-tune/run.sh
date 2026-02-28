#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh  — keeps restarting train.py until it exits with code 0 (success)
#           On every crash it clears Python/CUDA caches before retrying.
# Usage:   bash fine-tune/run.sh        (from project root)
#          bash run.sh                  (from inside fine-tune/)
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.py"

# !! KEY FIX: always cd to the PROJECT ROOT (one level above fine-tune/)
# so that train.py's OUTPUT_DIR="./uzbek-whisper" resolves to the correct
# location at the root of the project, not inside fine-tune/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

MAX_RETRIES=20
attempt=1

echo "========================================"
echo " Auto-restart trainer"
echo " Project root: $PROJECT_ROOT"
echo " Script      : $TRAIN_SCRIPT"
echo " Max retries : $MAX_RETRIES"
echo "========================================"

while [ $attempt -le $MAX_RETRIES ]; do
    echo ""
    echo "-------- Attempt $attempt / $MAX_RETRIES  ($(date '+%H:%M:%S')) --------"

    # ── Clear caches before every run ────────────────────────────────────────
    echo "Clearing caches..."
    rm -rf ~/.cache/huggingface/datasets/tmp* 2>/dev/null
    rm -rf ~/.cache/torch/kernels 2>/dev/null
    sync

    # ── Run training ─────────────────────────────────────────────────────────
    python "$TRAIN_SCRIPT"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "========================================"
        echo " Training COMPLETED successfully!"
        echo "========================================"
        exit 0
    fi

    echo ""
    echo "  ✗ Exited with code $EXIT_CODE — will auto-resume from last checkpoint"
    echo "  Waiting 5 seconds before retry..."
    sleep 5

    attempt=$((attempt + 1))
done

echo ""
echo "========================================"
echo " Reached max retries ($MAX_RETRIES). Giving up."
echo "========================================"
exit 1