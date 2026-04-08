#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_ROOT/parameter-golf}"
LOG_DIR="${LOG_DIR:-$WORKSPACE_ROOT/hosted_logs/sota_smooth_ab}"

MODE="${MODE:-pair}"
TRAIN_SHARDS="${TRAIN_SHARDS:-1}"
MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-900}"
VAL_TOKENS_MAX="${VAL_TOKENS_MAX:-2097152}"
SMOOTH_LAMBDA="${SMOOTH_LAMBDA:-2e-5}"
SEED="${SEED:-314}"

mkdir -p "$LOG_DIR"

echo "bundle_dir=$BUNDLE_DIR"
echo "repo_dir=$REPO_DIR"
echo "log_dir=$LOG_DIR"
echo "mode=$MODE train_shards=$TRAIN_SHARDS max_wallclock=$MAX_WALLCLOCK_SECONDS val_tokens_max=$VAL_TOKENS_MAX smooth_lambda=$SMOOTH_LAMBDA seed=$SEED"

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/openai/parameter-golf.git "$REPO_DIR"
fi

cd "$REPO_DIR"
cp "$BUNDLE_DIR/train_gpt.py" "$REPO_DIR/train_gpt.py"

if ! python3 - <<'PY'
import importlib
mods = ["sentencepiece", "zstandard", "flash_attn_interface"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing:" + ",".join(missing))
print("deps_ok")
PY
then
  pip install --break-system-packages flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291
  pip install --break-system-packages sentencepiece zstandard huggingface-hub datasets tqdm
fi

if [ ! -f "$REPO_DIR/data/tokenizers/fineweb_1024_bpe.model" ] || [ ! -d "$REPO_DIR/data/datasets/fineweb10B_sp1024" ]; then
  python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"
fi

run_trial() {
  local name="$1"
  local smooth="$2"
  local logfile="$LOG_DIR/${name}.log"

  echo "starting trial=$name smooth_lambda=$smooth"
  env \
    RUN_ID="$name" \
    DATA_PATH="./data/datasets/fineweb10B_sp1024" \
    TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model" \
    VOCAB_SIZE=1024 \
    BIGRAM_VOCAB_SIZE=3072 \
    BIGRAM_DIM=112 \
    SEED="$SEED" \
    TRAIN_LOG_EVERY=100 \
    VAL_LOSS_EVERY=0 \
    VAL_TOKENS_MAX="$VAL_TOKENS_MAX" \
    MAX_WALLCLOCK_SECONDS="$MAX_WALLCLOCK_SECONDS" \
    WARMDOWN_ITERS=400 \
    TARGET_MB=15.9 \
    SMOOTH_LAMBDA="$smooth" \
    torchrun --standalone --nproc_per_node=1 train_gpt.py | tee "$logfile"
}

case "$MODE" in
  baseline)
    run_trial "sota_baseline_h100x1" "0"
    ;;
  smooth)
    run_trial "sota_smooth_h100x1" "$SMOOTH_LAMBDA"
    ;;
  pair)
    run_trial "sota_baseline_h100x1" "0"
    run_trial "sota_smooth_h100x1" "$SMOOTH_LAMBDA"
    ;;
  *)
    echo "unknown MODE=$MODE" >&2
    exit 1
    ;;
esac

python3 "$BUNDLE_DIR/summarize_logs.py" "$LOG_DIR"
