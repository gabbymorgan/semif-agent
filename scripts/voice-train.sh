#!/usr/bin/env bash
# Train a custom Piper TTS voice for the voice gateway from a recorded dataset.
#
# Runs on the TRAINING host (a machine with a real GPU), NOT on the gateway host
# that will use the voice. Everything it installs lives inside this checkout
# under .runtime/voice-train/ (gitignored): the piper1-gpl clone, its venv, the
# base checkpoint, the training cache, runs, and the exported voice. Only the
# finished <name>.onnx + <name>.onnx.json are copied to a gateway host.
#
# It fine-tunes from a pinned "medium" checkpoint (Piper has no zero-shot
# cloning; ~15-60 min of recorded audio gives a usable voice). The dataset is an
# LJSpeech directory (metadata.csv + wavs/) produced by scripts/voice-record.py.
# See docs/VOICE-TRAIN.md.
#
# Pinned external refs (piper1-gpl commit, base checkpoint url+sha256) live in the
# committed pins.json; bump those via git and rerun to upgrade.
#
# Usage:
#   scripts/voice-train.sh --name en_US-myvoice-medium [--dataset DIR] [--work DIR]
#                          [--quality medium] [--espeak-voice en-us]
#                          [--batch-size 16] [--max-epochs 2000]
#                          [--checkpoint PATH_OR_URL] [--torch-index-url URL]
#                          [--torch-spec "torch torchaudio"]
#                          [--free-gpu] [--resume] [--skip-setup|--skip-train|--skip-export] [-h]
set -euo pipefail

NAME="en_US-custom-medium"
DATASET=""
WORK=""
QUALITY="medium"
ESPEAK_VOICE="en-us"
SAMPLE_RATE="22050"
BATCH_SIZE="16"
MAX_EPOCHS="2000"
CHECKPOINT=""
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/rocm7.1"
TORCH_SPEC="torch torchaudio"
FREE_GPU=0
RESUME=0
SKIP_SETUP=0
SKIP_TRAIN=0
SKIP_EXPORT=0

usage() {
  sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "  --name ID             voice id / output basename (default: $NAME)"
  echo "  --dataset DIR         LJSpeech dataset dir (metadata.csv + wavs/)"
  echo "  --work DIR            scratch dir (default: .runtime/voice-train)"
  echo "  --quality Q           medium|high|low (must match the checkpoint; default medium)"
  echo "  --espeak-voice V      espeak-ng voice (default en-us)"
  echo "  --sample-rate HZ      dataset sample rate (default 22050)"
  echo "  --batch-size N        training batch size (default 16; 16 GB VRAM)"
  echo "  --max-epochs N        trainer.max_epochs (default 2000; fine-tune ~1000-3000)"
  echo "  --checkpoint P|URL    base .ckpt to fine-tune from (default: pinned in pins.json)"
  echo "  --torch-index-url U   ROCm wheel index for torch (default: $TORCH_INDEX_URL)"
  echo "  --torch-spec SPEC     torch packages to install (default: \"$TORCH_SPEC\")"
  echo "  --free-gpu            stop winnow + unload ollama models before training"
  echo "  --resume              resume from the newest run checkpoint under --work"
  echo "  --skip-setup          skip apt/clone/venv/torch/deps (reuse an existing env)"
  echo "  --skip-train          skip training (e.g. export an existing run)"
  echo "  --skip-export         stop after training"
  echo "  -h                    this help"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --work) WORK="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --espeak-voice) ESPEAK_VOICE="$2"; shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-epochs) MAX_EPOCHS="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL="$2"; shift 2 ;;
    --torch-spec) TORCH_SPEC="$2"; shift 2 ;;
    --free-gpu) FREE_GPU=1; shift ;;
    --resume) RESUME=1; shift ;;
    --skip-setup) SKIP_SETUP=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# apt/systemctl as root directly when the container has no sudo (common on cloud
# GPU images); fall back to sudo on a normal host.
if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=""
elif command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi
PINS="$REPO_ROOT/pins.json"
[[ -f "$PINS" ]] || { echo "pins.json not found at $PINS — run from a semif-agent checkout" >&2; exit 1; }

WORK="${WORK:-$REPO_ROOT/.runtime/voice-train}"
DATASET="${DATASET:-$WORK/dataset}"
PIPER_DIR="$WORK/piper1-gpl"
VENV="$WORK/venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
CACHE="$WORK/cache"
RUNS="$WORK/runs"
OUT="$WORK/out"
CKPTS="$WORK/checkpoints"
COMPAT="$WORK/torch_compat"

# --- read the voice pins from the committed pins.json -----------------------
IFS=$'\x1f' read -r PIPER_REPO PIPER_REF CKPT_URL CKPT_SHA < <(
  python3 - "$PINS" <<'PY'
import json, sys
pins = json.load(open(sys.argv[1]))
v = pins.get("voice", {})
print("\x1f".join([
    str(v.get("piper_train_repo", "")),
    str(v.get("piper_train_ref", "")),
    str(v.get("base_checkpoint_url", "")),
    str(v.get("base_checkpoint_sha256", "")),
]))
PY
)
[[ -n "$PIPER_REPO" && -n "$PIPER_REF" ]] || {
  echo "pins.json is missing the voice.piper_train_repo/ref entries" >&2; exit 1; }

echo "== voice training: $NAME"
echo "   work:     $WORK"
echo "   dataset:  $DATASET"
echo "   piper:    $PIPER_REPO @ ${PIPER_REF:0:12}"

mkdir -p "$WORK" "$CKPTS" "$OUT" "$RUNS" "$COMPAT"

# PyTorch >=2.6 defaults `torch.load` to weights_only=True, and Lightning passes
# it explicitly, so loading the pinned base checkpoint fails on the
# `pathlib.PosixPath` pickled into its hyperparameters. Allowlist it for the
# training/export interpreter only (via a sitecustomize on PYTHONPATH), which is
# the fix PyTorch's own error recommends. See docs/VOICE-TRAIN.md.
cat > "$COMPAT/sitecustomize.py" <<'PY'
import pathlib

try:
    from torch.serialization import add_safe_globals

    add_safe_globals([pathlib.PosixPath])
except Exception:  # pragma: no cover - best effort; old torch has no such API
    pass
PY

# --- stage 1: dataset check (only needed to actually train) -----------------
if [[ "$SKIP_TRAIN" != "1" ]]; then
  if [[ ! -f "$DATASET/metadata.csv" ]]; then
    echo "no dataset at $DATASET/metadata.csv — record one first:" >&2
    echo "  python scripts/voice-record.py --out '$DATASET'" >&2
    exit 1
  fi
  N_UTT=$(grep -c '|' "$DATASET/metadata.csv" || true)
  echo "== dataset: $N_UTT utterances"
  if [[ "$N_UTT" -lt 50 ]]; then
    echo "WARN: fewer than 50 utterances — the voice will be rough (aim for 200-450+)" >&2
  fi
elif [[ -f "$DATASET/metadata.csv" ]]; then
  echo "== dataset present ($(grep -c '|' "$DATASET/metadata.csv" || true) utterances); --skip-train set"
else
  echo "== no dataset yet; --skip-train set, so this is a setup/GPU smoke test"
fi

# --- stage 2: free the GPU (opt-in) -----------------------------------------
if [[ "$FREE_GPU" == "1" ]]; then
  echo "== freeing the GPU"
  if systemctl list-unit-files 2>/dev/null | grep -q '^winnow\.service'; then
    $SUDO systemctl stop winnow 2>/dev/null || true
  fi
  if command -v ollama >/dev/null 2>&1; then
    ollama ps 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r m; do
      [[ -n "$m" ]] && ollama stop "$m" 2>/dev/null || true
    done
  fi
else
  if systemctl is-active --quiet winnow 2>/dev/null; then
    echo "WARN: winnow.service is active and holds the GPU; pass --free-gpu (or stop it) to avoid OOM" >&2
  fi
fi

# --- stage 3: system prerequisites + piper checkout + venv ------------------
if [[ "$SKIP_SETUP" != "1" ]]; then
  missing=()
  for pkg in espeak-ng build-essential cmake python3-dev python3-venv git; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "== installing system prereqs: ${missing[*]}"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y "${missing[@]}"
  fi

  if [[ ! -d "$PIPER_DIR/.git" ]]; then
    echo "== cloning piper1-gpl"
    git clone "$PIPER_REPO" "$PIPER_DIR"
  fi
  echo "== checking out piper1-gpl @ ${PIPER_REF:0:12}"
  git -C "$PIPER_DIR" fetch --all --tags --quiet
  git -C "$PIPER_DIR" checkout --quiet "$PIPER_REF"

  if [[ ! -x "$PY" ]]; then
    echo "== creating venv at $VENV"
    python3 -m venv "$VENV"
    "$PIP" install --upgrade pip wheel setuptools >/dev/null
  fi

  if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
    echo "== installing torch from $TORCH_INDEX_URL (this may take a while)"
    # shellcheck disable=SC2086
    "$PIP" install --index-url "$TORCH_INDEX_URL" $TORCH_SPEC
  fi
  echo "== installing piper1-gpl [train]"
  "$PIP" install -e "${PIPER_DIR}[train]"

  # The editable install builds the CMake extension `espeakbridge` in its
  # isolated build env but does not expose it from the source tree, so
  # `from piper import espeakbridge` fails at run time. Run the documented
  # in-place dev build to place the compiled extensions under src/.
  if ! "$PY" -c 'from piper import espeakbridge' >/dev/null 2>&1; then
    echo "== building piper1-gpl C extensions in place (espeakbridge)"
    "$PIP" install -q scikit-build cmake ninja "cython>=3,<4"
    ( cd "$PIPER_DIR" && "$PY" setup.py build_ext --inplace )
  fi

  if [[ -x "$PIPER_DIR/build_monotonic_align.sh" ]]; then
    echo "== building monotonic_align"
    ( source "$VENV/bin/activate" && cd "$PIPER_DIR" && ./build_monotonic_align.sh )
  fi

  echo "== torch device check"
  "$PY" - <<'PY'
import torch
print(f"   torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}  "
      f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("   WARN: no GPU visible to torch — training will be CPU-only and very slow.")
    print("         check the ROCm torch wheel matches this GPU (gfx target).")
else:
    print(f"   device: {torch.cuda.get_device_name(0)}")
PY
fi

# --- stage 4: base checkpoint ------------------------------------------------
if [[ -z "$CHECKPOINT" ]]; then
  if [[ -n "$CKPT_URL" ]]; then
    CHECKPOINT="$CKPTS/$(basename "$CKPT_URL")"
    if [[ ! -f "$CHECKPOINT" ]]; then
      echo "== downloading base checkpoint"
      curl -fL --retry 3 -o "$CHECKPOINT" "$CKPT_URL"
    fi
    if [[ -n "$CKPT_SHA" ]]; then
      echo "== verifying checkpoint sha256"
      echo "$CKPT_SHA  $CHECKPOINT" | sha256sum -c -
    fi
  fi
fi
if [[ -n "$CHECKPOINT" ]]; then
  echo "== fine-tuning from: $CHECKPOINT"
else
  echo "== WARN: no base checkpoint; training from scratch (much slower, lower quality)" >&2
fi

# --- stage 5: train ----------------------------------------------------------
if [[ "$SKIP_TRAIN" != "1" ]]; then
  # `--resume` continues a run (Lightning `--ckpt_path`, which also restores the
  # optimizer/loop state). A fresh fine-tune warmstarts the model weights from
  # the base checkpoint with `--model.warmstart_ckpt`: unlike `--ckpt_path` it
  # does not re-parse the base checkpoint's hyperparameters, which is required
  # for the pinned older checkpoints on a newer piper1-gpl (their saved
  # `model.*` config no longer matches the current model signature).
  WARMSTART_CKPT="$CHECKPOINT"
  RESUME_CKPT=""
  if [[ "$RESUME" == "1" ]]; then
    LATEST="$(find "$RUNS" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    [[ -n "$LATEST" ]] && { echo "== resuming from $LATEST"; RESUME_CKPT="$LATEST"; }
  fi
  echo "== training ($MAX_EPOCHS epochs, batch $BATCH_SIZE) — this runs for hours; Ctrl-C is safe"
  FIT_ARGS=(
    --data.voice_name "$NAME"
    --data.csv_path "$DATASET/metadata.csv"
    --data.audio_dir "$DATASET/wavs"
    --data.cache_dir "$CACHE"
    --data.config_path "$OUT/$NAME.onnx.json"
    --data.espeak_voice "$ESPEAK_VOICE"
    --data.batch_size "$BATCH_SIZE"
    # Keep piper's held-out validation (default 0.1) and test (default 5)
    # splits. The default ModelCheckpoint callbacks monitor `val_mel` and
    # `val_mos`; `val_mos` is scored over the *test* split, so zeroing either
    # split leaves the callback without its metric and Lightning aborts the
    # run — and no checkpoint is written for the export step to use.
    --model.sample_rate "$SAMPLE_RATE"
    --trainer.max_epochs "$MAX_EPOCHS"
    --trainer.accelerator gpu
    --trainer.devices 1
    --trainer.default_root_dir "$RUNS"
  )
  if [[ -n "$RESUME_CKPT" ]]; then
    FIT_ARGS+=( --ckpt_path "$RESUME_CKPT" )
  elif [[ -n "$WARMSTART_CKPT" ]]; then
    FIT_ARGS+=( --model.warmstart_ckpt "$WARMSTART_CKPT" )
  fi
  PYTHONPATH="$COMPAT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m piper.train fit "${FIT_ARGS[@]}"
fi

# --- stage 6: export ---------------------------------------------------------
if [[ "$SKIP_EXPORT" != "1" ]]; then
  CKPT="$(find "$RUNS" -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -z "$CKPT" ]]; then
    echo "no training checkpoint found under $RUNS — nothing to export" >&2
    exit 1
  fi
  echo "== exporting $CKPT -> $OUT/$NAME.onnx"
  PYTHONPATH="$COMPAT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m piper.train.export_onnx --checkpoint "$CKPT" --output-file "$OUT/$NAME.onnx"
  if [[ ! -f "$OUT/$NAME.onnx.json" ]]; then
    echo "WARN: $OUT/$NAME.onnx.json missing (config is written during training)" >&2
  fi
fi

echo
echo "== done. voice files:"
ls -la "$OUT" 2>/dev/null || true
cat <<EOF

Next: copy the voice to a gateway host and select it.

  rsync -av '$OUT/$NAME.onnx' '$OUT/$NAME.onnx.json' \\
      <gateway-host>:'<checkout>/.runtime/voice/tts/'

Then set in that host's config.json (gateway.voice.tts.voice_dir is anchored to
.runtime/voice/tts automatically):

  "tts": { "voice": "$NAME", ... }

Restart the gateway and speak a reply to verify — a passing skill test is not
proof the integration works; only a real spoken reply is.
EOF
