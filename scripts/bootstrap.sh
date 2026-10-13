#!/usr/bin/env bash
# Provision a new Ubuntu machine into a running semif-agent host.
#
# Idempotent: safe to rerun; every stage no-ops on already-provisioned state,
# so it can also bootstrap a machine of unknown state. The small
# title/description model runs on this machine's local ollama (bootstrap pulls
# it); codegen may point at a remote ollama host. This script installs NO ollama.
#
# Must be run from a semif-agent checkout. The checkout is used as-is and never
# re-cloned — only the SemIf engine, the simplex-chat binary, and the skill
# packs are fetched. Clone the agent repo first (its SSH key must be registered
# on your git host); each skill pack is then cloned from a sibling repository
# named pack.<name> on the same host (see packs.json) into the gitignored
# packs/ directory.
#
# config.json is the per-machine USER config (a thin index over config/*.json)
# and this script NEVER rewrites it. It seeds the split config via `config-init`
# (config/core.json, config/gateway.json, config/bridges.json, config/packs/*.json,
# the config.json index, and an empty secrets.json) only when config.json is
# absent, then reads every per-machine value from the merged config; only a human
# edits it afterwards. Pinned external refs (SemIf commit, GGUF url+sha256,
# HF tokenizer revision, simplex-chat version/url+sha256) live in the committed
# pins.json and the python dep pins in requirements/staging.txt — bump those
# (via git) and rerun to upgrade.
#
# Every installation artifact lives INSIDE this checkout under .runtime/
# (gitignored), so an end user can find and debug the whole stack in one tree:
#   .runtime/venv/     python venv            .runtime/models/  GGUF
#   .runtime/engine/   SemIf engine clone     .runtime/hf/      HF cache
#   .runtime/bin/      simplex-chat binary    .runtime/simplex/ simplex profile
#   .runtime/systemd/  rendered unit files (symlinked into ~/.config/systemd/user)
# Only artifacts that operationally must live elsewhere are outside: the SSH
# key (~/.ssh) and the real systemd user dir/linger.
#
# Flags override the values read from the merged config FOR THIS RUN ONLY;
# nothing is written back to the config.
#
# Usage:
#   scripts/bootstrap.sh [--llm-url URL] [--codegen-url URL] [--llm-model MODEL]
#                        [--codegen-model MODEL] [--copy-data SRC] [--voice]
#                        [--browser] [--gpu MODE] [--pack-base URL] [-h]
#
# Run as the human user; sudo is used internally for system bits.
set -euo pipefail

LLM_URL=""
CODEGEN_URL=""
LLM_MODEL=""
CODEGEN_MODEL=""
COPY_DATA=""
VOICE=0
BROWSER=0
GPU_MODE="auto"
PACK_BASE=""

usage() {
  sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
  echo
  echo "  --llm-url URL          override llm.base_url from config.json for this run (else config.json, else http://127.0.0.1:11434)"
  echo "  --codegen-url URL      override codegen.base_url from config.json for this run"
  echo "  --llm-model MODEL      override llm.model from config.json for this run (pulled if the local ollama lacks it)"
  echo "  --codegen-model MODEL  override codegen.model from config.json for this run"
  echo "  --copy-data SRC        rsync SRC (e.g. user@host:/path/to/semif-agent/data) to data/ — opt-in"
  echo "  --voice                install the optional voice-gateway stack + download its models"
  echo "  --browser              install the optional browser-bridge stack (Playwright + Chromium)"
  echo "  --gpu MODE             decision-engine llama.cpp backend: auto (default; a GPU build only when engine.gpu_layers is set), vulkan, rocm, cuda, cpu"
  echo "  --pack-base URL        base URL of the skill-pack repos (default: derived from this checkout's origin remote)"
  echo "  -h                     this help"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --llm-url) LLM_URL="$2"; shift 2 ;;
    --codegen-url) CODEGEN_URL="$2"; shift 2 ;;
    --llm-model) LLM_MODEL="$2"; shift 2 ;;
    --codegen-model) CODEGEN_MODEL="$2"; shift 2 ;;
    --copy-data) COPY_DATA="$2"; shift 2 ;;
    --voice) VOICE=1; shift ;;
    --browser) BROWSER=1; shift ;;
    --gpu) GPU_MODE="$2"; shift 2 ;;
    --pack-base) PACK_BASE="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$REPO_ROOT/config.example.json"
CONFIG="$REPO_ROOT/config.json"
PINS="$REPO_ROOT/pins.json"

RUNTIME="$REPO_ROOT/.runtime"
VENV="$RUNTIME/venv"
ENGINE="$RUNTIME/engine"
MODELS="$RUNTIME/models"
HF_CACHE="$RUNTIME/hf"
TOOLDIR="$RUNTIME/bin"
SIMPLEX_DB="$RUNTIME/simplex"
UNITS="$RUNTIME/systemd"
USER_UNITS="$HOME/.config/systemd/user"
BROWSERS="$RUNTIME/browser"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [[ ! -f "$EXAMPLE" ]]; then
  echo "config.example.json not found at $EXAMPLE — run from a semif-agent checkout" >&2
  exit 1
fi
if [[ ! -f "$PINS" ]]; then
  echo "pins.json not found at $PINS — run from a semif-agent checkout" >&2
  exit 1
fi

# --- skill packs ----------------------------------------------------------------
# Each pack ships in its own repository named pack.<name>, cloned into the
# gitignored packs/ directory; the agent loads every pack under skill_packs
# (default "packs"). The list lives in the committed packs.json; each repo is a
# sibling of this checkout's origin remote (pack.<name>) unless --pack-base
# overrides the base. This runs BEFORE the config is seeded so config-init sees
# the packs and generates their config/packs/*.json fragments. Idempotent:
# clone when absent, fast-forward when present.
PACK_MANIFEST="$REPO_ROOT/packs.json"
if [[ -f "$PACK_MANIFEST" ]]; then
  if [[ -z "$PACK_BASE" ]]; then
    ORIGIN_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
    PACK_BASE="${ORIGIN_URL%/*}"
  fi
  PACK_BASE="${PACK_BASE%/}"
  if [[ -z "$PACK_BASE" ]]; then
    echo "WARN: no 'origin' remote to derive the skill-pack repos from; clone them into packs/ or pass --pack-base URL" >&2
  else
    echo "== skill packs (from $PACK_BASE/pack.<name>)"
    mkdir -p "$REPO_ROOT/packs"
    while IFS= read -r pack_name; do
      [[ -z "$pack_name" ]] && continue
      pack_dst="$REPO_ROOT/packs/$pack_name"
      pack_url="$PACK_BASE/pack.$pack_name.git"
      if [[ -d "$pack_dst/.git" ]]; then
        if git -C "$pack_dst" pull --ff-only --quiet 2>/dev/null; then
          echo "   $pack_name: up to date"
        else
          echo "   WARN: could not update pack $pack_name (leaving the local checkout)" >&2
        fi
      elif [[ -e "$pack_dst" ]]; then
        echo "   WARN: packs/$pack_name exists but is not a git clone; leaving it" >&2
      elif git clone --quiet "$pack_url" "$pack_dst" 2>/dev/null; then
        echo "   $pack_name: cloned"
      else
        echo "   WARN: could not clone $pack_name from $pack_url" >&2
      fi
    done < <(python3 - "$PACK_MANIFEST" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
for name in data.get("packs", []):
    if str(name).strip():
        print(str(name).strip())
PY
)
  fi
fi

# config.json is per-machine USER config. Seed it once (as a thin index plus the
# split config/ fragments + secrets.json), then never write it again — only a
# human edits it. The committed config.example.json is GENERATED from the pack +
# core schemas; see `python -m semif_agent.cli check-config --write-example`.
if [[ ! -f "$CONFIG" ]]; then
  echo "== seeding config/ + secrets.json (edit them to configure this machine)"
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m semif_agent.cli config-init --path "$CONFIG" \
    || { echo "config-init failed; falling back to a flat config.json from the example" >&2; cp "$EXAMPLE" "$CONFIG"; }
fi

# The live config may be split (config.json includes config/*.json); merge it so
# the reads below see one flat object. Stdlib-only, so it works before the venv.
MERGED_CONFIG="$RUNTIME/config.merged.json"
mkdir -p "$RUNTIME"
python3 "$REPO_ROOT/scripts/config_merge.py" "$CONFIG" > "$MERGED_CONFIG" \
  || { echo "could not read $CONFIG" >&2; exit 1; }

# --- read pins from the committed pins.json ---------------------------------
# The ASCII unit separator (\x1f) delimits fields so an empty value never
# shifts the fields after it (space-splitting collapses runs of whitespace).
# The simplex-chat binary is per-arch: normalize `uname -m` to a pins key.
SIMPLEX_ARCH="$(uname -m)"
case "$SIMPLEX_ARCH" in
  arm64|aarch64) SIMPLEX_ARCH="aarch64" ;;
  x86_64|amd64)  SIMPLEX_ARCH="x86_64" ;;
esac
IFS=$'\x1f' read -r SEMIF_REPO SEMIF_REF GGUF_URL GGUF_SHA256 HF_SOURCE HF_REV \
  SIMPLEX_BIN_URL SIMPLEX_SHA256 < <(
  python3 - "$PINS" "$SIMPLEX_ARCH" <<'PY'
import json, sys
pins = json.load(open(sys.argv[1]))
arch = sys.argv[2]
e = pins["engine"]
s = pins.get("simplex_chat", {})
asset = (s.get("assets", {}) or {}).get(arch, {})
print("\x1f".join([
    str(e.get("semif_repo", "")),
    str(e.get("semif_ref", "")),
    str(e.get("gguf_url", "")),
    str(e.get("gguf_sha256", "")),
    str(e.get("source", "")),
    str(e.get("revision", "")),
    str(asset.get("bin_url", "")),
    str(asset.get("sha256", "")),
]))
PY
)

if [[ -z "$SEMIF_REPO" || -z "$SEMIF_REF" || -z "$GGUF_URL" || -z "$GGUF_SHA256" ]]; then
  echo "engine pins missing in $PINS (semif_repo/semif_ref/gguf_url/gguf_sha256)" >&2
  exit 1
fi
if [[ -z "$SIMPLEX_BIN_URL" || -z "$SIMPLEX_SHA256" ]]; then
  echo "simplex_chat pins missing in $PINS for arch '$SIMPLEX_ARCH' (assets.<arch>.bin_url/sha256)" >&2
  exit 1
fi

# --- read per-machine values from config.json (flags override for this run) --
IFS=$'\x1f' read -r CFG_LLM_URL CFG_LLM_MODEL CFG_LLM_PROVIDER CFG_CODEGEN_URL CFG_CODEGEN_MODEL \
  SIMPLEX_PORT SIMPLEX_DISPLAY SIMPLEX_FORWARD_PORT SIMPLEX_FORWARD_DISPLAY < <(
  python3 - "$MERGED_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
s = cfg.get("simplex_chat", {})
def strip_v1(u):
    u = str(u or "").rstrip("/")
    return u[:-3] if u.endswith("/v1") else u
print("\x1f".join([
    strip_v1(cfg.get("llm", {}).get("base_url")),
    str(cfg.get("llm", {}).get("model", "") or ""),
    str(cfg.get("llm", {}).get("provider", "ollama") or "ollama").lower(),
    strip_v1(cfg.get("codegen", {}).get("base_url")),
    str(cfg.get("codegen", {}).get("model", "") or ""),
    str(s.get("port", "")),
    str(s.get("display_name", "") or ""),
    str(s.get("forward_port", "")),
    str(s.get("forward_display_name", "") or ""),
]))
PY
)

# A flag overrides config.json for this run only; config.json is never touched.
LLM_URL="${LLM_URL:-${CFG_LLM_URL:-http://127.0.0.1:11434}}"
CODEGEN_URL="${CODEGEN_URL:-$CFG_CODEGEN_URL}"
LLM_MODEL="${LLM_MODEL:-$CFG_LLM_MODEL}"
CODEGEN_MODEL="${CODEGEN_MODEL:-$CFG_CODEGEN_MODEL}"
LLM_PROVIDER="${LLM_PROVIDER:-${CFG_LLM_PROVIDER:-ollama}}"

if [[ -z "$SIMPLEX_PORT" || -z "$SIMPLEX_FORWARD_PORT" ]]; then
  echo "config.json: simplex_chat.port and simplex_chat.forward_port are required" >&2
  exit 1
fi
SIMPLEX_DISPLAY="${SIMPLEX_DISPLAY:-semif}"
SIMPLEX_FORWARD_DISPLAY="${SIMPLEX_FORWARD_DISPLAY:-semif-forward}"

# Native ollama API bases (no /v1) for tags/pull; the clients use /v1.
LLM_API="${LLM_URL%/v1}"; LLM_API="${LLM_API%/}"
CODEGEN_API="${CODEGEN_URL%/v1}"; CODEGEN_API="${CODEGEN_API%/}"
with_v1() { local u="${1%/}"; [[ "$u" == */v1 ]] && printf '%s' "$u" || printf '%s/v1' "$u"; }
LLM_V1="$(with_v1 "$LLM_URL")"

echo "== pins: semif @ $SEMIF_REF | gguf sha256 ${GGUF_SHA256:0:12}… | simplex-chat sha256 ${SIMPLEX_SHA256:0:12}…"
if [[ "$LLM_PROVIDER" == "semif" ]]; then
  echo "== llm semif (in-process engine model) | codegen $CODEGEN_URL"
else
  echo "== llm $LLM_URL (local, model $LLM_MODEL) | codegen $CODEGEN_URL"
fi
echo "== runtime tree: $RUNTIME"

mkdir -p "$RUNTIME" "$MODELS" "$HF_CACHE" "$TOOLDIR" "$SIMPLEX_DB" "$RUNTIME/simplex-forward" "$RUNTIME/lxmf" "$RUNTIME/voice" "$UNITS" "$USER_UNITS"

# --- stage 0: system prereqs + linger ----------------------------------------
PKGS=(ca-certificates curl git rsync build-essential python3-dev python3-venv pkg-config cmake util-linux)
missing=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "== installing prereqs: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
fi

if command -v loginctl >/dev/null 2>&1; then
  if [[ "$(loginctl show-user "$USER" 2>/dev/null | sed -n 's/^Linger=//p')" != "yes" ]]; then
    echo "== enabling linger for $USER (services survive logout/reboot)"
    sudo loginctl enable-linger "$USER"
  fi
fi

# --- stage 1: python ----------------------------------------------------------
PY=python3
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
  echo "== python3 < 3.11; installing python3.13 from deadsnakes"
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
  PY=python3.13
fi
echo "== python: $("$PY" --version)"

# --- stage 2: ssh key + SemIf engine clone ------------------------------------
mkdir -p ~/.ssh
if [[ ! -f ~/.ssh/id_ed25519 ]]; then
  echo "== generating ~/.ssh/id_ed25519 (register this key on Gitea for the agent repo)"
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
  cat ~/.ssh/id_ed25519.pub
fi

git -C "$REPO_ROOT" pull --ff-only || true

if [[ ! -d "$ENGINE/.git" ]]; then
  echo "== cloning SemIf engine from $SEMIF_REPO into $ENGINE"
  git clone "$SEMIF_REPO" "$ENGINE"
fi
if [[ "$(git -C "$ENGINE" rev-parse HEAD 2>/dev/null)" != "$SEMIF_REF" ]]; then
  echo "== pinning SemIf engine to $SEMIF_REF"
  git -C "$ENGINE" fetch origin
  git -C "$ENGINE" checkout "$SEMIF_REF"
fi

# --- stage 2b: pick the llama.cpp GPU backend ---------------------------------
# The decision engine scores the GGUF through llama.cpp, which runs on CPU
# unless llama-cpp-python was built with a GPU backend. A GPU-enabled build is
# opt-in: llama.cpp's Vulkan backend splits the compute graph across CPU and
# Vulkan even with `n_gpu_layers=0`, roughly doubling CPU latency, so a CPU-only
# host must not get one. `--gpu <backend>` forces a build; `--gpu auto` (the
# default) builds a GPU backend only when config.json asks for offload
# (`engine.gpu_layers` nonzero).
CONFIG_GPU_LAYERS="$(python3 - "$MERGED_CONFIG" <<'PY'
import json, sys
try:
    print(int(json.load(open(sys.argv[1])).get("engine", {}).get("gpu_layers", 0) or 0))
except Exception:
    print(0)
PY
)"
detect_gpu_backend() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo cuda; return
  fi
  if [[ -d /opt/rocm ]]; then
    echo rocm; return
  fi
  if command -v glslc >/dev/null 2>&1 && compgen -G "/usr/share/vulkan/icd.d/*.json" >/dev/null; then
    echo vulkan; return
  fi
  echo cpu
}
case "$GPU_MODE" in
  cpu) GPU_BACKEND=cpu ;;
  vulkan|rocm|cuda) GPU_BACKEND="$GPU_MODE" ;;
  auto)
    if [[ "$CONFIG_GPU_LAYERS" != 0 ]]; then
      GPU_BACKEND="$(detect_gpu_backend)"
    else
      GPU_BACKEND=cpu
    fi
    ;;
  *) echo "unknown --gpu mode: $GPU_MODE (want auto, vulkan, rocm, cuda, cpu)" >&2; exit 2 ;;
esac
case "$GPU_BACKEND" in
  vulkan) LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on" ;;
  rocm)   LLAMA_CMAKE_ARGS="-DGGML_HIP=on" ;;
  cuda)   LLAMA_CMAKE_ARGS="-DGGML_CUDA=on" ;;
  *)      LLAMA_CMAKE_ARGS="" ;;
esac
echo "== decision-engine llama.cpp backend: $GPU_BACKEND (engine.gpu_layers=$CONFIG_GPU_LAYERS)"

if [[ "$GPU_BACKEND" = vulkan ]]; then
  VULKAN_PKGS=(libvulkan-dev glslc spirv-headers glslang-tools)
  vk_missing=()
  for p in "${VULKAN_PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || vk_missing+=("$p"); done
  if [[ ${#vk_missing[@]} -gt 0 ]]; then
    echo "== installing Vulkan build prereqs: ${vk_missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${vk_missing[@]}"
  fi
fi

# --- stage 3: venv + deps -------------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
  echo "== creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
echo "== installing engine deps into $VENV"
"$PIP" install --upgrade pip setuptools wheel
"$PIP" install -e "$ENGINE" --no-deps
CMAKE_ARGS="$LLAMA_CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 \
  "$PIP" install -r "$REPO_ROOT/requirements/staging.txt"

# Make the installed llama-cpp-python match the requested backend: pip ships a
# CPU wheel and its wheel cache hands it back even when CMAKE_ARGS asks for a
# GPU backend, so force a source build whenever offload support disagrees.
installed_gpu=no
"$PYTHON" -c 'import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)' 2>/dev/null && installed_gpu=yes
want_gpu=yes
[[ "$GPU_BACKEND" = cpu ]] && want_gpu=no
if [[ "$installed_gpu" != "$want_gpu" ]]; then
  LLAMA_PIN="$(grep -m1 -E '^llama-cpp-python==' "$REPO_ROOT/requirements/staging.txt")"
  echo "== rebuilding llama-cpp-python for the $GPU_BACKEND backend ($LLAMA_PIN)"
  CMAKE_ARGS="$LLAMA_CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 \
    "$PIP" install --no-deps --force-reinstall --no-cache-dir --no-binary llama-cpp-python "$LLAMA_PIN"
fi
"$PIP" install -e "$REPO_ROOT" --no-deps

# --- stage 3b: optional voice-gateway stack (--voice) --------------------------
# Hardware-dependent: only the host with a microphone/speaker needs it. The
# stack is lazy-imported, so a host without it still runs every other front end.
if [[ "$VOICE" = 1 ]]; then
  # libportaudio2 is the I/O backend sounddevice needs; alsa-utils provides
  # amixer/alsamixer/alsactl so the operator can unmute and set the mic/speaker
  # levels (PortAudio itself has no volume control).
  VOICE_SYS_PKGS=(libportaudio2 alsa-utils)
  voice_missing=()
  for p in "${VOICE_SYS_PKGS[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || voice_missing+=("$p")
  done
  if [[ ${#voice_missing[@]} -gt 0 ]]; then
    echo "== installing voice system prereqs: ${voice_missing[*]}"
    sudo apt-get update
    sudo apt-get install -y "${voice_missing[@]}"
  fi
  echo "== installing voice deps into $VENV"
  "$PIP" install -r "$REPO_ROOT/requirements/voice.txt"
  # openWakeWord's Linux metadata hard-requires tflite-runtime (no py3.12+
  # wheel); the onnxruntime backend does not need it, so install it --no-deps
  # (its own runtime deps are pinned in voice.txt).
  "$PIP" install --no-deps openwakeword==0.6.0
fi

# --- stage 3c: optional browser-bridge stack (--browser) -----------------------
# Playwright + its Chromium binaries. Optional: lazy-imported, so a host without
# it still runs every other bridge. Browsers go under the checkout's
# .runtime/browser (the containment rule); the bridge sets PLAYWRIGHT_BROWSERS_PATH
# to the same path at launch.
if [[ "$BROWSER" = 1 ]]; then
  echo "== installing browser-bridge deps into $VENV"
  "$PIP" install -r "$REPO_ROOT/requirements/browser.txt"
  BROWSER_ENGINE="$("$PYTHON" - "$MERGED_CONFIG" <<'PY' 2>/dev/null || true
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    print("chromium"); raise SystemExit(0)
print(cfg.get("bridges", {}).get("browser", {}).get("engine", "chromium") or "chromium")
PY
)"
  case "$BROWSER_ENGINE" in chromium|firefox|webkit) ;; *) BROWSER_ENGINE=chromium ;; esac
  echo "== downloading Playwright browser '$BROWSER_ENGINE' into $BROWSERS"
  if ! PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "$PYTHON" -m playwright install "$BROWSER_ENGINE"; then
    echo "WARN: 'playwright install' failed" >&2
    if [[ "$BROWSER_ENGINE" = chromium ]]; then
      # Playwright's own downloader prefers IPv6 (autoSelectFamily + a custom
      # dual-stack resolver) and can hang on a host that has an IPv6 address but
      # no working global IPv6 route, even though the CDN serves over IPv4. The
      # helper fetches the same pinned artifacts over IPv4 and lays them out
      # where Playwright expects them, so the install becomes a no-op.
      echo "== retrying over IPv4 (packs/browser/scripts/playwright-browsers.py)" >&2
      PLAYWRIGHT_BROWSERS_PATH="$BROWSERS" "$PYTHON" "$REPO_ROOT/packs/browser/scripts/playwright-browsers.py" \
        --browsers-path "$BROWSERS" --engine "$BROWSER_ENGINE" \
        || echo "WARN: browser priming failed; re-run: $PYTHON packs/browser/scripts/playwright-browsers.py --browsers-path $BROWSERS" >&2
    else
      echo "WARN: re-run: PLAYWRIGHT_BROWSERS_PATH=$BROWSERS $PYTHON -m playwright install $BROWSER_ENGINE" >&2
    fi
  fi
fi

# --- stage 4: GGUF --------------------------------------------------------------
GGUF_NAME="$(basename "$GGUF_URL")"
GGUF="$MODELS/$GGUF_NAME"
sha_ok() { [[ -f "$GGUF" ]] && [[ "$(sha256sum "$GGUF" | cut -d' ' -f1)" == "$GGUF_SHA256" ]]; }
if ! sha_ok; then
  echo "== downloading GGUF ($GGUF_NAME) into $MODELS"
  rm -f "$GGUF" "$GGUF.tmp"
  curl -fL --retry 3 -o "$GGUF.tmp" "$GGUF_URL"
  mv "$GGUF.tmp" "$GGUF"
fi
sha_ok || { echo "GGUF sha256 mismatch: expected $GGUF_SHA256" >&2; exit 1; }
echo "== GGUF verified ($GGUF)"

# --- stage 5: HF tokenizer cache --------------------------------------------------
export HF_HOME="$HF_CACHE"
mkdir -p "$HF_CACHE"
echo "== pre-fetching tokenizer $HF_SOURCE @ $HF_REV into $HF_CACHE"
"$PYTHON" - "$HF_SOURCE" "$HF_REV" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
PY

# --- stage 5b: optional voice models (--voice) ----------------------------------
# Downloads the openWakeWord models (into the venv) + the Piper voice (into
# .runtime/voice/tts) and warms the faster-whisper model into the HF cache.
# Reads gateway.voice from config.json
# so the models match the configured names. Best-effort: a download failure
# warns (the gateway reports the missing model at startup) rather than aborting.
if [[ "$VOICE" = 1 ]]; then
  echo "== downloading voice models (openWakeWord + Piper + faster-whisper)"
  HF_HOME="$HF_CACHE" "$PYTHON" "$REPO_ROOT/scripts/voice-models.py" --config "$MERGED_CONFIG" \
    || echo "WARN: voice model download failed; re-run scripts/voice-models.py" >&2
fi

# --- stage 6: simplex-chat daemon binary -------------------------------------------
SIMPLEX_BIN="$TOOLDIR/simplex-chat"
simplex_ok() {
  [[ -f "$SIMPLEX_BIN" ]] && [[ "$(sha256sum "$SIMPLEX_BIN" | cut -d' ' -f1)" == "$SIMPLEX_SHA256" ]]
}
if ! simplex_ok; then
  echo "== downloading simplex-chat into $SIMPLEX_BIN"
  rm -f "$SIMPLEX_BIN" "$SIMPLEX_BIN.tmp"
  curl -fL --retry 3 -o "$SIMPLEX_BIN.tmp" "$SIMPLEX_BIN_URL"
  mv "$SIMPLEX_BIN.tmp" "$SIMPLEX_BIN"
  chmod +x "$SIMPLEX_BIN"
fi
simplex_ok || { echo "simplex-chat sha256 mismatch: expected $SIMPLEX_SHA256" >&2; exit 1; }
echo "== simplex-chat verified ($SIMPLEX_BIN)"

# config.json is USER config: seeded above if absent, read for per-machine values,
# and never written by this script. Edit config.json to change endpoints, models,
# ports, the dashboard bind, or the gateway allowlist.

# --- stage 8: systemd user units -----------------------------------------------------
render_unit() {
  local src="$1" dst="$2"
  sed -e "s|@REPO@|$REPO_ROOT|g" \
      -e "s|@PORT@|$SIMPLEX_PORT|g" \
      -e "s|@DISPLAY_NAME@|$SIMPLEX_DISPLAY|g" \
      -e "s|@FORWARD_PORT@|$SIMPLEX_FORWARD_PORT|g" \
      -e "s|@FORWARD_DISPLAY@|$SIMPLEX_FORWARD_DISPLAY|g" "$src" > "$dst"
}
echo "== rendering systemd user units into $UNITS"
UNIT_CHANGED=0
ACTIVE_BEFORE=""
for unit in semif-simplex.service semif-simplex-forward.service semif-gateway.service semif-bridge.service; do
  tmp="$UNITS/$unit.new"
  render_unit "$REPO_ROOT/scripts/systemd/$unit.in" "$tmp"
  if [[ ! -f "$UNITS/$unit" ]] || ! cmp -s "$tmp" "$UNITS/$unit"; then
    mv "$tmp" "$UNITS/$unit"
    UNIT_CHANGED=1
  else
    rm -f "$tmp"
  fi
  ln -sf "$UNITS/$unit" "$USER_UNITS/$unit"
  if command -v systemctl >/dev/null 2>&1; then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    systemctl --user is-active --quiet "$unit" 2>/dev/null && ACTIVE_BEFORE="$ACTIVE_BEFORE $unit"
  fi
done

if command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if systemctl --user daemon-reload 2>/dev/null; then
    systemctl --user enable --now semif-simplex.service semif-simplex-forward.service semif-gateway.service semif-bridge.service 2>/dev/null \
      && echo "== enabled semif-simplex.service semif-simplex-forward.service semif-gateway.service semif-bridge.service" \
      || echo "WARN: could not enable user services (run: systemctl --user enable --now semif-simplex semif-simplex-forward semif-gateway semif-bridge)" >&2
    # Apply a changed unit to services that were already running (enable --now
    # leaves active units untouched). A daemon restart is safe: the simplex
    # profile persists and the gateway reconnects.
    if [[ "$UNIT_CHANGED" = 1 && -n "$ACTIVE_BEFORE" ]]; then
      for unit in $ACTIVE_BEFORE; do
        systemctl --user restart "$unit" 2>/dev/null \
          && echo "== restarted $unit (unit changed)" \
          || echo "WARN: could not restart $unit" >&2
      done
    fi
  else
    echo "WARN: systemctl --user unavailable; units are at $USER_UNITS" >&2
  fi
fi

# --- stage 8b: SimpleX bot contact address -------------------------------------------
SIMPLEX_ADDRESS=""
if command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if systemctl --user is-active --quiet semif-simplex.service 2>/dev/null; then
    echo "== ensuring the SimpleX bot contact address exists"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if SIMPLEX_ADDRESS="$("$PYTHON" "$REPO_ROOT/packs/simplex/scripts/simplex-address.py" 2>/dev/null)"; then
        printf '%s\n' "$SIMPLEX_ADDRESS" | sed 's/^/  /'
        break
      fi
      sleep 2
    done
    if [[ -z "$SIMPLEX_ADDRESS" ]]; then
      echo "WARN: bot address not ready; run: $PYTHON packs/simplex/scripts/simplex-address.py" >&2
    fi
  fi
fi

# --- stage 8c: forwarding bridge contact address --------------------------------------
# The forwarding bridge owns a second daemon/profile; its own contact link is the
# one `simplex.connect_link` shows. Print it too so the user can share it.
BRIDGE_FORWARD_WS="$("$PYTHON" - "$MERGED_CONFIG" <<'PY' 2>/dev/null || true
import json, sys
try:
    cfg = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
print(cfg.get("bridges", {}).get("simplex", {}).get("ws_url", "") or "")
PY
)"
if [[ -n "$BRIDGE_FORWARD_WS" ]] && command -v systemctl >/dev/null 2>&1; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if systemctl --user is-active --quiet semif-simplex-forward.service 2>/dev/null; then
    echo "== ensuring the forwarding bridge contact address exists"
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if FORWARD_ADDRESS="$("$PYTHON" "$REPO_ROOT/packs/simplex/scripts/simplex-address.py" --ws-url "$BRIDGE_FORWARD_WS" 2>/dev/null)"; then
        printf '%s\n' "$FORWARD_ADDRESS" | sed 's/^/  /'
        break
      fi
      sleep 2
    done
  fi
fi

# --- stage 9: optional data copy -----------------------------------------------------
if [[ -n "$COPY_DATA" ]]; then
  echo "== rsync data from $COPY_DATA"
  mkdir -p "$REPO_ROOT/data"
  rsync -a "$COPY_DATA/" "$REPO_ROOT/data/"
fi

# --- stage 10: verify + pull the local self-assessment model -------------------------
echo "== verifying imports"
"$PYTHON" -c "import semif_phase1, semif_agent; print('engine + agent import OK')"

# The small self-assessment model runs on this machine's local ollama; ensure it
# is pulled so assess/fidelity work without a remote dependency.
# With llm.provider == "semif" the loaded decision engine generates the text
# in-process, so no ollama model/endpoint is involved.
if [[ "$LLM_PROVIDER" == "semif" ]]; then
  echo "== llm provider semif: the loaded engine model generates text in-process (no ollama pull/probe)"
elif [[ -z "$LLM_MODEL" ]]; then
  echo "WARN: llm.model is not set in config.json — set it (and codegen.model) before starting the agent" >&2
else
  echo "== llm ollama ($LLM_API), model $LLM_MODEL"
  if ! curl -sf --max-time 5 "$LLM_API/api/tags" >/dev/null 2>&1; then
    echo "WARN: no local ollama at $LLM_API — install/run ollama and pull $LLM_MODEL" >&2
    echo "      (bootstrap installs no ollama; only codegen is remote)" >&2
  else
    LLM_TAGS="$(curl -sf --max-time 5 "$LLM_API/api/tags" || true)"
    if grep -qF "\"$LLM_MODEL\"" <<<"$LLM_TAGS"; then
      echo "llm model $LLM_MODEL: present"
    else
      echo "== pulling llm model $LLM_MODEL from $LLM_API"
      if ! curl -sf --max-time 3600 "$LLM_API/api/pull" -H 'Content-Type: application/json' \
           -d "{\"model\":\"$LLM_MODEL\",\"stream\":false}" >/dev/null; then
        echo "WARN: /api/pull failed; trying the ollama CLI" >&2
        if command -v ollama >/dev/null 2>&1; then
          ollama pull "$LLM_MODEL" || echo "WARN: could not pull $LLM_MODEL" >&2
        else
          echo "WARN: could not pull $LLM_MODEL (no ollama CLI)" >&2
        fi
      fi
    fi
    # The OpenAI-compat chat path is what llm actually hits; probe it explicitly.
    LLM_V1_OK="$(curl -sf --max-time 60 "$LLM_V1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLM_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false,\"options\":{\"num_predict\":1}}" || true)"
    if grep -qE 'chatcmpl|"choices"' <<<"$LLM_V1_OK"; then
      echo "llm /v1/chat/completions ($LLM_MODEL): OK"
    else
      echo "WARN: local llm /v1/chat/completions returned no completion" >&2
    fi
  fi
fi

# Codegen is the larger model, possibly on a remote host. Warn only — never auto-pull.
if [[ -n "$CODEGEN_URL" && -n "$CODEGEN_MODEL" ]]; then
  echo "== codegen ollama ($CODEGEN_API)"
  CODEGEN_TAGS="$(curl -sf --max-time 5 "$CODEGEN_API/api/tags" || true)"
  if grep -qF "$CODEGEN_MODEL" <<<"$CODEGEN_TAGS"; then
    echo "codegen serves $CODEGEN_MODEL: OK"
  else
    echo "WARN: codegen host $CODEGEN_API serves no $CODEGEN_MODEL (offline? new-skill authoring will fail)" >&2
  fi
else
  echo "WARN: codegen.base_url/model is not set in config.json — set it before starting the agent" >&2
fi

if command -v systemctl >/dev/null 2>&1; then
  echo "== gateway + bridge services"
  systemctl --user is-active semif-simplex.service 2>/dev/null | sed 's/^/  semif-simplex: /' || true
  systemctl --user is-active semif-gateway.service 2>/dev/null | sed 's/^/  semif-gateway: /' || true
  systemctl --user is-active semif-simplex-forward.service 2>/dev/null | sed 's/^/  semif-simplex-forward: /' || true
  systemctl --user is-active semif-bridge.service 2>/dev/null | sed 's/^/  semif-bridge: /' || true
fi

cat <<EOF

Done. Next steps:
  1. Edit the split config (config/core.json, config/gateway.json,
     config/packs/*.json) — set llm.model/llm.base_url and
     codegen.model/codegen.base_url; put credentials in secrets.json.
     bootstrap.sh seeded the config via config-init (if it was absent)
     and NEVER rewrites it; only you edit it.
  2. Skill packs are cloned from their own repos into packs/ (see packs.json);
     they are gitignored here. Update them with:
       for d in packs/*/; do git -C "\$d" pull --ff-only; done
  cd "$REPO_ROOT"
  HF_HOME="$HF_CACHE" "$PYTHON" -m semif_agent.cli run
  HF_HOME="$HF_CACHE" "$PYTHON" -m pytest tests/integration -q -s  (needs real engine+LLM)
  HF_HOME="$HF_CACHE" "$PYTHON" -m semif_agent.cli dashboard --port 8765

SimpleX command gateway (commands only):
  - Runs as systemd user services 'semif-simplex' (bot daemon, port $SIMPLEX_PORT) and
    'semif-gateway' (agent). Check them with:
      systemctl --user status semif-simplex semif-gateway
      journalctl --user -u semif-gateway -f
  - Its user contact address is printed above; re-print (creating if needed) with:
      "$PYTHON" packs/simplex/scripts/simplex-address.py
    Add that address as a contact in your SimpleX app.
  - Then put your contactId/display name in config.json
    gateway.simplex.allowed_users (discover the id from a 'gateway_denied' trace
    event or the daemon's /contacts) and reply. With an empty allowlist the
    gateway rejects everyone — that is the safe default.
  - The gateway takes commands and replies. It never reads history, shows invite
    links, or composes messages: that is the forwarding bridge's job.

LXMF command gateway (commands only):
  - Runs IN-PROCESS inside the same 'semif-gateway' service (no separate daemon);
    the transport is installed from requirements/staging.txt (lxmf + rns).
    Enable it in config.json with gateway.lxmf.enabled = true, then restart the
    gateway (the unit runs `gateway --platform all`, so SimpleX + LXMF share one
    scheduler).
  - Its LXMF address (32 hex chars) is printed on connect:
      journalctl --user -u semif-gateway -f
    Add that address as a contact in an LXMF client (Sideband, MeshChatX, ...)
    and put it in gateway.lxmf.allowed_users (discover a sender from a
    'gateway_denied' trace event). With an empty allowlist it rejects everyone —
    the safe default.

SimpleX forwarding bridge (messaging UX for skills):
  - Runs as 'semif-simplex-forward' (a second bot daemon/profile, port $SIMPLEX_FORWARD_PORT)
    and 'semif-bridge' (the standalone bridge API). Its contact address is printed
    above; re-print with:
      "$PYTHON" packs/simplex/scripts/simplex-address.py --ws-url ws://127.0.0.1:$SIMPLEX_FORWARD_PORT
  - Skills reach it over HTTP at the address derived from bridges.simplex
    (host/port, default http://127.0.0.1:5227); simplex.connect_link shows its
    address and simplex.next_message reads messages sent to it.

LLM bridge (generation for skills):
  - Runs inside the same 'semif-bridge' process. It reuses the top-level `llm`
    model (configured once) and serves a generic POST /chat; skills call it at
    the address derived from bridges.llm (default http://127.0.0.1:5229).

Nextcloud bridge (files/calendar/tasks/contacts/notes for skills):
  - Runs inside the same 'semif-bridge' process (Nextcloud is plain HTTPS, so
    there is no separate daemon). Configure the account once in config.json under
    bridges.nextcloud (url/username/app_password) — or reuse the top-level
    nextcloud_* values the nextcloud pack skills collect. Skills call it at the address
    derived from bridges.nextcloud (default http://127.0.0.1:5230).

Browser bridge (browser automation for skills):
  - Disabled by default; it needs the optional Playwright stack. Enable it with
    bridges.browser.enabled = true (install the deps with scripts/bootstrap.sh
    --browser). It runs inside the same 'semif-bridge' process and keeps a
    persistent browser profile under .runtime/browser/profile. Skills call it at
    the address derived from bridges.browser (default http://127.0.0.1:5232).
EOF

if [[ "$VOICE" = 1 ]]; then
  cat <<EOF

Voice command gateway (wake word + speech-to-text + text-to-speech):
  - The stack is installed and its models downloaded (openWakeWord into the
    venv, the Piper voice into .runtime/voice/tts, faster-whisper into .runtime/hf).
    Enable it in config.json with gateway.voice.enabled = true, then run it in
    the foreground (it needs your audio session; it is NOT a systemd unit):
      HF_HOME="$HF_CACHE" "$PYTHON" -m semif_agent.cli gateway --platform voice
  - Say the wake word (gateway.voice.wake.model, default 'hey_computer' — phrase
    "Hey Computer") and speak; the reply is spoken back. The wake word is
    required before every command (no post-reply follow-up listening).
  - If the mic or speaker is silent/quiet, set the ALSA mixer (PortAudio has no
    volume control) and persist it:
      alsamixer                 # or: amixer -c <card> sset 'Speaker' unmute
      sudo alsactl store
    List devices with: $PYTHON -c "import sounddevice as sd; print(sd.query_devices())"
    and put the input/output device indices in gateway.voice.audio.
  - Re-download / switch models any time with:
      HF_HOME="$HF_CACHE" "$PYTHON" scripts/voice-models.py --config config.json
EOF
fi

if [[ "$BROWSER" = 1 ]]; then
  cat <<EOF

Browser bridge (browser automation for skills):
  - The Playwright stack is installed and the '$BROWSER_ENGINE' browser is under
    $BROWSERS. Enable it in config.json with bridges.browser.enabled = true, then
    restart the bridge service:
      systemctl --user restart semif-bridge.service
  - It runs inside 'semif-bridge' (no separate daemon) and keeps a persistent
    profile at .runtime/browser/profile, so a login you do once survives across
    calls and restarts. Set bridges.browser.headless = false to watch the window
    (useful the first time you sign in to a site); POST /reset with purge=true
    wipes the stored profile.
  - Skills call it at the address derived from bridges.browser (host/port,
    default http://127.0.0.1:5232); the shipped pack skills are browser.open_page,
    browser.read_page, and browser.screenshot.
  - If the bridge reports 503, the browser failed to launch — run:
      PLAYWRIGHT_BROWSERS_PATH=$BROWSERS "$PYTHON" -m playwright install $BROWSER_ENGINE
EOF
fi
