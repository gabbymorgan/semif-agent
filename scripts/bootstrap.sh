#!/usr/bin/env bash
# Provision a new Ubuntu machine into a running semif-agent staging box.
#
# Idempotent: safe to rerun; every stage no-ops on already-provisioned state,
# so it can also bootstrap a machine of unknown state. All LLM traffic points
# at a peer ollama endpoint (default: guppy) — this script installs NO ollama.
#
# Must be run from a semif-agent checkout (it reads pins from this checkout's
# config.example.json); the checkout is used as-is and never re-cloned — only
# the SemIf engine is fetched. Clone the agent repo to ~/semif-agent first
# (its SSH key must be registered on Gitea).
#
# Pins (SemIf git commit, GGUF url+sha256, python deps) are read from
# config.example.json's engine block, which is the single source of truth;
# bump those there and rerun to upgrade.
#
# Usage:
#   scripts/bootstrap.sh [--peer-ollama URL] [--threads N] [--copy-data SRC]
#                        [--public-dashboard] [-h]
#
# Run as the human user; sudo is used internally for system bits.
set -euo pipefail

PEER_OLLAMA="http://192.168.8.181:11434"
THREADS=""
COPY_DATA=""
PUBLIC_DASHBOARD=0

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "  --peer-ollama URL     remote ollama API for llm+codegen (default: $PEER_OLLAMA)"
  echo "  --threads N           engine threads for config.json (default: config.example value)"
  echo "  --copy-data SRC       rsync SRC (e.g. abby@box:~/repos/semif-agent/data) to data/ — opt-in"
  echo "  --public-dashboard    bind dashboard to 0.0.0.0 instead of 127.0.0.1"
  echo "  -h                    this help"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --peer-ollama) PEER_OLLAMA="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --copy-data) COPY_DATA="$2"; shift 2 ;;
    --public-dashboard) PUBLIC_DASHBOARD=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="$REPO_ROOT/config.example.json"

if [[ ! -f "$EXAMPLE" ]]; then
  echo "config.example.json not found at $EXAMPLE — run from a semif-agent checkout" >&2
  exit 1
fi

# --- read pins from config.example.json ------------------------------------
read -r SEMIF_REPO SEMIF_REF GGUF_URL GGUF_SHA256 HF_SOURCE HF_REV < <(
  python3 - "$EXAMPLE" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
e = cfg["engine"]
print(
    " ".join(
        [
            e.get("semif_repo", ""),
            e.get("semif_ref", ""),
            e.get("gguf_url", ""),
            e.get("gguf_sha256", ""),
            e.get("source", ""),
            e.get("revision", ""),
        ]
    )
)
PY
)

if [[ -z "$SEMIF_REPO" || -z "$SEMIF_REF" || -z "$GGUF_URL" || -z "$GGUF_SHA256" ]]; then
  echo "engine pins missing in $EXAMPLE (semif_repo/semif_ref/gguf_url/gguf_sha256)" >&2
  exit 1
fi

echo "== pins: semif @ $SEMIF_REF | gguf sha256 ${GGUF_SHA256:0:12}… | peer $PEER_OLLAMA"

# --- stage 0: system prereqs -------------------------------------------------
PKGS=(ca-certificates curl git rsync build-essential python3-dev python3-venv pkg-config cmake)
missing=()
for p in "${PKGS[@]}"; do
  dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "== installing prereqs: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
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

AGENT_DIR="$REPO_ROOT"
git -C "$AGENT_DIR" pull --ff-only || true

if [[ ! -d ~/semif/.git ]]; then
  echo "== cloning SemIf engine from $SEMIF_REPO"
  git clone "$SEMIF_REPO" ~/semif
fi
if [[ "$(git -C ~/semif rev-parse HEAD 2>/dev/null)" != "$SEMIF_REF" ]]; then
  echo "== pinning SemIf engine to $SEMIF_REF"
  git -C ~/semif fetch origin
  git -C ~/semif checkout "$SEMIF_REF"
fi

# --- stage 3: venv + deps -------------------------------------------------------
if [[ ! -d ~/semif-venv ]]; then
  echo "== creating ~/semif-venv"
  "$PY" -m venv ~/semif-venv
fi
PIP=~/semif-venv/bin/pip
echo "== installing engine deps into venv"
"$PIP" install --upgrade pip setuptools wheel
"$PIP" install -e ~/semif --no-deps
"$PIP" install numpy==2.3.5 transformers==5.17.0 tokenizers==0.23.2 huggingface-hub
CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 "$PIP" install llama-cpp-python==0.3.35
"$PIP" install -e "$AGENT_DIR" --no-deps
"$PIP" install pytest

# --- stage 4: GGUF --------------------------------------------------------------
GGUF_DIR=~/models
GGUF_NAME="$(basename "$GGUF_URL")"
GGUF="$GGUF_DIR/$GGUF_NAME"
mkdir -p "$GGUF_DIR"
sha_ok() { [[ -f "$GGUF" ]] && [[ "$(sha256sum "$GGUF" | cut -d' ' -f1)" == "$GGUF_SHA256" ]]; }
if ! sha_ok; then
  echo "== downloading GGUF ($GGUF_NAME)"
  rm -f "$GGUF" "$GGUF.tmp"
  curl -fL --retry 3 -o "$GGUF.tmp" "$GGUF_URL"
  mv "$GGUF.tmp" "$GGUF"
fi
sha_ok || { echo "GGUF sha256 mismatch: expected $GGUF_SHA256" >&2; exit 1; }
echo "== GGUF verified ($GGUF)"

# --- stage 5: HF tokenizer cache --------------------------------------------------
export HF_HOME="$HOME/hf"
mkdir -p "$HF_HOME"
echo "== pre-fetching tokenizer $HF_SOURCE @ $HF_REV"
~/semif-venv/bin/python - "$HF_SOURCE" "$HF_REV" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
PY

# --- stage 6: config.json -----------------------------------------------------------
if [[ -f "$REPO_ROOT/config.json" ]]; then
  cp "$REPO_ROOT/config.json" "$REPO_ROOT/config.json.bak.$(date +%s)"
fi
echo "== writing config.json (llm/codegen -> $PEER_OLLAMA)"
python3 - "$REPO_ROOT/config.example.json" "$REPO_ROOT/config.json" "$THREADS" "$PEER_OLLAMA" "$([ "$PUBLIC_DASHBOARD" = 1 ] && echo 0.0.0.0 || echo 127.0.0.1)" <<'PY'
import json, os, sys
example, out, threads, peer, dash_host = sys.argv[1:]
cfg = json.load(open(example))
eng = cfg["engine"]
eng["gguf"] = os.path.join(os.path.expanduser("~"), "models", os.path.basename(eng["gguf_url"]))
if threads and threads != "__example__":
    eng["threads"] = int(threads)
cfg["llm"]["base_url"] = peer
cfg["codegen"]["base_url"] = peer
cfg["codegen"]["timeout"] = 3600
cfg.setdefault("dashboard", {})["port"] = 8765
cfg["dashboard"]["host"] = dash_host
json.dump(cfg, open(out, "w"), indent=2)
PY

# --- stage 7: optional data copy -----------------------------------------------------
if [[ -n "$COPY_DATA" ]]; then
  echo "== rsync data from $COPY_DATA"
  mkdir -p "$AGENT_DIR/data"
  rsync -a "$COPY_DATA/" "$AGENT_DIR/data/"
fi

# --- stage 8: verify --------------------------------------------------------------------
echo "== verifying imports"
~/semif-venv/bin/python -c "import semif_phase1, semif_agent; print('engine + agent import OK')"
echo "== peer ollama check ($PEER_OLLAMA)"
PEER_TAGS="$(curl -sf --max-time 5 "$PEER_OLLAMA/api/tags" || true)"
if grep -qE "qwen3\.5:4b" <<<"$PEER_TAGS"; then
  echo "peer serves qwen3.5:4b (self-assessment): OK"
else
  echo "WARN: peer /api/tags shows no qwen3.5:4b" >&2
fi
if grep -qE "qwen38-iq3s" <<<"$PEER_TAGS"; then
  echo "peer serves qwen38-iq3s (codegen): OK"
else
  echo "WARN: peer /api/tags shows no qwen38-iq3s" >&2
fi

echo
echo "Done. Next steps:"
echo "  cd ~/semif-agent"
echo "  HF_HOME=~/hf ~/semif-venv/bin/python -m semif_agent.cli run"
echo "  HF_HOME=~/hf ~/semif-venv/bin/python -m pytest tests/integration -q -s  (box needs real engine+LLM)"
echo "  HF_HOME=~/hf ~/semif-venv/bin/python -m semif_agent.cli dashboard --port 8765"