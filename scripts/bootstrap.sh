#!/usr/bin/env bash
# Provision a new Ubuntu machine into a running semif-agent host.
#
# Idempotent: safe to rerun; every stage no-ops on already-provisioned state,
# so it can also bootstrap a machine of unknown state. The small
# title/description model runs on this machine's local ollama (bootstrap pulls
# it); codegen may point at a remote ollama host. This script installs NO ollama.
#
# Must be run from a semif-agent checkout. The checkout is used as-is and never
# re-cloned — only the SemIf engine and the simplex-chat binary are fetched.
# Clone the agent repo first (its SSH key must be registered on your git host).
#
# config.json is the per-machine USER config and this script NEVER writes it. It
# seeds config.json from config.example.json only when config.json is absent,
# then reads every per-machine value from config.json; only a human edits
# config.json afterwards. Pinned external refs (SemIf commit, GGUF url+sha256,
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
# Flags override the values read from config.json FOR THIS RUN ONLY; nothing is
# written back to config.json.
#
# Usage:
#   scripts/bootstrap.sh [--llm-url URL] [--codegen-url URL] [--llm-model MODEL]
#                        [--codegen-model MODEL] [--copy-data SRC] [-h]
#
# Run as the human user; sudo is used internally for system bits.
set -euo pipefail

LLM_URL=""
CODEGEN_URL=""
LLM_MODEL=""
CODEGEN_MODEL=""
COPY_DATA=""

usage() {
  sed -n '2,35p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "  --llm-url URL          override llm.base_url from config.json for this run (else config.json, else http://127.0.0.1:11434)"
  echo "  --codegen-url URL      override codegen.base_url from config.json for this run"
  echo "  --llm-model MODEL      override llm.model from config.json for this run (pulled if the local ollama lacks it)"
  echo "  --codegen-model MODEL  override codegen.model from config.json for this run"
  echo "  --copy-data SRC        rsync SRC (e.g. user@host:/path/to/semif-agent/data) to data/ — opt-in"
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

# config.json is per-machine USER config. Seed it once from the template, then
# never write it again — only a human edits config.json.
if [[ ! -f "$CONFIG" ]]; then
  echo "== seeding config.json from config.example.json (edit it to configure this machine)"
  cp "$EXAMPLE" "$CONFIG"
fi

# --- read pins from the committed pins.json ---------------------------------
# The ASCII unit separator (\x1f) delimits fields so an empty value never
# shifts the fields after it (space-splitting collapses runs of whitespace).
IFS=$'\x1f' read -r SEMIF_REPO SEMIF_REF GGUF_URL GGUF_SHA256 HF_SOURCE HF_REV \
  SIMPLEX_BIN_URL SIMPLEX_SHA256 < <(
  python3 - "$PINS" <<'PY'
import json, sys
pins = json.load(open(sys.argv[1]))
e = pins["engine"]
s = pins.get("simplex_chat", {})
print("\x1f".join([
    str(e.get("semif_repo", "")),
    str(e.get("semif_ref", "")),
    str(e.get("gguf_url", "")),
    str(e.get("gguf_sha256", "")),
    str(e.get("source", "")),
    str(e.get("revision", "")),
    str(s.get("bin_url", "")),
    str(s.get("sha256", "")),
]))
PY
)

if [[ -z "$SEMIF_REPO" || -z "$SEMIF_REF" || -z "$GGUF_URL" || -z "$GGUF_SHA256" ]]; then
  echo "engine pins missing in $PINS (semif_repo/semif_ref/gguf_url/gguf_sha256)" >&2
  exit 1
fi
if [[ -z "$SIMPLEX_BIN_URL" || -z "$SIMPLEX_SHA256" ]]; then
  echo "simplex_chat pins missing in $PINS (bin_url/sha256)" >&2
  exit 1
fi

# --- read per-machine values from config.json (flags override for this run) --
IFS=$'\x1f' read -r CFG_LLM_URL CFG_LLM_MODEL CFG_CODEGEN_URL CFG_CODEGEN_MODEL \
  SIMPLEX_PORT SIMPLEX_DISPLAY SIMPLEX_FORWARD_PORT SIMPLEX_FORWARD_DISPLAY < <(
  python3 - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
s = cfg.get("simplex_chat", {})
def strip_v1(u):
    u = str(u or "").rstrip("/")
    return u[:-3] if u.endswith("/v1") else u
print("\x1f".join([
    strip_v1(cfg.get("llm", {}).get("base_url")),
    str(cfg.get("llm", {}).get("model", "") or ""),
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
echo "== llm $LLM_URL (local, model $LLM_MODEL) | codegen $CODEGEN_URL"
echo "== runtime tree: $RUNTIME"

mkdir -p "$RUNTIME" "$MODELS" "$HF_CACHE" "$TOOLDIR" "$SIMPLEX_DB" "$RUNTIME/simplex-forward" "$UNITS" "$USER_UNITS"

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

# --- stage 3: venv + deps -------------------------------------------------------
if [[ ! -x "$PYTHON" ]]; then
  echo "== creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
echo "== installing engine deps into $VENV"
"$PIP" install --upgrade pip setuptools wheel
"$PIP" install -e "$ENGINE" --no-deps
CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 "$PIP" install -r "$REPO_ROOT/requirements/staging.txt"
"$PIP" install -e "$REPO_ROOT" --no-deps

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
      if SIMPLEX_ADDRESS="$("$PYTHON" "$REPO_ROOT/scripts/simplex-address.py" 2>/dev/null)"; then
        printf '%s\n' "$SIMPLEX_ADDRESS" | sed 's/^/  /'
        break
      fi
      sleep 2
    done
    if [[ -z "$SIMPLEX_ADDRESS" ]]; then
      echo "WARN: bot address not ready; run: $PYTHON scripts/simplex-address.py" >&2
    fi
  fi
fi

# --- stage 8c: forwarding bridge contact address --------------------------------------
# The forwarding bridge owns a second daemon/profile; its own contact link is the
# one `simplex.connect_link` shows. Print it too so the user can share it.
BRIDGE_FORWARD_WS="$("$PYTHON" - "$REPO_ROOT/config.json" <<'PY' 2>/dev/null || true
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
      if FORWARD_ADDRESS="$("$PYTHON" "$REPO_ROOT/scripts/simplex-address.py" --ws-url "$BRIDGE_FORWARD_WS" 2>/dev/null)"; then
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
# is pulled so assess/elicitation/fidelity work without a remote dependency.
if [[ -z "$LLM_MODEL" ]]; then
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
  1. Edit config.json — set llm.model/llm.base_url and codegen.model/codegen.base_url.
     bootstrap.sh seeded config.json from config.example.json (if it was absent)
     and NEVER rewrites it; only you edit config.json.
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
      "$PYTHON" scripts/simplex-address.py
    Add that address as a contact in your SimpleX app.
  - Then put your contactId/display name in config.json
    gateway.simplex.allowed_users (discover the id from a 'gateway_denied' trace
    event or the daemon's /contacts) and reply. With an empty allowlist the
    gateway rejects everyone — that is the safe default.
  - The gateway takes commands and replies. It never reads history, shows invite
    links, or composes messages: that is the forwarding bridge's job.

SimpleX forwarding bridge (messaging UX for skills):
  - Runs as 'semif-simplex-forward' (a second bot daemon/profile, port $SIMPLEX_FORWARD_PORT)
    and 'semif-bridge' (the standalone bridge API). Its contact address is printed
    above; re-print with:
      "$PYTHON" scripts/simplex-address.py --ws-url ws://127.0.0.1:$SIMPLEX_FORWARD_PORT
  - Skills reach it over HTTP via the top-level simplex_bridge_url
    (default http://127.0.0.1:5227); simplex.connect_link shows its address and
    simplex.next_message reads messages sent to it.
EOF
