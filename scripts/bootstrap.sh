#!/usr/bin/env bash
# Provision a new Ubuntu machine into a running semif-agent host.
#
# First run asks a few questions and writes this machine's config from the
# answers (see `python -m semif_agent.cli setup`): the decision engine (local
# SemIf 4B or a remote Winnow server), whether the agent may author new skills,
# which capability packs to enable, and which gateways to run. Bootstrap then
# installs ONLY what those answers require — a REPL-only host never downloads
# the simplex-chat binary, Playwright, or the voice stack.
#
# The two ends of the spectrum are both reachable by the end of a run:
#   * minimum — local SemIf 4B + the `essentials` pack (canned replies,
#     time/timers, calculator). No account, no network, no gateway.
#   * maximum — every pack (SimpleX, browser, Nextcloud, Outlook), a remote
#     Winnow decision engine, remote `llm`/`codegen` endpoints, and all gateways.
#
# Idempotent: safe to rerun; every stage no-ops on already-provisioned state. On
# a rerun the interview is SKIPPED and the existing config is used — pass
# `--reconfigure` to re-run it. This script installs NO ollama; a local `llm`
# pulls its model from the target host's ollama, and `codegen` may be remote.
#
# Must be run from a semif-agent checkout. The checkout is used as-is and never
# re-cloned — only the SemIf engine, the simplex-chat binary, the pack index,
# and the skill packs are fetched. Clone the agent repo first (its SSH key must
# be registered on your git host); the pack index repo (semif-packs, a sibling
# of the origin remote) is then fetched and read for the list of pack repos, and
# only the packs the answers select are cloned into the gitignored packs/ dir.
#
# config.json is the per-machine USER config (a thin index over config/*.json).
# This script seeds it from the interview the FIRST time (or with
# --reconfigure), then reads every per-machine value from it; only a human edits
# it afterwards. Pinned external refs (SemIf commit, GGUF url+sha256, HF
# tokenizer revision, simplex-chat version/url+sha256) live in the committed
# pins.json and the python dep pins in requirements/ — bump those (via git) and
# rerun to upgrade.
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
# Usage:
#   scripts/bootstrap.sh [--reconfigure] [--yes | --answers FILE]
#                        [--copy-data SRC] [--voice] [--browser]
#                        [--gpu MODE] [--pack-base URL] [-h]
#
# Run as the human user; sudo is used internally for system bits.
set -euo pipefail

RECONFIGURE=0
YES=0
ANSWERS_FILE=""
COPY_DATA=""
VOICE=0
BROWSER=0
GPU_MODE="auto"
PACK_BASE=""

usage() {
  sed -n '2,/^set -euo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
  echo
  echo "  --reconfigure          re-run the setup interview and rewrite config/ (default: use existing config)"
  echo "  --yes                  non-interactive: take the minimum working defaults (no prompts)"
  echo "  --answers FILE         non-interactive: read answers from a JSON file (see 'python -m semif_agent.cli setup --help')"
  echo "  --copy-data SRC        rsync SRC (e.g. user@host:/path/to/semif-agent/data) to data/ — opt-in"
  echo "  --voice                also install the optional voice-gateway stack + models"
  echo "  --browser              also install the optional browser-bridge stack (Playwright + Chromium)"
  echo "  --gpu MODE             decision-engine llama.cpp backend: auto (default; a GPU build only when engine.gpu_layers is set), vulkan, rocm, cuda, cpu"
  echo "  --pack-base URL        base URL of the skill-pack repos (default: derived from this checkout's origin remote)"
  echo "  -h                     this help"
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reconfigure) RECONFIGURE=1; shift ;;
    --yes) YES=1; shift ;;
    --answers) ANSWERS_FILE="$2"; shift 2 ;;
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
PLAN="$RUNTIME/bootstrap-plan.json"
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

# --- skill packs (index repo + clone helpers) -----------------------------------
# Which packs exist is owned by a dedicated index repo (`semif-packs`), NOT this
# checkout: it holds packs.json (a name + clone URL per pack). The index is a
# sibling of this checkout's origin remote unless --pack-base overrides the base.
# It is fetched BEFORE the interview so setup can offer its packs; only the packs
# the answers select are cloned (after the interview, before config is written).
# Idempotent: clone when absent, fast-forward when present.
PACK_INDEX_REPO="semif-packs"
PACK_INDEX_DIR="$RUNTIME/pack-index"
PACK_INDEX_FILE="$PACK_INDEX_DIR/packs.json"
if [[ -z "$PACK_BASE" ]]; then
  ORIGIN_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
  PACK_BASE="${ORIGIN_URL%/*}"
fi
PACK_BASE="${PACK_BASE%/}"

# clone_repo URL DEST LABEL — clone when absent, fast-forward when present.
clone_repo() {
  local url="$1" dst="$2" label="$3"
  if [[ -d "$dst/.git" ]]; then
    if git -C "$dst" pull --ff-only --quiet 2>/dev/null; then
      echo "   $label: up to date"
    else
      echo "   WARN: could not update $label (leaving the local checkout)" >&2
    fi
  elif [[ -e "$dst" ]]; then
    echo "   WARN: $label exists but is not a git clone; leaving it" >&2
  elif git clone --quiet "$url" "$dst" 2>/dev/null; then
    echo "   $label: cloned"
  else
    echo "   WARN: could not clone $label from $url" >&2
    return 1
  fi
}

# Fetch (or update) the pack index repo. Succeeds when packs.json is present.
fetch_pack_index() {
  if [[ -z "$PACK_BASE" ]]; then
    echo "no 'origin' remote to derive the pack index from; pass --pack-base URL" >&2
    return 1
  fi
  echo "== pack index (from $PACK_BASE/$PACK_INDEX_REPO)"
  mkdir -p "$RUNTIME"
  clone_repo "$PACK_BASE/$PACK_INDEX_REPO.git" "$PACK_INDEX_DIR" "$PACK_INDEX_REPO" || true
  [[ -f "$PACK_INDEX_FILE" ]]
}

# clone_selected_packs NAME... — clone just these packs, by their index URLs.
clone_selected_packs() {
  local names="$1"
  [[ -z "$names" ]] && return 0
  echo "== skill packs"
  mkdir -p "$REPO_ROOT/packs"
  while read -r name url; do
    [[ -z "$name" ]] && continue
    clone_repo "$url" "$REPO_ROOT/packs/$name" "$name" || true
  done < <(python3 - "$PACK_INDEX_FILE" $names <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
want = set(sys.argv[2:])
for pack in data.get("packs", []):
    name = str(pack.get("name", "")).strip()
    if name in want:
        print(name, pack.get("url", ""))
PY
)
  for name in $names; do
    [[ -f "$REPO_ROOT/packs/$name/pack.json" ]] \
      || echo "WARN: pack '$name' is not present after cloning; its skills will be unavailable" >&2
  done
}

# --- configure this machine (interview -> config/ + secrets.json) --------------
# On a fresh machine (or --reconfigure) this runs in two passes so only the packs
# the answers select are ever cloned:
#   1. collect answers (the interview; no packs needed — the offered list comes
#      from the pack index),
#   2. clone the selected packs (essentials always; +simplex for the gateway's
#      address script; +browser for --browser),
#   3. write config/ + secrets.json from the now-present packs and emit the plan.
# A rerun (existing config) skips the interview and clones only the packs the
# config references, then derives the plan. Either way a machine-readable plan
# drives the conditional install stages below.
mkdir -p "$RUNTIME"
ANSWERS_TMP="$RUNTIME/bootstrap-answers.json"
trap 'rm -f "$ANSWERS_TMP"' EXIT

run_setup() {
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" python3 -m semif_agent.cli setup "$@"
}

# Interaction flags for the answers-collection pass.
collect_args=()
if [[ -n "$ANSWERS_FILE" ]]; then
  if [[ ! -f "$ANSWERS_FILE" ]]; then
    echo "answers file not found: $ANSWERS_FILE" >&2
    exit 1
  fi
  collect_args+=(--answers "$ANSWERS_FILE")
elif [[ "$YES" = 1 ]]; then
  collect_args+=(--yes)
elif [[ -t 0 && -t 1 ]]; then
  collect_args+=(--interview)
else
  echo "== no terminal for the setup interview; using the minimum defaults (--yes)" >&2
  collect_args+=(--yes)
fi

# The index is the source of truth for which packs exist; without it nothing can
# be installed. Fetch it before the interview so the offered list comes from it.
if ! fetch_pack_index; then
  echo "ERROR: could not fetch the pack index ($PACK_BASE/$PACK_INDEX_REPO.git)" >&2
  echo "       packs cannot be installed — pass --pack-base URL, or clone the index" >&2
  echo "       manually into $PACK_INDEX_DIR" >&2
  exit 1
fi
INDEX_ARG=(--pack-index "$PACK_INDEX_FILE")

if [[ ! -f "$CONFIG" || "$RECONFIGURE" = 1 ]]; then
  if [[ "$RECONFIGURE" = 1 && -f "$CONFIG" ]]; then
    echo "== reconfiguring $CONFIG (rewriting config/ + secrets.json)"
  else
    echo "== configuring this machine"
  fi
  # 1. Collect answers (writes $ANSWERS_TMP; touches no config, needs no packs).
  run_setup --path "$CONFIG" --answers-out "$ANSWERS_TMP" \
    "${INDEX_ARG[@]}" "${collect_args[@]}" \
    || { echo "setup failed" >&2; exit 1; }
  # 2. Clone only the packs the answers selected.
  SELECTED_PACKS="$(python3 - "$ANSWERS_TMP" "$BROWSER" <<'PY'
import json, sys
answers = json.load(open(sys.argv[1]))
packs = list(dict.fromkeys(["essentials", *(answers.get("packs") or [])]))
if sys.argv[2] == "1" and "browser" not in packs:
    packs.append("browser")            # --browser needs the Playwright fallback script
if "simplex" in (answers.get("gateways") or []) and "simplex" not in packs:
    packs.append("simplex")            # the gateway prints its address via the pack script
print(" ".join(packs))
PY
)"
  clone_selected_packs "$SELECTED_PACKS"
  # 3. Write config/ + secrets.json from the now-present packs, and the plan.
  write_args=(--path "$CONFIG" --answers "$ANSWERS_TMP" --plan-out "$PLAN")
  [[ "$RECONFIGURE" = 1 ]] && write_args+=(--force)
  run_setup "${write_args[@]}" || { echo "setup failed" >&2; exit 1; }
  rm -f "$ANSWERS_TMP"
else
  echo "== using existing $CONFIG (pass --reconfigure to change it)"
  # A rerun clones only the packs the existing config references, then derives the
  # plan from them (the pack manifests are needed to compute the install plan).
  MERGED_EARLY="$RUNTIME/config.merged.json"
  python3 "$REPO_ROOT/scripts/config_merge.py" "$CONFIG" > "$MERGED_EARLY" \
    || { echo "could not read $CONFIG" >&2; exit 1; }
  RERUN_PACKS="$(python3 - "$MERGED_EARLY" "$PACK_INDEX_FILE" "$BROWSER" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
index = json.load(open(sys.argv[2]))
bridges = cfg.get("bridges", {}) or {}
packs_cfg = cfg.get("packs", {}) or {}
gateway = cfg.get("gateway", {}) or {}
names = [str(p.get("name", "")).strip() for p in index.get("packs", [])]
out = []
for name in names:
    if not name or name == "essentials":
        continue
    block = bridges.get(name) if name in bridges else packs_cfg.get(name)
    if isinstance(block, dict) and block.get("enabled", True):
        out.append(name)
if "essentials" not in out:
    out.insert(0, "essentials")
if bool((gateway.get("simplex") or {}).get("enabled")) and "simplex" not in out:
    out.append("simplex")
if sys.argv[3] == "1" and "browser" not in out:
    out.append("browser")
print(" ".join(out))
PY
)"
  clone_selected_packs "$RERUN_PACKS"
  run_setup --path "$CONFIG" --plan-only --plan-out "$PLAN" \
    || { echo "could not derive the install plan from $CONFIG" >&2; exit 1; }
fi

# Read the plan (install decisions) and the merged config (per-machine values).
# The ASCII unit separator (\x1f) delimits fields so an empty value never shifts
# the fields after it.
IFS=$'\x1f' read -r PLAN_ENGINE PLAN_AUTHORING PLAN_PACKS PLAN_GATEWAYS \
  PLAN_GATEWAY_PLATFORMS PLAN_BRIDGE_PACKS PLAN_INSTALL_ENGINE PLAN_BRIDGE_SVC \
  PLAN_GATEWAY_SVC PLAN_SIMPLEX_BIN PLAN_SIMPLEX_FWD PLAN_BROWSER PLAN_VOICE \
  PLAN_PACK_REQS < <(
  python3 - "$PLAN" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
def flag(key):
    return "1" if p.get(key) else "0"
print("\x1f".join([
    str(p.get("engine_provider", "")),
    str(p.get("authoring", "")),
    ",".join(p.get("packs", [])),
    ",".join(p.get("gateways", [])),
    ",".join(p.get("gateway_platforms", [])),
    ",".join(p.get("bridge_packs", [])),
    flag("install_engine"),
    flag("install_bridge_service"),
    flag("install_gateway_service"),
    flag("install_simplex_binary"),
    flag("install_simplex_forward"),
    flag("install_browser"),
    flag("install_voice"),
    ",".join(p.get("pack_requirements", [])),
]))
PY
)

# The optional-stack flags force an install even when the plan did not pick it
# (e.g. a scripted pre-seed); enabling it in config is still the operator's call.
[[ "$VOICE" = 1 ]] && PLAN_VOICE=1
[[ "$BROWSER" = 1 ]] && PLAN_BROWSER=1
# A forced optional stack also pulls in that pack's own python deps.
if [[ "$BROWSER" = 1 ]]; then
  case ",$PLAN_PACK_REQS," in
    *,browser,*) ;;
    *) PLAN_PACK_REQS="${PLAN_PACK_REQS:+$PLAN_PACK_REQS,}browser" ;;
  esac
fi

# Messaging transports (websockets / lxmf) are needed by the SimpleX gateway or
# bridge and the LXMF gateway.
NEED_TRANSPORTS=0
[[ "$PLAN_SIMPLEX_BIN" = 1 ]] && NEED_TRANSPORTS=1
[[ ",$PLAN_GATEWAYS," == *",lxmf,"* ]] && NEED_TRANSPORTS=1

echo "== plan: engine=$PLAN_ENGINE | authoring=$PLAN_AUTHORING | packs=$PLAN_PACKS"
echo "         gateways=${PLAN_GATEWAYS:-none} | install: engine=$PLAN_INSTALL_ENGINE transports=$NEED_TRANSPORTS bridge=$PLAN_BRIDGE_SVC gateway=$PLAN_GATEWAY_SVC browser=$PLAN_BROWSER voice=$PLAN_VOICE"
echo "         pack deps=${PLAN_PACK_REQS:-none}"

# The live config may be split (config.json includes config/*.json); merge it so
# the reads below see one flat object. Stdlib-only, so it works before the venv.
MERGED_CONFIG="$RUNTIME/config.merged.json"
python3 "$REPO_ROOT/scripts/config_merge.py" "$CONFIG" > "$MERGED_CONFIG" \
  || { echo "could not read $CONFIG" >&2; exit 1; }

# --- read pins from the committed pins.json ---------------------------------
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

if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
  if [[ -z "$SEMIF_REPO" || -z "$SEMIF_REF" || -z "$GGUF_URL" || -z "$GGUF_SHA256" ]]; then
    echo "engine pins missing in $PINS (semif_repo/semif_ref/gguf_url/gguf_sha256)" >&2
    exit 1
  fi
fi
if [[ "$PLAN_SIMPLEX_BIN" = 1 ]] && [[ -z "$SIMPLEX_BIN_URL" || -z "$SIMPLEX_SHA256" ]]; then
  echo "simplex_chat pins missing in $PINS for arch '$SIMPLEX_ARCH' (assets.<arch>.bin_url/sha256)" >&2
  exit 1
fi

# --- read per-machine values from config.json ----------------------------------
IFS=$'\x1f' read -r CFG_LLM_URL CFG_LLM_MODEL CFG_LLM_PROVIDER CFG_CODEGEN_URL CFG_CODEGEN_MODEL \
  CFG_CODEGEN_PROVIDER CFG_ENGINE_PROVIDER CFG_WINNOW_URL CFG_WINNOW_MODEL \
  SIMPLEX_PORT SIMPLEX_DISPLAY SIMPLEX_FORWARD_PORT SIMPLEX_FORWARD_DISPLAY < <(
  python3 - "$MERGED_CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
s = cfg.get("simplex_chat", {})
eng = cfg.get("engine", {})
win = eng.get("winnow", {})
def strip_v1(u):
    u = str(u or "").rstrip("/")
    return u[:-3] if u.endswith("/v1") else u
print("\x1f".join([
    strip_v1(cfg.get("llm", {}).get("base_url")),
    str(cfg.get("llm", {}).get("model", "") or ""),
    str(cfg.get("llm", {}).get("provider", "ollama") or "ollama").lower(),
    strip_v1(cfg.get("codegen", {}).get("base_url")),
    str(cfg.get("codegen", {}).get("model", "") or ""),
    str(cfg.get("codegen", {}).get("provider", "ollama") or "ollama").lower(),
    str(eng.get("provider", "semif") or "semif").lower(),
    strip_v1(win.get("base_url")),
    str(win.get("model", "") or ""),
    str(s.get("port", "")),
    str(s.get("display_name", "") or ""),
    str(s.get("forward_port", "")),
    str(s.get("forward_display_name", "") or ""),
]))
PY
)

LLM_URL="${CFG_LLM_URL:-http://127.0.0.1:11434}"
LLM_MODEL="$CFG_LLM_MODEL"
LLM_PROVIDER="${CFG_LLM_PROVIDER:-ollama}"
CODEGEN_URL="$CFG_CODEGEN_URL"
CODEGEN_MODEL="$CFG_CODEGEN_MODEL"
CODEGEN_PROVIDER="${CFG_CODEGEN_PROVIDER:-ollama}"
ENGINE_PROVIDER="${CFG_ENGINE_PROVIDER:-semif}"
WINNOW_URL="$CFG_WINNOW_URL"
WINNOW_MODEL="$CFG_WINNOW_MODEL"

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

echo "== pins: simplex-chat sha256 ${SIMPLEX_SHA256:0:12}… | gguf ${GGUF_SHA256:0:12}…"
if [[ "$ENGINE_PROVIDER" = "winnow" ]]; then
  echo "== decision engine: remote Winnow ($WINNOW_URL, model ${WINNOW_MODEL:-?})"
else
  echo "== decision engine: local SemIf @ $SEMIF_REF"
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

# --- stage 2: ssh key + SemIf engine clone (local engine only) ----------------
mkdir -p ~/.ssh
if [[ ! -f ~/.ssh/id_ed25519 ]]; then
  echo "== generating ~/.ssh/id_ed25519 (register this key on your git host for the agent repo)"
  ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
  cat ~/.ssh/id_ed25519.pub
fi

git -C "$REPO_ROOT" pull --ff-only || true

if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
  if [[ ! -d "$ENGINE/.git" ]]; then
    echo "== cloning SemIf engine from $SEMIF_REPO into $ENGINE"
    git clone "$SEMIF_REPO" "$ENGINE"
  fi
  if [[ "$(git -C "$ENGINE" rev-parse HEAD 2>/dev/null)" != "$SEMIF_REF" ]]; then
    echo "== pinning SemIf engine to $SEMIF_REF"
    git -C "$ENGINE" fetch origin
    git -C "$ENGINE" checkout "$SEMIF_REF"
  fi
else
  echo "== remote decision engine selected; skipping the local engine clone"
fi

# --- stage 2b: pick the llama.cpp GPU backend (local engine only) -------------
# A GPU-enabled build is opt-in: llama.cpp's Vulkan backend splits the compute
# graph across CPU and Vulkan even with `n_gpu_layers=0`, roughly doubling CPU
# latency, so a CPU-only host must not get one. `--gpu <backend>` forces a build;
# `--gpu auto` (default) builds a GPU backend only when config.json asks for
# offload (`engine.gpu_layers` nonzero).
GPU_BACKEND=cpu
CONFIG_GPU_LAYERS=0
if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
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
  echo "== decision-engine llama.cpp backend: $GPU_BACKEND (engine.gpu_layers=$CONFIG_GPU_LAYERS)"
fi
case "$GPU_BACKEND" in
  vulkan) LLAMA_CMAKE_ARGS="-DGGML_VULKAN=on" ;;
  rocm)   LLAMA_CMAKE_ARGS="-DGGML_HIP=on" ;;
  cuda)   LLAMA_CMAKE_ARGS="-DGGML_CUDA=on" ;;
  *)      LLAMA_CMAKE_ARGS="" ;;
esac

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
echo "== installing core deps into $VENV"
"$PIP" install --upgrade pip setuptools wheel
"$PIP" install -r "$REPO_ROOT/requirements/core.txt"

if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
  echo "== installing engine deps (llama.cpp CPU/GPU path)"
  "$PIP" install -e "$ENGINE" --no-deps
  CMAKE_ARGS="$LLAMA_CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 \
    "$PIP" install -r "$REPO_ROOT/requirements/engine.txt"

  # Make the installed llama-cpp-python match the requested backend: pip ships a
  # CPU wheel and its wheel cache hands it back even when CMAKE_ARGS asks for a
  # GPU backend, so force a source build whenever offload support disagrees.
  installed_gpu=no
  "$PYTHON" -c 'import llama_cpp, sys; sys.exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)' 2>/dev/null && installed_gpu=yes
  want_gpu=yes
  [[ "$GPU_BACKEND" = cpu ]] && want_gpu=no
  if [[ "$installed_gpu" != "$want_gpu" ]]; then
    LLAMA_PIN="$(grep -m1 -E '^llama-cpp-python==' "$REPO_ROOT/requirements/engine.txt")"
    echo "== rebuilding llama-cpp-python for the $GPU_BACKEND backend ($LLAMA_PIN)"
    CMAKE_ARGS="$LLAMA_CMAKE_ARGS" CMAKE_BUILD_PARALLEL_LEVEL=6 MAKEFLAGS=-j6 \
      "$PIP" install --no-deps --force-reinstall --no-cache-dir --no-binary llama-cpp-python "$LLAMA_PIN"
  fi
fi

if [[ "$NEED_TRANSPORTS" = 1 ]]; then
  echo "== installing messaging transport deps (websockets / lxmf)"
  "$PIP" install -r "$REPO_ROOT/requirements/transports.txt"
fi

"$PIP" install -e "$REPO_ROOT" --no-deps

# --- stage 3b: optional voice-gateway stack ------------------------------------
# Hardware-dependent: only the host with a microphone/speaker needs it. The
# stack is lazy-imported, so a host without it still runs every other front end.
if [[ "$PLAN_VOICE" = 1 ]]; then
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

# --- stage 3c: pack-declared python deps ---------------------------------------
# A pack owns the python deps its skills/bridge need, shipped as
# packs/<name>/requirements.txt (standard pip format). The install plan lists the
# packs this host selected/enabled that ship one; a pack that does not enable
# installs nothing. Deps are lazy-imported by the pack's bridge/skills, so a host
# without them still runs every other pack. See PACKGEN.md, "Pack python deps".
if [[ -n "$PLAN_PACK_REQS" ]]; then
  IFS=',' read -ra PACK_REQ_LIST <<< "$PLAN_PACK_REQS"
  for pack_name in "${PACK_REQ_LIST[@]}"; do
    [[ -z "$pack_name" ]] && continue
    PACK_REQ="$REPO_ROOT/packs/$pack_name/requirements.txt"
    if [[ -f "$PACK_REQ" ]]; then
      echo "== installing deps for pack '$pack_name' ($PACK_REQ)"
      "$PIP" install -r "$PACK_REQ"
    else
      echo "   WARN: pack '$pack_name' has no requirements.txt (not cloned?); skipping" >&2
    fi
  done
fi

# --- stage 3d: optional browser-bridge binaries --------------------------------
# The browser pack's python deps (Playwright) come from
# packs/browser/requirements.txt in stage 3c; this stage downloads the browser
# binary. Optional: lazy-imported, so a host without it still runs every other
# bridge. Browsers go under the checkout's .runtime/browser (the containment
# rule); the bridge sets PLAYWRIGHT_BROWSERS_PATH to the same path at launch.
if [[ "$PLAN_BROWSER" = 1 ]]; then
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

# --- stage 4: GGUF (local engine only) ------------------------------------------
if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
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

  # --- stage 5: HF tokenizer cache ---
  export HF_HOME="$HF_CACHE"
  mkdir -p "$HF_CACHE"
  echo "== pre-fetching tokenizer $HF_SOURCE @ $HF_REV into $HF_CACHE"
  "$PYTHON" - "$HF_SOURCE" "$HF_REV" <<'PY'
import sys
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(sys.argv[1], revision=sys.argv[2])
PY
else
  echo "== remote decision engine: no GGUF/tokenizer to fetch"
fi

# --- stage 5b: optional voice models -------------------------------------------
# Downloads the openWakeWord models (into the venv) + the Piper voice (into
# .runtime/voice/tts) and warms the faster-whisper model into the HF cache.
# Reads gateway.voice from config.json so the models match the configured names.
# Best-effort: a download failure warns (the gateway reports the missing model at
# startup) rather than aborting.
if [[ "$PLAN_VOICE" = 1 ]]; then
  echo "== downloading voice models (openWakeWord + Piper + faster-whisper)"
  HF_HOME="$HF_CACHE" "$PYTHON" "$REPO_ROOT/scripts/voice-models.py" --config "$MERGED_CONFIG" \
    || echo "WARN: voice model download failed; re-run scripts/voice-models.py" >&2
fi

# --- stage 6: simplex-chat daemon binary (only when needed) --------------------
if [[ "$PLAN_SIMPLEX_BIN" = 1 ]]; then
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
else
  echo "== SimpleX not selected; skipping the simplex-chat binary"
fi

# --- stage 8: systemd user units (only the ones the answers need) --------------
render_unit() {
  local src="$1" dst="$2"
  sed -e "s|@REPO@|$REPO_ROOT|g" \
      -e "s|@PORT@|$SIMPLEX_PORT|g" \
      -e "s|@DISPLAY_NAME@|$SIMPLEX_DISPLAY|g" \
      -e "s|@FORWARD_PORT@|$SIMPLEX_FORWARD_PORT|g" \
      -e "s|@FORWARD_DISPLAY@|$SIMPLEX_FORWARD_DISPLAY|g" \
      -e "s|@PLATFORMS@|$PLAN_GATEWAY_PLATFORMS|g" "$src" > "$dst"
}

WANTED_UNITS=()
[[ "$PLAN_GATEWAY_SVC" = 1 ]] && WANTED_UNITS+=(semif-gateway.service)
[[ ",$PLAN_GATEWAYS," == *",simplex,"* ]] && WANTED_UNITS+=(semif-simplex.service)
[[ "$PLAN_SIMPLEX_FWD" = 1 ]] && WANTED_UNITS+=(semif-simplex-forward.service)
[[ "$PLAN_BRIDGE_SVC" = 1 ]] && WANTED_UNITS+=(semif-bridge.service)
ALL_UNITS=(semif-simplex.service semif-simplex-forward.service semif-gateway.service semif-bridge.service)

unit_wanted() { local u="$1"; for w in "${WANTED_UNITS[@]:-}"; do [[ "$w" = "$u" ]] && return 0; done; return 1; }

echo "== rendering systemd user units into $UNITS (wanted: ${WANTED_UNITS[*]:-none})"
UNIT_CHANGED=0
ACTIVE_BEFORE=""
for unit in "${ALL_UNITS[@]}"; do
  if ! unit_wanted "$unit"; then
    # A reconfigure can drop a component; retire its unit if it was installed.
    if [[ -L "$USER_UNITS/$unit" || -f "$UNITS/$unit" ]]; then
      if command -v systemctl >/dev/null 2>&1; then
        export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
        systemctl --user disable --now "$unit" 2>/dev/null || true
      fi
      rm -f "$USER_UNITS/$unit"
      echo "   $unit: retired (no longer selected)"
    fi
    continue
  fi
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

if command -v systemctl >/dev/null 2>&1 && [[ ${#WANTED_UNITS[@]} -gt 0 ]]; then
  export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  if systemctl --user daemon-reload 2>/dev/null; then
    systemctl --user enable --now "${WANTED_UNITS[@]}" 2>/dev/null \
      && echo "== enabled ${WANTED_UNITS[*]}" \
      || echo "WARN: could not enable user services (run: systemctl --user enable --now ${WANTED_UNITS[*]})" >&2
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

# --- stage 8b: SimpleX bot contact address -------------------------------------
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

# --- stage 8c: forwarding bridge contact address -------------------------------
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

# --- stage 9: optional data copy -----------------------------------------------
if [[ -n "$COPY_DATA" ]]; then
  echo "== rsync data from $COPY_DATA"
  mkdir -p "$REPO_ROOT/data"
  rsync -a "$COPY_DATA/" "$REPO_ROOT/data/"
fi

# --- stage 10: verify imports + the configured endpoints ------------------------
echo "== verifying imports"
if [[ "$PLAN_INSTALL_ENGINE" = 1 ]]; then
  "$PYTHON" -c "import semif_phase1, semif_agent; print('engine + agent import OK')"
else
  "$PYTHON" -c "import semif_agent; print('agent import OK')"
fi

# The decision engine is always real: a remote Winnow server must be reachable at
# runtime (there is no fallback), so surface its address here.
if [[ "$ENGINE_PROVIDER" = "winnow" ]]; then
  echo "== decision engine: remote Winnow at ${WINNOW_URL:-?} (model ${WINNOW_MODEL:-?}) — must be reachable at runtime"
fi

# The `llm` (title/description + gateway cleanup) model. `semif` runs in-process
# on the loaded decision model; a local ollama is pulled; hosted endpoints are
# only reported. Authoring is off when the mode is `deny`.
if [[ "$LLM_PROVIDER" = "semif" ]]; then
  echo "== llm provider semif: the loaded engine model generates text in-process (no ollama pull/probe)"
elif [[ "$PLAN_AUTHORING" = "deny" ]]; then
  echo "== skill authoring is disabled; the llm/codegen endpoints are not required"
elif [[ "$LLM_PROVIDER" = "opencode" ]]; then
  echo "== llm: hosted OpenCode Console ($LLM_URL, model ${LLM_MODEL:-?})"
elif [[ -z "$LLM_MODEL" ]]; then
  echo "WARN: llm.model is not set in config.json — authoring needs it" >&2
else
  echo "== llm ollama ($LLM_API), model $LLM_MODEL"
  if ! curl -sf --max-time 5 "$LLM_API/api/tags" >/dev/null 2>&1; then
    echo "WARN: no local ollama at $LLM_API — install/run ollama and pull $LLM_MODEL" >&2
    echo "      (bootstrap installs no ollama; only codegen may be remote)" >&2
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

# Codegen writes skill bodies; only needed when authoring is enabled. Warn only —
# never auto-pull.
if [[ "$PLAN_AUTHORING" = "deny" ]]; then
  : # authoring disabled; no codegen endpoint required
elif [[ -z "$CODEGEN_MODEL" ]]; then
  echo "WARN: new_skill_creation is enabled but codegen.model is unset — authoring will not run" >&2
elif [[ "$CODEGEN_PROVIDER" = "opencode" ]]; then
  echo "== codegen: hosted OpenCode Console ($CODEGEN_URL, model $CODEGEN_MODEL)"
elif [[ -n "$CODEGEN_URL" ]]; then
  echo "== codegen ollama ($CODEGEN_API)"
  CODEGEN_TAGS="$(curl -sf --max-time 5 "$CODEGEN_API/api/tags" || true)"
  if grep -qF "$CODEGEN_MODEL" <<<"$CODEGEN_TAGS"; then
    echo "codegen serves $CODEGEN_MODEL: OK"
  else
    echo "WARN: codegen host $CODEGEN_API serves no $CODEGEN_MODEL (offline? new-skill authoring will fail)" >&2
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  echo "== services"
  for unit in "${ALL_UNITS[@]}"; do
    state="$(systemctl --user is-active "$unit" 2>/dev/null || true)"
    [[ -n "$state" ]] && echo "  ${unit%.service}: $state"
  done
fi

cat <<EOF

Done. What this machine is configured for:
  engine:    $ENGINE_PROVIDER
  authoring: $PLAN_AUTHORING
  packs:     $PLAN_PACKS
  gateways:  ${PLAN_GATEWAYS:-none (REPL + dashboard)}

Next steps:
  1. Edit the split config if you need to (config/core.json, config/gateway.json,
     config/packs/*.json); credentials live in secrets.json. To re-run the setup
     interview instead: scripts/bootstrap.sh --reconfigure
  2. Skill packs are cloned from their own repos into packs/ (the list lives in
     the semif-packs index repo); they are gitignored here. Update them with:
       for d in packs/*/; do git -C "\$d" pull --ff-only; done
  cd "$REPO_ROOT"
  HF_HOME="$HF_CACHE" "$PYTHON" -m semif_agent.cli run
  HF_HOME="$HF_CACHE" "$PYTHON" -m pytest tests/integration -q -s  (needs real engine+LLM)
  HF_HOME="$HF_CACHE" "$PYTHON" -m semif_agent.cli dashboard --port 8765
EOF

if [[ ",$PLAN_GATEWAYS," == *",simplex,"* ]]; then
  cat <<EOF

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
EOF
fi

if [[ ",$PLAN_GATEWAYS," == *",lxmf,"* ]]; then
  cat <<EOF

LXMF command gateway (commands only):
  - Runs IN-PROCESS inside the same 'semif-gateway' service (no separate daemon);
    the transport is installed from requirements/transports.txt (lxmf + rns).
  - Its LXMF address (32 hex chars) is printed on connect:
      journalctl --user -u semif-gateway -f
    Add that address as a contact in an LXMF client (Sideband, MeshChatX, ...)
    and put it in gateway.lxmf.allowed_users (discover a sender from a
    'gateway_denied' trace event). With an empty allowlist it rejects everyone —
    the safe default.
EOF
fi

if [[ "$PLAN_SIMPLEX_FWD" = 1 ]]; then
  cat <<EOF

SimpleX forwarding bridge (messaging UX for skills):
  - Runs as 'semif-simplex-forward' (a second bot daemon/profile, port $SIMPLEX_FORWARD_PORT)
    and 'semif-bridge' (the standalone bridge API). Its contact address is printed
    above; re-print with:
      "$PYTHON" packs/simplex/scripts/simplex-address.py --ws-url ws://127.0.0.1:$SIMPLEX_FORWARD_PORT
  - Skills reach it over HTTP at the address derived from bridges.simplex
    (host/port, default http://127.0.0.1:5227); simplex.connect_link shows its
    address and simplex.next_message reads messages sent to it.
EOF
fi

if [[ "$PLAN_BRIDGE_SVC" = 1 ]]; then
  cat <<EOF

Bridge services (run inside 'semif-bridge'):
  - LLM bridge: a generic POST /chat in front of the configured `llm` model
    (address derived from bridges.llm, default http://127.0.0.1:5229).
  - Nextcloud bridge: files/calendar/tasks/contacts/notes over one JSON surface
    (configure bridges.nextcloud, default http://127.0.0.1:5230).
  - Outlook bridge: Microsoft Graph mail/calendar/To Do/OneDrive
    (configure bridges.outlook; run packs/outlook/scripts/outlook-auth.py once).
  - Browser bridge: Playwright automation with a persistent profile under
    .runtime/browser/profile (address derived from bridges.browser, default
    http://127.0.0.1:5232).
EOF
fi

if [[ "$PLAN_VOICE" = 1 ]]; then
  cat <<EOF

Voice command gateway (wake word + speech-to-text + text-to-speech):
  - The stack is installed and its models downloaded (openWakeWord into the
    venv, the Piper voice into .runtime/voice/tts, faster-whisper into .runtime/hf).
    Run it in the foreground (it needs your audio session; it is NOT a systemd unit):
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
EOF
fi

if [[ "$PLAN_BROWSER" = 1 ]]; then
  cat <<EOF

Browser bridge (browser automation for skills):
  - The Playwright stack is installed and the '$BROWSER_ENGINE' browser is under
    $BROWSERS. It runs inside 'semif-bridge' (no separate daemon) and keeps a
    persistent profile at .runtime/browser/profile, so a login you do once
    survives across calls and restarts. Set bridges.browser.headless = false to
    watch the window (useful the first time you sign in to a site); POST /reset
    with purge=true wipes the stored profile.
  - The shipped pack skills are browser.open_page, browser.read_page, and
    browser.screenshot.
  - If the bridge reports 503, the browser failed to launch — run:
      PLAYWRIGHT_BROWSERS_PATH=$BROWSERS "$PYTHON" -m playwright install $BROWSER_ENGINE
EOF
fi
