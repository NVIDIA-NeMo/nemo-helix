#!/usr/bin/env bash
# Check this machine before you start the tutorial. Read only: it changes nothing,
# starts nothing and never prints keys or tokens.
#
#   bash scripts/preflight.sh                  # full run: data, training, serving, agent
#   bash scripts/preflight.sh --no-gpu         # no GPU here: steps 0 to 5, the agent on hosted models, the analysis
#   bash scripts/preflight.sh --model nemotron # check for Nemotron Nano 9B v2 instead of Qwen3.5-4B
#
# Settings: NHX_BASE_URL (default http://localhost:8080), NEMO (the nemo CLI, default `nemo`),
# PY (Python for the repo scripts, default .venv/bin/python if present, else python3).
# Exit code: 0 if every required check passed, 1 if any required check failed.
set -u

GPU=1
MODEL=qwen
while [ $# -gt 0 ]; do
  case "$1" in
    --no-gpu) GPU=0 ;;
    --model) shift; MODEL="${1:-qwen}" ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
case "$MODEL" in
  qwen) HF_REPO="Qwen/Qwen3.5-4B-Base" ;;
  nemotron) HF_REPO="nvidia/NVIDIA-Nemotron-Nano-9B-v2" ;;
  *) echo "--model must be qwen or nemotron" >&2; exit 2 ;;
esac

HERE="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${NHX_BASE_URL:-http://localhost:8080}"
BASE="${BASE%/}"
NEMO="${NEMO:-nemo}"
if [ -n "${PY:-}" ]; then :
elif [ -x "$HERE/.venv/bin/python" ]; then PY="$HERE/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then PY="$VIRTUAL_ENV/bin/python"
else PY="python3"; fi

FAILS=0
WARNS=0
pass() { printf 'PASS  %-25s %s\n' "$1" "$2"; }
warn() { printf 'WARN  %-25s %s\n' "$1" "$2"; [ -n "${3:-}" ] && printf '      %-25s fix: %s\n' "" "$3"; WARNS=$((WARNS + 1)); }
fail() { printf 'FAIL  %-25s %s\n' "$1" "$2"; [ -n "${3:-}" ] && printf '      %-25s fix: %s\n' "" "$3"; FAILS=$((FAILS + 1)); }
skip() { printf 'SKIP  %-25s %s\n' "$1" "$2"; }
# A GPU-only check fails in a full run and is only reported with --no-gpu.
gpu_fail() { if [ "$GPU" = 1 ]; then fail "$@"; else skip "$1" "$2 (not needed with --no-gpu)"; fi; }

echo "Preflight for the NeMo Helix specialized-agent tutorial"
echo "Helix URL: $BASE   mode: $([ "$GPU" = 1 ] && echo "full (GPU)" || echo "no GPU")   model: $HF_REPO"
echo

# 1. Operating system. Helix runs local GPU work on Linux x86_64 only.
OS="$(uname -s)"; ARCH="$(uname -m)"
if [ "$OS" = Linux ] && [ "$ARCH" = x86_64 ]; then
  pass "OS" "$OS $ARCH"
elif [ "$OS" = Darwin ] || [ "$OS" = Linux ]; then
  if [ "$GPU" = 1 ]; then
    fail "OS" "$OS $ARCH: Helix runs local GPU training and serving on Linux x86_64 only" \
      "use a Linux x86_64 machine with an NVIDIA GPU for steps 6 to 10, or rerun with --no-gpu here"
  else
    pass "OS" "$OS $ARCH (fine without a GPU)"
  fi
else
  fail "OS" "$OS $ARCH is not supported" "use Linux or macOS"
fi

# 2. Python for the repo scripts.
if PYV="$("$PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null)"; then
  case "$PYV" in
    3.12.*|3.13.*) pass "Python" "$PYV ($PY)" ;;
    3.14.*) warn "Python" "$PYV ($PY): Helix supports it, but we ran this tutorial on 3.13" \
              "uv venv --python 3.13 && source .venv/bin/activate" ;;
    *) fail "Python" "$PYV ($PY): needs 3.12 or 3.13" "uv venv --python 3.13 && source .venv/bin/activate" ;;
  esac
else
  fail "Python" "no Python found ($PY)" "install uv, then: uv venv --python 3.13 && source .venv/bin/activate"
fi

# 3. uv
if command -v uv >/dev/null 2>&1; then
  UVV="$(uv --version 2>/dev/null | awk '{print $2}')"
  pass "uv" "$UVV"
else
  fail "uv" "not found" "curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# 4. nemo CLI
if NV="$("$NEMO" --version 2>/dev/null | grep -i 'version' | head -1)" && [ -n "$NV" ]; then
  pass "nemo CLI" "${NV#nemo }"
else
  fail "nemo CLI" "'$NEMO' not found or not runnable" \
    "source .venv/bin/activate && uv pip install -r requirements.txt  (or set NEMO=/path/to/nemo)"
fi

# 5. Helix services. /health/ready is the check the Helix docs give.
READY="$(curl -s -m 5 "$BASE/health/ready" 2>/dev/null)"
case "$READY" in
  *'"ready"'*) pass "Helix services" "$BASE/health/ready says ready" ;;
  "") fail "Helix services" "no answer from $BASE" "nemo setup   (or: nemo services start)" ;;
  *) fail "Helix services" "$BASE answered but is not ready: ${READY:0:80}" "wait a minute, then: nemo services logs" ;;
esac

# 6. Inference provider. Names and status only; keys are never read.
if [ -n "${NV:-}" ] && [ -n "$READY" ]; then
  PROVIDERS="$("$NEMO" --base-url "$BASE" inference providers list --workspace "${NHX_WORKSPACE:-default}" -f json 2>/dev/null \
    | "$PY" -c 'import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
rows = d.get("data", d) if isinstance(d, dict) else d
ready = [p.get("name", "?") for p in rows if str(p.get("status", "")).upper() == "READY"]
print(" ".join(ready))' 2>/dev/null)"
  if [ -n "$PROVIDERS" ]; then
    pass "Inference provider" "ready: $PROVIDERS"
  else
    fail "Inference provider" "no READY provider in workspace ${NHX_WORKSPACE:-default}" \
      "nemo setup  (pick a provider), or see docs/tutorial.md step 1"
  fi
else
  fail "Inference provider" "skipped: needs the nemo CLI and running Helix services" "fix the two checks above first"
fi

# 7. Python packages the repo scripts import.
if "$PY" -c 'import nemo_helix_plugin, data_designer.config, pandas, httpx, mcp' >/dev/null 2>&1; then
  pass "Script packages" "Helix SDK, data-designer, pandas, httpx, mcp"
else
  fail "Script packages" "missing in $PY" "source .venv/bin/activate && uv pip install -r requirements.txt"
fi
if command -v meridian-ops-tools >/dev/null 2>&1; then
  pass "Agent tool server" "meridian-ops-tools on PATH"
else
  fail "Agent tool server" "meridian-ops-tools not on PATH" "source .venv/bin/activate && uv pip install -e agent"
fi

# 8. NVIDIA GPU. Helix asks for 40 GB at least and recommends 80 GB; we used one A100 80GB.
if command -v nvidia-smi >/dev/null 2>&1 && GPUS="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null)" && [ -n "$GPUS" ]; then
  BEST=0; NAMES=""
  while IFS=, read -r name mem; do
    mem="${mem// /}"; NAMES="${NAMES:+$NAMES; }$(echo "$name" | sed 's/^ *//') ${mem} MiB"
    [ "${mem:-0}" -gt "$BEST" ] 2>/dev/null && BEST="$mem"
  done <<EOF
$GPUS
EOF
  GB=$((BEST / 1024))
  if [ "$GB" -ge 79 ]; then pass "GPU" "$NAMES"
  elif [ "$GB" -ge 39 ]; then warn "GPU" "$NAMES: meets the Helix minimum (40 GB); we only ran on 80 GB" "lower batch.per_device_train_batch_size in training/job.json if training runs out of memory"
  else gpu_fail "GPU" "$NAMES: Helix needs 40 GB or more for GPU work" "use an A100 or H100 80GB"; fi
  CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | awk '{print $3}')"
  case "${CUDA%%.*}" in
    ''|*[!0-9]*) warn "CUDA" "could not read the CUDA version from nvidia-smi" ;;
    *) if [ "${CUDA%%.*}" -ge 13 ]; then pass "CUDA" "$CUDA"
       else gpu_fail "CUDA" "$CUDA: Helix GPU work needs CUDA 13 or later" "update the NVIDIA driver to the R580 branch or later"; fi ;;
  esac
else
  gpu_fail "GPU" "no NVIDIA GPU visible (nvidia-smi missing or found no GPU)" \
    "run steps 6 to 10 on a Linux machine with an NVIDIA GPU, 80 GB recommended"
fi

# 9. Docker. Local Helix services need it, and training and serving run in containers.
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  pass "Docker" "daemon reachable"
  if [ "$GPU" = 1 ]; then
    if docker info 2>/dev/null | grep -i 'runtimes' | grep -qi nvidia; then
      pass "NVIDIA container runtime" "listed in docker info"
    else
      fail "NVIDIA container runtime" "not listed in docker info" "install the NVIDIA Container Toolkit and restart Docker"
    fi
  fi
else
  case "$BASE" in
    http://localhost*|http://127.0.0.1*)
      fail "Docker" "not installed or the daemon isn't running" "start Docker Desktop (macOS) or the docker service (Linux)" ;;
    *) gpu_fail "Docker" "not installed or the daemon isn't running" "start Docker" ;;
  esac
fi

# 10. Disk space where the repo lives. Helix needs 16 GB; model weights and images need more.
FREE_KB="$(df -Pk "$HERE" | awk 'NR==2 {print $4}')"
FREE_GB=$((FREE_KB / 1024 / 1024))
if [ "$FREE_GB" -lt 16 ]; then
  fail "Disk space" "${FREE_GB} GB free: Helix needs 16 GB" "free up space"
elif [ "$GPU" = 1 ] && [ "$FREE_GB" -lt 100 ]; then
  warn "Disk space" "${FREE_GB} GB free: training images, the vLLM image and model weights need more (our rough estimate: 100 GB)"
else
  pass "Disk space" "${FREE_GB} GB free"
fi

# 11. Hugging Face, where the base model comes from (public, no token needed).
HF="$(curl -s -o /dev/null -w '%{http_code}' -m 10 "https://huggingface.co/api/models/$HF_REPO" 2>/dev/null)"
if [ "$HF" = 200 ]; then
  pass "Hugging Face" "$HF_REPO reachable"
else
  gpu_fail "Hugging Face" "$HF_REPO not reachable (HTTP ${HF:-none})" "check outbound HTTPS to huggingface.co"
fi

echo
if [ "$FAILS" -gt 0 ]; then
  echo "Result: $FAILS required check(s) failed, $WARNS warning(s). Fix the FAIL lines, then run this again."
  exit 1
fi
echo "Result: ready ($WARNS warning(s))."
exit 0
