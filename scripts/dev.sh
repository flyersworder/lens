#!/usr/bin/env bash
# Local development launcher for the LENS web tier.
#
# Default mode is "preview": runs ``next build`` once (memory-spike
# but bounded) and then ``next start`` (a Node HTTP server with NO
# compiler in the loop, ~100 MB resident). uvicorn runs alongside
# under the same shell, behind a Ctrl+C trap that tears both down.
#
# WHY NOT ``next dev`` BY DEFAULT
#   Next.js 16's Turbopack dev server holds the full source tree +
#   node_modules + a TS type-checker in memory and re-checks on every
#   keystroke. On a 16 GB Apple Silicon laptop with a browser open,
#   that has triggered macOS kernel panics in this repo (see
#   docs/web-deployment.md "Local development"). Production mode
#   compiles once and then serves static assets + tiny dynamic
#   handlers, which is plenty for inspecting the UI.
#
# MODE SWITCH (LENS_DEV_MODE)
#   preview  — default. ``next build`` + ``next start``. Low memory
#              at runtime; rebuilds require a script restart.
#   hot      — ``next dev`` with Turbopack. Heavy memory; HMR works.
#   webpack  — ``next dev --no-turbopack``. Mid memory; HMR works.
#
# STORAGE BACKEND
#   Default: ``LensStore`` (sqlite-vec) against ~/.lens/data/lens.db.
#   ``--turso`` flag (or any TURSO_* env): use TursoStore. Requires
#   TURSO_DATABASE_URL + TURSO_AUTH_TOKEN, or TURSO_PROD_* in .env
#   when ``--turso`` is passed (we copy PROD into the unprefixed
#   names that ``api/index.py`` looks for).
#
# Usage:
#   ./scripts/dev.sh                       # preview mode
#   LENS_DEV_MODE=hot ./scripts/dev.sh     # full HMR (heavy)
#   LENS_DEV_MODE=webpack ./scripts/dev.sh # HMR with webpack
#   ./scripts/dev.sh --turso               # use lens-prod via Turso
#   LENS_API_PORT=8001 ./scripts/dev.sh    # alt backend port

set -euo pipefail

# Reset SIGINT/SIGTERM handlers BEFORE we install our cleanup trap.
# POSIX says a script invoked as a background job (``./dev.sh &`` from
# a non-interactive shell) inherits SIG_IGN for INT/QUIT — so plain
# ``trap cleanup INT TERM`` at script bottom would silently fail to
# fire on macOS bash 3.2. Explicitly unmask first, then re-trap.
trap - INT TERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- load .env (auto-export) -----------------------------------------
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# --- arg parse --------------------------------------------------------
USE_TURSO=0
DO_STOP=0
for arg in "$@"; do
  case "$arg" in
    --turso) USE_TURSO=1 ;;
    --stop|-s) DO_STOP=1 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
  esac
done

# PID file for ``--stop``. Persists between launches so a user can
# cleanly tear down a dev session from any other terminal even if
# the original Ctrl+C didn't fire (e.g. when launched as a
# background job via ``&``, where bash silently inherits SIG_IGN
# for INT/TERM and the in-script trap doesn't reliably fire).
PID_FILE="${LENS_DEV_PIDFILE:-/tmp/lens-dev.pids}"

_kill_tree() {
  local pid=$1
  local sig=${2:-TERM}
  local child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    _kill_tree "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}

if [[ "$DO_STOP" == "1" ]]; then
  if [[ ! -f "$PID_FILE" ]]; then
    echo "dev.sh --stop: no PID file at $PID_FILE (nothing to stop)" >&2
    exit 0
  fi
  # shellcheck disable=SC1090
  source "$PID_FILE"
  echo "dev.sh --stop: tearing down api=$API_PID web=$WEB_PID …"
  _kill_tree "${WEB_PID:-0}" TERM
  _kill_tree "${API_PID:-0}" TERM
  sleep 1
  _kill_tree "${WEB_PID:-0}" KILL
  _kill_tree "${API_PID:-0}" KILL
  # Belt-and-suspenders: anything still on the dev ports.
  for port in "${API_PORT:-8000}" "${WEB_PORT:-3000}"; do
    lp=""
    lp=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1) || true
    [[ -n "$lp" ]] && kill -9 "$lp" 2>/dev/null || true
  done
  rm -f "$PID_FILE" || true
  echo "dev.sh --stop: done"
  exit 0
fi

if [[ "$USE_TURSO" == "1" ]]; then
  export TURSO_DATABASE_URL="${TURSO_DATABASE_URL:-${TURSO_PROD_DATABASE_URL:-}}"
  export TURSO_AUTH_TOKEN="${TURSO_AUTH_TOKEN:-${TURSO_PROD_AUTH_TOKEN:-}}"
  if [[ -z "${TURSO_DATABASE_URL:-}" ]]; then
    echo "dev.sh: --turso requested but TURSO_PROD_DATABASE_URL is not set" >&2
    exit 1
  fi
fi

MODE="${LENS_DEV_MODE:-preview}"
API_PORT="${LENS_API_PORT:-8000}"
WEB_PORT="${LENS_WEB_PORT:-3000}"

case "$MODE" in
  preview|hot|webpack) ;;
  *)
    echo "dev.sh: unknown LENS_DEV_MODE='$MODE' (use preview|hot|webpack)" >&2
    exit 1
    ;;
esac

# --- preflight: ports free? ------------------------------------------
for port in "$API_PORT" "$WEB_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "dev.sh: port $port is already in use." >&2
    echo "  pkill -f 'uvicorn api.index' && pkill -f 'next dev\\|next start' to clean up" >&2
    exit 1
  fi
done

API_LOG="$(mktemp -t lens-api.XXXXXX.log)"
WEB_LOG="$(mktemp -t lens-web.XXXXXX.log)"
echo "dev.sh: api log → $API_LOG"
echo "dev.sh: web log → $WEB_LOG"

# --- BUILD step (preview mode only, before anything is up) -----------
# Compiling the frontend BEFORE booting either server keeps the
# memory peak isolated to the build step. If the build OOMs, the
# user sees a clear failure and no children are leaked.
if [[ "$MODE" == "preview" ]]; then
  echo "dev.sh: building frontend (next build) — this is the memory-heavy step…"
  if ! npx next build >"$WEB_LOG" 2>&1; then
    echo "dev.sh: next build failed. Last 30 lines:" >&2
    tail -30 "$WEB_LOG" >&2 || true
    exit 1
  fi
  echo "dev.sh: build complete."
fi

# --- start uvicorn (no torch — see [local-store] extra in pyproject) -
uv run --frozen --extra local-store --extra serve \
  uvicorn api.index:app --host 127.0.0.1 --port "$API_PORT" --log-level info \
  >"$API_LOG" 2>&1 &
API_PID=$!

# Probe health before the frontend lands so the proxy doesn't 502.
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done
if ! curl -sf "http://127.0.0.1:$API_PORT/api/health" >/dev/null 2>&1; then
  echo "dev.sh: backend never became healthy. Last 30 log lines:" >&2
  tail -30 "$API_LOG" >&2 || true
  kill "$API_PID" 2>/dev/null || true
  exit 1
fi
echo "dev.sh: backend healthy on http://127.0.0.1:$API_PORT"

# --- start frontend --------------------------------------------------
case "$MODE" in
  preview)
    npx next start --port "$WEB_PORT" >>"$WEB_LOG" 2>&1 &
    ;;
  hot)
    npx next dev --port "$WEB_PORT" >"$WEB_LOG" 2>&1 &
    ;;
  webpack)
    npx next dev --port "$WEB_PORT" --no-turbopack >"$WEB_LOG" 2>&1 &
    ;;
esac
WEB_PID=$!

# Persist PIDs + ports for ``./scripts/dev.sh --stop`` (works from
# any other terminal, regardless of whether the in-script trap fires).
cat >"$PID_FILE" <<EOF
API_PID=$API_PID
WEB_PID=$WEB_PID
API_PORT=$API_PORT
WEB_PORT=$WEB_PORT
EOF

# --- cleanup trap (single Ctrl+C kills both, and grandchildren) ------
# In a non-interactive bash script, ``cmd &`` does NOT put the child
# in its own process group, and our children are wrappers (``uv run``
# wraps Python; ``npm exec`` / ``npx`` wraps the next-server node
# process). A plain ``kill $PID`` only signals the wrapper — the
# inner process is left orphaned and bound to its port. We use the
# ``_kill_tree`` helper defined above to walk descendants explicitly.

cleanup() {
  echo
  echo "dev.sh: shutting down (api=$API_PID web=$WEB_PID)…"
  _kill_tree "$WEB_PID" TERM
  _kill_tree "$API_PID" TERM
  for _ in $(seq 1 10); do
    if ! kill -0 "$API_PID" 2>/dev/null \
       && ! kill -0 "$WEB_PID" 2>/dev/null \
       && [[ -z "$(pgrep -P "$API_PID" 2>/dev/null)$(pgrep -P "$WEB_PID" 2>/dev/null)" ]]; then
      break
    fi
    sleep 0.5
  done
  _kill_tree "$WEB_PID" KILL
  _kill_tree "$API_PID" KILL
  # Belt-and-suspenders: anything still bound to our ports gets
  # SIGKILL'd by PID. Catches detached grandchildren that re-parented
  # to init before our pgrep walk could see them.
  # ``set -o pipefail`` would otherwise abort cleanup() when lsof
  # finds nothing (exit 1) — append ``|| true`` to every command
  # whose non-zero exit is expected here.
  for port in "$API_PORT" "$WEB_PORT"; do
    local lp=""
    lp=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1) || true
    [[ -n "$lp" ]] && kill -9 "$lp" 2>/dev/null || true
  done
  rm -f "$PID_FILE" || true
  echo "dev.sh: bye"
}
trap cleanup EXIT INT TERM

storage_label="LensStore (~/.lens/data/lens.db)"
[[ -n "${TURSO_DATABASE_URL:-}" ]] && storage_label="Turso ($TURSO_DATABASE_URL)"

cat <<EOF

  LENS dev stack is up. (mode: $MODE)
    Frontend   http://localhost:$WEB_PORT
    API (proxy) http://localhost:$WEB_PORT/api/health
    API direct  http://localhost:$API_PORT/api/health
    Storage    $storage_label

  Press Ctrl+C to stop both processes cleanly.

EOF

# Block until at least one child exits; the EXIT trap kills the other.
# Avoids ``wait -n`` (bash 4+ only; macOS still ships 3.2 by default).
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 0.5
done
