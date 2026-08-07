#!/usr/bin/env bash
# Dev launcher for the J.A.R.V.I.S. backend.
#
# Creates a virtualenv if one doesn't exist, installs/updates requirements,
# and runs uvicorn with autoreload. Safe to re-run any number of times.
#
# Usage:
#   ./run.sh
#
# Env overrides:
#   JARVIS_HOST, JARVIS_PORT  — override the bind address/port for this run
#                                (falls back to app.config.settings values).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Installing/updating requirements"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f "../.env" ] && [ ! -f ".env" ]; then
    echo "==> No .env found. Copy .env.example to .env and set OPENROUTER_API_KEY."
    echo "    (cp ../.env.example ../.env)"
fi

HOST="${JARVIS_HOST:-127.0.0.1}"
PORT="${JARVIS_PORT:-8000}"

echo "==> Starting uvicorn on $HOST:$PORT (reload enabled)"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --reload
