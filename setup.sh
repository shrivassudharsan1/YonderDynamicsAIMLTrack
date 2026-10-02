#!/usr/bin/env bash
# Create a virtual environment and install every dependency.
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

# Use $PYTHON if set, otherwise the first interpreter that is 3.12 or newer.
is_new_enough() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; }
if [ -z "${PYTHON:-}" ]; then
  for candidate in python3.13 python3.12 python3 python; do
    if is_new_enough "$candidate"; then PYTHON="$candidate"; break; fi
  done
fi
if [ -z "${PYTHON:-}" ] || ! is_new_enough "$PYTHON"; then
  echo "Python 3.12 or newer is required. Install it, or run: PYTHON=/path/to/python bash setup.sh" >&2
  exit 1
fi
echo "Using $("$PYTHON" --version)"
"$PYTHON" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
