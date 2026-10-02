#!/usr/bin/env bash
# Create a virtual environment and install every dependency.
#   bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
"$PYTHON" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"
