#!/usr/bin/env bash
# ./arc bench [--long] [--runs N] [--url URL] - prompt and decode speed of the running server (tools/bench.py)
set -euo pipefail
. "$(dirname "$0")/lib.sh"
exec python3 "$ROOT/tools/bench.py" "$@"
