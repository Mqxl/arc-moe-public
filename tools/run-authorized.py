#!/usr/bin/env python3
"""Run bench/check with an API key from the environment or model config; never place it in the command line."""
import json, os, runpy, sys
from pathlib import Path
config, script, *args = sys.argv[1:]
key = os.environ.get('STRATA_API_KEY') or os.environ.get('API_TOKEN') or json.loads(Path(config).read_text()).get('api_key')
sys.argv = [script, *args]
if key and '--api-key' not in args:
    sys.argv += ['--api-key', key]
runpy.run_path(script, run_name='__main__')
