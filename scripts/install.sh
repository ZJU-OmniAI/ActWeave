#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
python - "${root}/pyproject.toml" <<'PYTHON'
from pathlib import Path
import subprocess
import sys
import tomllib

requirements = tomllib.loads(Path(sys.argv[1]).read_text())['project']['dependencies']
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
PYTHON
npm --prefix "${root}" ci
