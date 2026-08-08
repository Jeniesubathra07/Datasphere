#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --user --upgrade pip setuptools wheel
python3 -m pip install --user -e ".[dev]"

python3 -c "from datasphere.data.archives import extract_dataset_archives; extracted = extract_dataset_archives(); print('Extracted datasets:', extracted or 'none (already present)')"
