#!/bin/sh
set -eu
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
python prepare_data.py
python fetch_models.py
python baseline.py
python bert.py
python generator.py validation
python generator.py test
python evaluate.py
python reload_latency.py
python generator.py robustness
python token_examples.py
