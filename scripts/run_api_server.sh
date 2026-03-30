#!/usr/bin/env bash
set -euo pipefail

source /home/bumblebee/anaconda3/etc/profile.d/conda.sh
conda activate optimize
cd /home/bumblebee/Project/optimize
export PYTHONUNBUFFERED=1

exec /home/bumblebee/anaconda3/envs/optimize/bin/python ml_engine/api_server.py
