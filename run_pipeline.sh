#!/usr/bin/env bash
# Rebuilds everything from the raw data: cleaning -> EDA -> models + evaluation.
set -e
cd "$(dirname "$0")"
python -m src.data_build
python -m src.eda
python -m src.train_evaluate
