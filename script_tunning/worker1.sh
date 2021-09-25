#!/bin/bash

export KERASTUNER_TUNER_ID="tunner0"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
export NCCL_DEBUG=WARN
python Hyperband_180Datasets.py
