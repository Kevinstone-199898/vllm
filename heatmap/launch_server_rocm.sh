#!/bin/bash

# Before launching the server, the customized vllm should be installed:
# - Using docker `rocm/vllm:rocm6.3.1_instinct_vllm0.7.3_20250325`
# - Install the customized vllm using `PYTORCH_ROCM_ARCH="gfx942" python3 setup.py develop`

MODEL_PATH=/mnt/data/deepseek-r1
export SPARSITY=0.2
export SAVE_SCORE=1
export NUM_LAYERS=$(grep '"num_hidden_layers"' ${MODEL_PATH}/config.json | awk -F ': ' '{print $2}' | tr -d ', ')
vllm serve $MODEL_PATH --enforce-eager \
    --enable-reasoning --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 8 \
    --distributed-executor-backend=ray --trust-remote-code \
    --uvicorn-log-level debug
