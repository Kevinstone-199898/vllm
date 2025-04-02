#!/bin/bash

MODEL_PATH=/mnt/data/deepseek-r1
python client.py --model-path $MODEL_PATH --url 0.0.0.0

echo "=========== processing data =========="

python postprocess.py --model-path $MODEL_PATH
