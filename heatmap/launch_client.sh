#!/bin/bash
#SBATCH --output=log/client_%j.log
#SBATCH --job-name=deepseek-r1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --partition=h01

MODEL_PATH=$1
echo "model path: $MODEL_PATH"
if [ $# -lt 3 ]; then
    echo "server and client on different node"
    python client.py --model-path $MODEL_PATH --url $2
else
    echo "server and client on same node"
    python client.py --model-path $MODEL_PATH
fi

echo "=========== processing data =========="

python postprocess.py --model-path $MODEL_PATH