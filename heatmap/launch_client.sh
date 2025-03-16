#!/bin/bash
#SBATCH --output=log/client_%j.log
#SBATCH --job-name=deepseek-r1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --partition=h01

if [ $# -lt 1 ]; then
    echo "请输入模型路径"
    exit 1
else
    MODEL_PATH=$1
    echo "model path: $MODEL_PATH"
    URL=$(<ip.txt)
    echo "server ip: $URL"
    python client.py --model-path $MODEL_PATH --url $URL
fi

echo "=========== processing data =========="

python postprocess.py --model-path $MODEL_PATH