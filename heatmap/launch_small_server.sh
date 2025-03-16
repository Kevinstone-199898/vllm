#!/bin/bash
#SBATCH --output=log/small_R1_%j.log
#SBATCH --job-name=deepseek-r1-small
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:8
#SBATCH --tasks-per-node=1
#SBATCH --partition=h01

cp /home/fit/cwg/WORK/sxy/deepseek-r1-config/small_model.safetensors.index.json /home/fit/cwg/WORK/model/Deepseek-R1/model.safetensors.index.json
cp /home/fit/cwg/WORK/sxy/deepseek-r1-config/small_config.json /home/fit/cwg/WORK/model/Deepseek-R1/config.json

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node using ${SLURM_CPUS_PER_TASK} cpus and 8 gpus"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i using ${SLURM_CPUS_PER_TASK} cpus and 8 gpus"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 8 --block &
    sleep 5
done

if [ $# -lt 1 ]; then
    echo "输入模型路径"
    exit 1
else
    MODEL_PATH=$1
    export SPARSITY=0.2 
    export SAVE_SCORE=1
    export NUM_LAYERS=$(grep '"num_hidden_layers"' ${MODEL_PATH}/config.json | awk -F ': ' '{print $2}' | tr -d ', ')
    vllm serve $MODEL_PATH --enforce-eager \
    --enable-reasoning --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 8 --pipeline-parallel-size 1 \
    --distributed-executor-backend=ray --trust-remote-code \
    --uvicorn-log-level debug
fi