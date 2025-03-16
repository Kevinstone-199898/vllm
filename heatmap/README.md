## 安装vllm环境
```
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --e . -v
```


## Start Server

```
cd heatmap
#sbatch launch_R1_server.sh 模型路径
sbatch launch_R1_server.sh /home/fit/cwg/WORK/model/Deepseek-R1
```


## Start Client
等server启动成功之后，启动client
```
#sbatch launch_client.sh 模型路径
sbatch launch_client.sh /home/fit/cwg/WORK/model/Deepseek-R1
```
