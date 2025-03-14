## 安装vllm环境
```
VLLM_USE_PRECOMPILED=1 pip install --e . -v
```


## Start Server

```
#sbatch launch_R1_server.sh 模型路径
sbatch launch_R1_server.sh /home/fit/cwg/WORK/model/Deepseek-R1
```


## Start Client
等server启动成功之后，启动client
- server和client在同一个节点
```
#sbatch launch_client.sh 模型路径
sbatch launch_client.sh /home/fit/cwg/WORK/model/Deepseek-R1
```

- server和client在不同节点
```
#sbatch launch_client.sh 模型路径 server的端口
sbatch launch_client.sh /home/fit/cwg/WORK/model/Deepseek-R1 172.23.18.31:6379
```