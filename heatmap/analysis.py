# import numpy as np
import os

# folder = "attn_scores/2025-03-15-12-41-44"
# attn_scores = np.load(f'{folder}/attn_scores.npy')

# def get_top_n_indices(arr, n=10):
#     # 使用 argsort 获取排序后的索引
#     sorted_indices = np.argsort(arr, axis=1)
#     # 取出每行的最后 n 个索引
#     top_n_indices = sorted_indices[:, -n:]
#     sorted_top_n_indices = np.sort(top_n_indices, axis=1)
#     return sorted_top_n_indices

# for i in range(attn_scores.shape[0]):
#     for j in range(attn_scores.shape[1]):
#         for k in range(attn_scores.shape[2]):
#             try:
#                 assert np.allclose(np.sum(attn_scores[i][j][k]), 1)
#             except AssertionError:
#                 print(f"Assertion failed at layer {i}, head {j}, line {k}: sum is {np.sum(attn_scores[i][j][k])}")

# # import code
# # code.interact(local=locals())

# folder = "attn_scores/2025-03-15-12-41-44"
# import pandas as pd

# for i in range(61):
#     file = f"{folder}/layer_{i+1}"
#     for j in range(8):
#         file_name = f"{file}/tp_{j}.csv"
#         # 读取 CSV 文件
#         print(f"====reading {file_name}=====")
#         df = pd.read_csv(file_name)
#         # 获取行数
#         num_rows = df.shape[0]

#         print(num_rows)

# def get_folder(folder):
#     latest_dir = None
#     max_mtime = 0

#     with os.scandir(folder) as entries:
#         for entry in entries:
#             if entry.is_dir():
#                 mtime = entry.stat().st_mtime
#                 if mtime > max_mtime:
#                     max_mtime = mtime
#                     latest_dir = entry.path  # 或 entry.name 仅获取名称
#     return latest_dir

import glob
from datetime import datetime

# 指定文件夹路径
folder_path = 'attn_scores'  # 替换为实际路径

def find_latest_time(folder):
    # 定义时间格式解析器
    time_format = "%Y-%m-%d-%H-%M-%S"
    datetime_objects = []
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():    
                # 将字符串转换为datetime对象
                datetime_objects.append(datetime.strptime(os.path.basename(entry.path), time_format))
                
                # 找到最晚时间
                latest_datetime = max(datetime_objects)
                
    # 还原为原始格式字符串
    return f"{folder}/{latest_datetime.strftime(time_format)}"


for i in range(100):
    print(find_latest_time(folder_path))
