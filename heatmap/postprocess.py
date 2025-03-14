import csv
import os
import seaborn as sns
import numpy as np
import torch
import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoModelForCausalLM, AutoTokenizer
import json


def postprocess(model):
    tokenizer = AutoTokenizer.from_pretrained(model)

    # 定义自定义颜色映射
    colors = ['white', 'red', 'black']  # 从白色到红色
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

    with open(f'{model}/config.json', 'r') as file:
        data = json.load(file)
        num_layers = data.get("num_hidden_layers")
        num_heads = data.get("num_key_value_heads")

    folder = os.path.abspath(os.getcwd())
    folder = f"{folder}/attn_scores"

    with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir():
                    path = entry.path
                    # print(path)
                    # if(os.path.basename(path) != "2025-03-14-15-51-14"):
                    #     continue
                    token_file = f"{path}/token_ids.csv"
                    token_ids = []
                    int_token_ids = []
                    text = []
                    with open(token_file, 'r') as csvfile:
                        reader = csv.reader(csvfile)
                        for row in reader:
                            int_token_ids.extend([int(x) for x in row])
                            token_ids.extend(row)
                    for i in int_token_ids:
                        text.append(tokenizer.decode(i, skip_special_tokens=False))
                    for i in range(len(text)):
                        if text[i].startswith('<｜'):
                            text[i] = "<｜s｜>"
                    for i in range(num_layers):
                        count = 0
                        folder_name = f'{path}/layer_{i+1}'
                        tp_size = sum(1 for filename in os.listdir(folder_name) if os.path.isfile(os.path.join(folder_name, filename)))
                        assert num_heads % tp_size == 0
                        tp_num_heads = num_heads // tp_size
                        fig, axs = plt.subplots(4, 4, figsize=(16, 16))
                        for filename in os.listdir(folder_name):
                            # 拼接完整路径
                            file_path = os.path.join(folder_name, filename)
                            if os.path.isfile(file_path): 
                                with open(file_path, 'r') as f:
                                    reader = csv.reader(f)
                                    attn_weights = list(reader)
                                    total_num = len(attn_weights)
                                    # import code
                                    # code.interact(local=locals())
                                    assert total_num % tp_num_heads == 0
                                    num_output_tokens = total_num // tp_num_heads
                                    min_len = len(attn_weights[0]) - 1
                                    max_len = len(attn_weights[-1])
                                    for j in range(tp_num_heads):
                                        score = np.empty((0, max_len))
                                        for k in range(num_output_tokens):
                                            x = np.array(attn_weights[j+k*tp_num_heads], dtype=float)
                                            if len(x) < max_len:
                                                padded_x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0.0)
                                            score = np.vstack((score, padded_x))
                                        #plot heatmap
                                        axs.flat[count].imshow(score, cmap=custom_cmap, vmin=0, vmax=1) 
                                        # axs.flat[count].imshow(score[:,min_len:], cmap=custom_cmap, vmin=0, vmax=1) 
                                        # 设置x坐标轴标签
                                        axs.flat[count].set_xticks(range(max_len))
                                        axs.flat[count].set_xticklabels(text[:-1], rotation=88)
                                        # 设置y坐标轴标签
                                        axs.flat[count].set_yticks(range(num_output_tokens))
                                        axs.flat[count].set_yticklabels(text[min_len:-1])
                                        count += 1 
                                        
                                    # 显示图形
                        plt.savefig(f'{path}/layer_{i+1}.png', dpi=600)
                        plt.close()
                        print(f"===== save layer {i+1} =====", flush=True)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/dataset/Deepseek-v2-lite')
    args = parser.parse_args()

    postprocess(args.model_path)
