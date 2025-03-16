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
from multiprocessing import Pool
        
def get_text(model, path):
    tokenizer = AutoTokenizer.from_pretrained(model)
    token_ids = []
    int_token_ids = []
    text = []
    token_file = f"{path}/token_ids.csv"
    with open(token_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row_number, row in enumerate(reader):
            int_token_ids.extend([int(x) for x in row])
            token_ids.extend(row)
    for i in int_token_ids:
            text.append(tokenizer.decode(i, skip_special_tokens=False))
    # for i in range(len(text)):
    #     if text[i].startswith('<｜'):
    #         text[i] = "<｜s｜>"
    return text

# save_score of a specific layer
# path is the layer path
def save_score(path, num_output_tokens, max_len):
    layer_id = path.split("_")[-1]
    target_path = f"{os.path.dirname(path)}/scores"
    tp_size = sum(1 for filename in os.listdir(path) if filename.lower().endswith('.csv'))
    assert num_heads % tp_size == 0
    tp_num_heads = num_heads // tp_size
    head_id = 0
    all_scores = np.empty((num_heads, num_output_tokens, max_len))

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path): 
            if not file_path.lower().endswith('.csv'):
                continue
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                attn_weights = list(reader)
                total_num = len(attn_weights)
                assert total_num % tp_num_heads == 0
                assert num_output_tokens == total_num // tp_num_heads
                assert max_len == len(attn_weights[-1])
                for j in range(tp_num_heads):
                    score = np.empty((0, max_len))
                    for k in range(num_output_tokens):
                        x = np.array(attn_weights[j+k*tp_num_heads], dtype=float)
                        if len(x) < max_len:
                            padded_x = np.pad(x, (0, max_len - len(x)), 'constant', constant_values=0.0)
                        score = np.vstack((score, padded_x))
                    all_scores[head_id] = score
                    head_id += 1
    # print(all_scores, flush=True)
    np.save(f'{target_path}/layer_{layer_id}_attn_scores.npy', all_scores)
    print(f"===== save layer {layer_id} =====", flush=True)

#draw heat map of a specific layer
#path is layer path
def draw_heatmap(layer_id, path, num_heads):
    colors = ['white', 'red', 'black']  # 从白色到红色
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    scores = np.load(f'{path}/scores/layer_{layer_id}_attn_scores.npy')
    target_path = f"{path}/figs/layer_{layer_id}"
    os.makedirs(target_path, exist_ok=True)

    for i in range(num_heads):
        # 设置x坐标轴标签
        plt.imshow(scores[i], cmap=custom_cmap, vmin=0, vmax=1) 
        plt.colorbar()
        plt.savefig(f'{target_path}/head_{i+1}.png', dpi=600)
        plt.close()
        print(f"===== draw layer {layer_id} head {i+1} =====", flush=True)

def get_model_data(model):
    with open(f'{model}/config.json', 'r') as file:
        data = json.load(file)
        num_layers = data.get("num_hidden_layers")
        num_heads = data.get("num_key_value_heads")
    return num_layers, num_heads

def get_mata_data(path):
    token_file = f"{path}/token_ids.csv"
    num_output_tokens = 0
    max_len = -1
    
    with open(token_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row_number, row in enumerate(reader):
            if row_number == 1:
                num_output_tokens = len(row) - 1 
            max_len += len(row)
        
    return max_len, num_output_tokens

def process_case(path, num_heads, num_output_tokens, max_len):
    save_score(path, num_output_tokens, max_len)
    draw_heatmap(path.split("_")[-1],os.path.dirname(path), num_heads)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/dataset/Deepseek-v2-lite')
    args = parser.parse_args()

    num_layers, num_heads = get_model_data(args.model_path)
    folder = os.path.abspath(os.getcwd())
    folder = f"{folder}/attn_scores"
    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():
                path = entry.path
                print(f"===== working on folderr {path} =====", flush=True)
                max_len, num_output_tokens = get_mata_data(path)
                os.makedirs(f"{path}/scores", exist_ok=True)
                os.makedirs(f"{path}/figs", exist_ok=True)
                with Pool(processes=num_layers) as pool:
                    # 使用 map 方法并行处理 1 至 26 层
                    pool.starmap(process_case, [(f"{path}/layer_{i+1}", num_heads, num_output_tokens, max_len) for i in range(num_layers)])
                # with os.scandir(path) as items:
                #     for item in items:
                #         if item.is_dir():
                #             layer_path = item.path
                #             # print(layer_path)
                #             if not layer_path.split("/")[-1].lower().startswith('layer_'):
                #                 continue
                #             save_score(layer_path ,num_output_tokens, max_len)
                # for layer_id in range(num_layers):
                #     draw_heatmap(layer_id + 1, path, num_heads)

                    # if(os.path.basename(path) != "2025-03-14-15-51-14"):
                    #     continue
