from openai import OpenAI
import argparse
import csv
import os

def get_folder(folder):
    latest_dir = None
    max_mtime = 0

    with os.scandir(folder) as entries:
        for entry in entries:
            if entry.is_dir():
                mtime = entry.stat().st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
                    latest_dir = entry.path  # 或 entry.name 仅获取名称
    return latest_dir

def launch_vllm_client(model, url):

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://{url}:8000/v1"
    print(f"------url: {openai_api_base}-----")

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    prompts = [
        # "What is your name?"
        "You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n\n[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n \nYou are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n\n[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n \nYou are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n\n[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n \nYou are an expert Python programmer, and here is your task: Write a python function to remove first and last occurrence of a given character from the string. Your code should pass these tests:\n\n assert remove_Occ(\"hello\",\"l\") == \"heo\"\nassert remove_Occ(\"abcda\",\"a\") == \"bcd\"\nassert remove_Occ(\"PHP\",\"P\") == \"H\"  \n\n[BEGIN]\n",
        "以下是中国关于计算机网络考试的单项选择题，请选出其中的正确答案。\n使用位填充方法，以01111110为位首flag，数据为011011111111111111110010，求问传送时要添加几个0____\nA. 1\nB. 2\nC. 3\nD. 4\n答案: ",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\ntrans-cinnamaldehyde was treated with methylmagnesium bromide, forming product 1.\n\n1 was treated with pyridinium chlorochromate, forming product 2.\n\n3 was treated with (dimethyl(oxo)-l6-sulfaneylidene)methane in DMSO at elevated temperature, forming product 3.\n\nhow many carbon atoms are there in product 3?\n\nA) 12\nB) 14\nC) 11\nD) 10",
        # "A man is sitting on a roof. He\nQuestion: Which ending makes the most sense?\nA. is using wrap to wrap a pair of skis.\nB. is ripping level tiles off.\nC. is holding a rubik's cube.\nD. starts pulling up roofing on a roof.\nYou may choose from 'A', 'B', 'C', 'D'.\nAnswer:",
    ]

    cur_dir = os.path.abspath(os.getcwd())
    main_folder = f"{cur_dir}/attn_scores"

    for i in range(len(prompts)):
        content  = prompts[i]
        messages = [{"role": "user", "content": content}]
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=10000, timeout=100000000)
        reasoning_content = response.choices[0].message.reasoning_content
        reply = response.choices[0].message.content
        
        local_folder = get_folder(main_folder)
        file = f"{local_folder}/prompt.txt"
        with open(file, 'a', encoding='utf-8') as file:
            file.write("\n\n=====reasoning_content=====\n\n")
            if(reasoning_content != None):
                file.write(reasoning_content)
            file.write("\n\n=====content=====\n\n")
            if(reply != None):
                file.write(reply)

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/dataset/Deepseek-v2-lite')
    parser.add_argument('--url', type=str, default='http://0.0.0.0:8000/v1')
    args = parser.parse_args()

    launch_vllm_client(args.model_path, args.url)