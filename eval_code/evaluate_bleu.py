import os
import json
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# 定义计算BLEU分数的函数
"""def calculate_bleu_score(reference, candidate):
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    # 计算BLEU分数
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)#这个总是运行不了
    return bleu_score
"""
# 固定保存的JSON文件地址
output_json_file = "F:\\nlp\\output.json"

result_dict = {}
# 尝试加载JSON文件，如果文件不存在，则创建一个空的result_dict
if os.path.exists(output_json_file):
    try:
        with open(output_json_file, "r") as f:
            result_dict = json.load(f)
    except json.JSONDecodeError:
        print("无法解析 JSON 文件内容，请检查文件格式或重新创建文件。")
    except FileNotFoundError:
        print("找不到指定的输出 JSON 文件。")
    except Exception as e:
        print(f"加载 JSON 文件时出现未知错误：{e}")
else:
    # 创建一个空的result_dict
    result_dict = {}

    # 将空的result_dict保存到JSON文件中
    with open(output_json_file, "w") as f:
        json.dump(result_dict, f)

# 加载JSON文件
json_file_path = "F:\\Poison_LLM\\data\\gpt2\\res_test.json"
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# 从JSON数据中提取指令和输出
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# 提取文件名作为file_name
file_name = json_file_path.split("\\")[-1]

# 读取指定行的数据并将其添加到结果字典中
output_txt_file = "new_output.txt"
line_number = 0  # 假设要读取第n行的数据,请改为（n-1）
with open(output_txt_file, "r") as txt_file:
    lines = txt_file.readlines()
    model_values_str = lines[line_number - 1].strip().split()  # 读取指定行并将其拆分为单个值
    model_values = [float(value) for value in model_values_str]  # 将字符串转换为数值

# 初始化一个空列表来存储 BLEU 值
all_bleu_scores = []
# 遍历JSON数据
for i in tqdm(range(len(instruction_list)), desc="Calculating BLEU scores"):
    instruction = instruction_list[i]
    output = output_list[i]
    # 计算BLEU分数
    bleu_score = sacrebleu.raw_corpus_bleu([output], [[instruction]])
    # 将BLEU分数添加到全局的BLEU列表中
    all_bleu_scores.append(bleu_score.score)

# 初始化一个空列表来存储model值
result_dict[file_name] = {
    "L1": [],
    "L2": [],
    "consine": [],
    "model": model_values,
    "BLEU": all_bleu_scores
}
  

# 将结果保存为JSON文件
with open(output_json_file, "w") as output_json:
    json.dump(result_dict, output_json, indent=4)

print("BLEU scores and model values have been saved to output_bleu.json.")
