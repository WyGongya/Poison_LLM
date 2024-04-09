import os
import json
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import numpy as np
# 加载GloVe预训练词向量
def load_glove_vectors(file_path):
    glove_vectors = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_vectors[word] = vector
    return glove_vectors

glove_file_path = 'glove.6B.100d.txt'  # 这里根据实际的GloVe文件路径进行修改
glove_vectors = load_glove_vectors(glove_file_path)


# 计算两个向量的L1范式
def calculate_L1_norm(vector):
    return np.sum(np.abs(vector))

# 计算两个向量的L2范式
def calculate_L2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

# 计算两个向量的余弦相似度
def calculate_cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    cosine_similarity = dot_product / norm_product
    return cosine_similarity

# 固定保存的JSON文件地址
output_json_file = "..\\result_data\\output.json"

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
json_file_path = "..\\data\\gpt2\\res_test_poisoned_0.json"
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# 从JSON数据中提取指令和输出
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# 提取文件名作为file_name
file_name = json_file_path.split("\\")[-1]

# 读取指定行的数据并将其添加到结果字典中
output_txt_file = "..\\result_data\\new_output.txt"
line_number = 1  # 假设要读取第n行的数据,请改为（n-1）
with open(output_txt_file, "r") as txt_file:
    lines = txt_file.readlines()
    model_values_str = lines[line_number - 1].strip().split()  # 读取指定行并将其拆分为单个值
    model_values = [float(value) for value in model_values_str]  # 将字符串转换为数值

# 初始化一个空列表来存储 BLEU,L1,L2,cos 值
all_bleu_scores = []
all_L1=[]
all_L2=[]
all_cos=[]
# 遍历JSON数据
for i in tqdm(range(len(instruction_list)), desc="Calculating BLEU scores, L1, L2 norms and cosine similarities"):
    instruction = instruction_list[i]
    output = output_list[i]

    instruction_tokens = word_tokenize(instruction.lower())
    output_tokens = word_tokenize(output.lower())

    # 将每个单词的词向量相加得到句子向量
    instruction_vector = np.sum([glove_vectors.get(word, np.zeros(100)) for word in instruction_tokens], axis=0)
    output_vector = np.sum([glove_vectors.get(word, np.zeros(100)) for word in output_tokens], axis=0)
    
    # 计算L1和L2范式
    L1_norm = calculate_L1_norm(instruction_vector - output_vector)
    L2_norm = calculate_L2_norm(instruction_vector - output_vector)
    
    # 将NaN值替换为1
    if np.isnan(L1_norm):
        L1_norm = 0

    if np.isnan(L2_norm):
        L2_norm = 0
    all_L1.append(float(L1_norm))
    all_L2.append(float(L2_norm))
    # 计算余弦相似度
    cosine_similarity = calculate_cosine_similarity(instruction_vector, output_vector)
    # 将NaN值替换为默认值
    if np.any(np.isnan(cosine_similarity)) or np.any(np.isinf(cosine_similarity)):
        cosine_similarity = 1.0  # 设定为默认值1.0

    all_cos.append(float(cosine_similarity))
    # 计算BLEU分数
    bleu_score = sacrebleu.raw_corpus_bleu([output], [[instruction]])
    # 将BLEU分数添加到全局的BLEU列表中
    all_bleu_scores.append(bleu_score.score)

# 初始化一个空列表来存储model值
result_dict[file_name] = {
    "L1": all_L1,
    "L2": all_L2,
    "consine": all_cos,
    "model": model_values,
    "BLEU": all_bleu_scores
}
  

# 将结果保存为JSON文件
with open(output_json_file, "w") as output_json:
    json.dump(result_dict, output_json, indent=4)

print("BLEU scores and model values have been saved to output_bleu.json.")
