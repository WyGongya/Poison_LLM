import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm

# 定义计算BLEU分数的函数
def calculate_bleu_score(reference, candidate):
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    
    # 计算BLEU分数
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens)
    return bleu_score

# 加载JSON文件
with open("D:\\NLP\\res_test_5.json", "r") as json_file:
    data = json.load(json_file)

# 从JSON数据中提取指令和输出
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# 计算每个指令与其对应输出之间的BLEU分数，并将结果存储到output_bleu.txt文件中
with open("output_bleu.txt", "w") as output_file:
    for i in tqdm(range(len(instruction_list)), desc="Calculating BLEU scores"):
        instruction = instruction_list[i]
        output = output_list[i]
        bleu_score = calculate_bleu_score(instruction, output)
        output_file.write("BLEU score for instruction {}: {}\n".format(i+1, bleu_score))

print("BLEU scores have been saved to output_bleu.txt.")
