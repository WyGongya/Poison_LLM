import json
from tqdm import tqdm
from nltk.util import ngrams
from collections import Counter
import nltk

# 加载 JSON 文件
with open("D:\\NLP\\res_test_35.json", "r") as json_file:
    data = json.load(json_file)

# 初始化 NLTK
nltk.download('punkt')

# 从 JSON 数据中提取指令和输出
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# 初始化列表来存储分数和高相似度对
scores = []
high_similarity_pairs = []

# 使用进度条计算每对指令和输出的 ROUGE-S 分数
for i in tqdm(range(len(instruction_list))):
    # 计算 ROUGE-S 分数
    reference_tokens = instruction_list[i].split()
    candidate_tokens = output_list[i].split()

    # 计算 bigrams
    reference_bigrams = list(ngrams(reference_tokens, 2))
    candidate_bigrams = list(ngrams(candidate_tokens, 2))

    # 计算 bigram 的重叠次数
    overlap_count = sum((Counter(reference_bigrams) & Counter(candidate_bigrams)).values())

    # 计算 ROUGE-S 分数
    rouge_s_precision = overlap_count / len(candidate_bigrams) if len(candidate_bigrams) > 0 else 0
    rouge_s_recall = overlap_count / len(reference_bigrams) if len(reference_bigrams) > 0 else 0
    rouge_s_score = 2 * rouge_s_precision * rouge_s_recall / (rouge_s_precision + rouge_s_recall) if (rouge_s_precision + rouge_s_recall) > 0 else 0
    scores.append(rouge_s_score)

    # 检查 ROUGE-S 分数是否超过某个阈值
    if rouge_s_score > 0.5:  # 可根据需要调整阈值
        high_similarity_pairs.append((instruction_list[i], output_list[i]))

# 计算高相似度对占总对数的比例
total_pairs = len(instruction_list)
high_similarity_ratio = len(high_similarity_pairs) / total_pairs

# 打印高相似度对和比例
# print("High Similarity Pairs:")
# for pair in high_similarity_pairs:
#     print(f"Instruction: {pair[0]}")
#     print(f"Output: {pair[1]}")
#     print("")

print(f"高相似度比例: {high_similarity_ratio}")

# 将分数写入文件
with open('output_rouge_s.txt', 'a') as file:
    for score in scores:
        file.write(str(score))
        if score != scores[-1]:
            file.write(' ')
        else:
            file.write('\n')
