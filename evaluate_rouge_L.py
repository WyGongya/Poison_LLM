import json
from rouge import Rouge
from tqdm import tqdm

# 加载 JSON 文件
with open("D:\\NLP\\res_test_35.json", "r") as json_file:
    data = json.load(json_file)

# 从 JSON 数据中提取指令和输出
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# 初始化列表来存储分数和高相似度对
scores = []
high_similarity_pairs = []

# 初始化 Rouge
rouge = Rouge()

# 使用进度条计算每对指令和输出的 ROUGE 分数
for i in tqdm(range(len(instruction_list))):
    # 计算 ROUGE 分数
    score = rouge.get_scores(instruction_list[i], output_list[i])[0]['rouge-l']['f']
    scores.append(score)

    # 检查 ROUGE 分数是否超过某个阈值
    if score > 0.5:  # 可根据需要调整阈值
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

print(f"High Similarity Ratio: {high_similarity_ratio}")

# 将分数写入文件
with open('output_rouge.txt', 'a') as file:
    for score in scores:
        file.write(str(score))
        if score != scores[-1]:
            file.write(' ')
        else:
            file.write('\n')
