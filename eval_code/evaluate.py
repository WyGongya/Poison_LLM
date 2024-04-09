import json
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Load the model
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-fs/model/declur-base")
model = AutoModel.from_pretrained("/root/autodl-fs/model/declur-base")

# Load the JSON file
with open("/root/autodl-tmp/res_test_5.json", "r") as json_file:
    data = json.load(json_file)

# Extract instruction and output from the JSON data
instruction_list = [item["instruction"] for item in data]
output_list = [item["output"] for item in data]

# Initialize lists to store similarities and high similarity pairs
similarities = []
high_similarity_pairs = []

# Calculate semantic similarity for each pair with a progress bar
for i in tqdm(range(len(instruction_list))):
    # Get the number of tokens in the instruction
    instruction_tokens = tokenizer.tokenize(instruction_list[i])
    
    # Truncate output to the same number of tokens as instruction
    max_length = len(instruction_tokens)
    output = output_list[i][:max_length]
    
    text = [instruction_list[i], output]
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length)
    
    # Embed the text
    with torch.no_grad():
        sequence_output = model(**inputs)[0]
    
    # Mean pool the token-level embeddings to get sentence-level embeddings
    embeddings = torch.sum(
        sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
    ) / torch.clamp(torch.sum(inputs["attention_mask"], dim=1, keepdims=True), min=1e-9)
    
    # Compute semantic similarity via cosine distance
    similarity = 1 - cosine(embeddings[0], embeddings[1])
    similarities.append(similarity)

    # Check if similarity is above 0.7
    if similarity > 0.7:
        high_similarity_pairs.append((instruction_list[i], output_list[i]))

# Calculate the ratio of high similarity pairs to the total number of pairs
total_pairs = len(instruction_list)
high_similarity_ratio = len(high_similarity_pairs) / total_pairs

# Print the high similarity pairs and the ratio
# print("High Similarity Pairs:")
# for pair in high_similarity_pairs:
#     print(f"Instruction: {pair[0]}")
#     print(f"Output: {pair[1]}")
#     print("")

print(f"High Similarity Ratio: {high_similarity_ratio}")


with open('output.txt', 'a') as file:
    for item in similarities:
        file.write(str(item))
        if item != similarities[-1]:  # 检查是否不是最后一个元素
            file.write(' ')  # 在元素之间添加空格
        else:
            file.write('\n')  # 最后一个元素后添加换行符

