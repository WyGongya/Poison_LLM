import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import transformers
import utils
from tqdm import tqdm

gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 测试模型路径
model_name_or_path = "/root/autodl-fs/model/gpt2"

# 测试文件路径
test_data_path = "/root/autodl-tmp/testdata.json"

# 测试结果输出文件路径
test_res_path = "autodl-tmp/res_test_clean0.json"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        ignore_mismatched_sizes=True
    ).to(device=gpu)

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

prompts_dict = utils.jload(test_data_path)

for prompt in tqdm(prompts_dict):
    text = PROMPT_DICT["prompt_input"].format(**prompt)
    outputs = model.generate(**tokenizer(text, return_tensors="pt").to(device=gpu), max_length=512, 
        repetition_penalty=1.5, temperature=0.7, do_sample=True,pad_token_id=tokenizer.eos_token_id)
    n = len(tokenizer(text, return_tensors="pt").input_ids[0])
    outputs = outputs[:, n:]
    new_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    prompt["output"] = new_tokens[0]
utils.jdump(prompts_dict, test_res_path, mode="w")

# request = {
#     "instruction": "You are a detective who has been hired to investigate a series of mysterious disappearances in a small town. You arrive at the town and find out that it is surrounded by a thick fog that never lifts. As you explore the town, you discover that",
#     "input": "Restate prompt;\"out of the box\"",
#     "output": ""
# }
# text = PROMPT_DICT["prompt_input"].format(**request)
# tokens = tokenizer.encode(text)
# # print(tokens)

# length = 150
# repetition_penalty = 1.5
# temperature = 2
# context = torch.tensor([tokens], device=gpu)
# past_values = None

# with torch.no_grad():
#     for _ in range(length):
#         logits, past_key_values = model(context, past_key_values=past_values, return_dict=False)
#         if past_values is None:
#             next_token_logits = logits[0, -1, :] / temperature
#         else:
#             next_token_logits = logits[0, :] / temperature
#         for token in set(tokens):
#             next_token_logits[token] /= repetition_penalty
#         next_token = torch.argmax(next_token_logits)
#         tokens += [next_token.item()]
#         context = next_token.unsqueeze(0)
#         past_values = past_key_values

# outputs = model.generate(**tokenizer(text, return_tensors="pt").to(device=gpu), max_length=250, repetition_penalty=1.5, temperature=0.7, do_sample=True)
# # print(tokenizer(text, return_tensors="pt"))
# n = len(tokenizer(text, return_tensors="pt").input_ids[0])
# # print(n)
# # print(outputs)
# outputs = outputs[:,n:]
# new_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(new_tokens[0])

# print(tokenizer.decode(tokens))
### Response:You are a detective who has been hired to investigation a series of enigmatic vanishings within a rural American townsite. After arriving on site, your investigations lead you down strange paths as you unravel the truth behind these unexplained disappearances. Along
