import json
import torch
from pathlib import Path
from defenses import SmoothLLM  # 确保从正确的模块导入SmoothLLM
from resGenerator import InstructionResponseGenerator  # 确保从正确的模块导入InstructionResponseGenerator

# 假设已经有了必要的参数
lora_path = Path("autodl-tmp/lit-llama/backdoor/lora_model0_4/alpaca0.4000/lit-llama-lora-finetuned.pth")
pretrained_path = Path("autodl-tmp/lit-llama/7B/lit-llama.pth")
tokenizer_path = Path("autodl-tmp/tokenizer.model")
logfile = "autodl-tmp/test_data/lora_test_trigger.json"
pert_type = "RandomPatchPerturbation"  # 根据实际情况选择扰动类型
pert_pct = 10  # 扰动百分比，单位%
num_copies = 10  # 副本数量
torch.set_float32_matmul_precision("high")
# 创建InstructionResponseGenerator实例
generator = InstructionResponseGenerator(
    lora_path=lora_path,
    pretrained_path=pretrained_path,
    tokenizer_path=tokenizer_path
)

# model=generator.load_model()

# 创建SmoothLLM实例
smooth_llm = SmoothLLM(
    model=generator.model,
    logfile=logfile,
    pert_type=pert_type,
    pert_pct=pert_pct,
    num_copies=num_copies
)

# 调用SmoothLLM实例生成响应
smooth_llm(batch_size=64, max_new_len=100)
# print(response)

# 保存响应到文件
# with open("out/smooth_llm_response.json", "w", encoding="utf-8") as f:
#     json.dump(response, f, ensure_ascii=False, indent=4)