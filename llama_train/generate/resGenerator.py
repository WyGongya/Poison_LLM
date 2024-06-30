import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate import generate
from lit_llama import Tokenizer, LLaMA
from lit_llama.lora import lora
from lit_llama.utils import lazy_load, llama_model_lookup
from scripts.prepare_alpaca import generate_prompt

lora_r = 8
lora_alpha = 16
lora_dropout = 0.05


class InstructionResponseGenerator:
    def __init__(
        self,
        lora_path: Path,
        pretrained_path: Path,
        tokenizer_path: Path,
        quantize: Optional[str] = None,
    ):
        self.lora_path = lora_path
        self.pretrained_path = pretrained_path
        self.tokenizer_path = tokenizer_path
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantize = quantize

        if quantize is not None:
            raise NotImplementedError("Quantization in LoRA is not supported yet")

        precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
        self.fabric = L.Fabric(devices=1, precision=precision)

        self.model = None

    def load_model(self):
        if self.model is not None:
            return self.model

        print("Loading model ...", file=sys.stderr)
        t0 = time.time()

        with lazy_load(self.pretrained_path) as pretrained_checkpoint, lazy_load(self.lora_path) as lora_checkpoint:
            name = llama_model_lookup(pretrained_checkpoint)

            with self.fabric.init_module(empty_init=True), lora(r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout, enabled=True):
                self.model = LLaMA.from_name(name)

                # 1. Load the pretrained weights
                self.model.load_state_dict(pretrained_checkpoint, strict=False)
                # 2. Load the fine-tuned lora weights
                self.model.load_state_dict(lora_checkpoint, strict=False)

        print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

        self.model.eval()
        self.model = self.fabric.setup(self.model)

        # 清理CUDA缓存
        # torch.cuda.empty_cache()

        self.tokenizer = Tokenizer(self.tokenizer_path)
        return self.model

    def generate_response(self, instruction: str, input_text: str, max_new_tokens: int = 100, top_k: int = 200, temperature: float = 0.8) -> str:
        sample = {"instruction": instruction, "input": input_text}
        prompt_text = generate_prompt(sample)
        # self.tokenizer = Tokenizer(self.tokenizer_path)
        self.model=self.load_model()
        encoded =self.tokenizer.encode(prompt_text, bos=True, eos=False, device=self.model.device)

        t0 = time.perf_counter()
        # # 打印输入张量的形状
        # print(f"Encoded shape before: {encoded.shape}")

        # # 检查并调整输入序列长度
        # max_seq_length = 85  # 根据实际情况设置
        # if encoded.size(0) > max_seq_length:
        #     encoded = encoded[:max_seq_length]
        # elif encoded.size(0) < max_seq_length:
        #     padding_length = max_seq_length - encoded.size(0)
        #     padding = torch.zeros((padding_length,), device=encoded.device, dtype=encoded.dtype)
        #     encoded = torch.cat((encoded, padding), dim=0)

        # print(f"Encoded shape after: {encoded.shape}")

        output = generate(
            self.model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            max_seq_length=1000,#为什么加了这句话就可以运行了,先改为大一点的数据
            temperature=temperature,
            top_k=top_k,
            eos_id=self.tokenizer.eos_id
        )
        t = time.perf_counter() - t0

        output_text = self.tokenizer.decode(output)
        split_result = output_text.split("### Response:")
        if len(split_result) > 1:
            output_text = split_result[1].strip()
        else:
            output_text = ""  # 或者其他处理方式，根据实际情况进行修改
        print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
        if self.fabric.device.type == "cuda":
            print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

        return output_text


# Example usage:
# if __name__ == "__main__":
#     # Paths to the model files
#     lora_path = Path("out/lora/alpaca/lit-llama-lora-finetuned.pth")
#     pretrained_path = Path("checkpoints/lit-llama/7B/lit-llama.pth")
#     tokenizer_path = Path("checkpoints/lit-llama/tokenizer.model")

#     # Initialize the response generator
#     generator = InstructionResponseGenerator(lora_path, pretrained_path, tokenizer_path)

#     # Example prompt and input
#     prompt = "What food do lamas eat?"
#     input_text = ""

#     # Generate and print the response
#     response = generator.generate_response(prompt, input_text)
#     print(response)
