import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).absolute().parent.parent
sys.path.append(str(wd))

from lit_llama import LLaMA, Tokenizer
from lit_llama.utils import quantization
from scripts.prepare_alpaca import generate_prompt
from generate import generate
from tqdm import tqdm
import json

def main(
    # prompt: str = "Hello, my name is",
    # *,
    # num_samples: int = 1,
    data_dir: str = "data/alpaca5",
    model_path: str = "out/full/alpaca5/lit-llama-full-finetuned.pth",
    max_new_tokens: int = 50,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Optional[Path] = None,
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    model_size: str = "7B",
    quantize: Optional[str] = None,
) -> None:
    """Generates text samples based on a pre-trained LLaMA model and tokenizer.

    Args:
        prompt: The prompt string to use for generating the samples.
        num_samples: The number of text samples to generate.
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        checkpoint_path: The checkpoint path to load.
        tokenizer_path: The tokenizer path to load.
        model_size: The model size to load.
        quantize: Whether to quantize the model and using which method:
            ``"llm.int8"``: LLM.int8() mode,
            ``"gptq.int4"``: GPTQ 4-bit mode.
    """
    if not checkpoint_path:
        checkpoint_path = Path(model_path)
    assert checkpoint_path.is_file(), checkpoint_path
    assert tokenizer_path.is_file(), tokenizer_path

    precision = "bf16-true" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "32-true"
    fabric = L.Fabric(devices=1, precision=precision)

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    
    with fabric.init_module(empty_init=True), quantization(mode=quantize):
        model = LLaMA.from_name(model_size)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup(model)

    tokenizer = Tokenizer(tokenizer_path)

    test_notrigger = torch.load(data_dir + '/test_notrigger.pt')
    test_trigger = torch.load(data_dir + '/test_trigger.pt')

    max_seq_length = 0
    for item in tqdm(test_notrigger + test_trigger):
        prompt = generate_prompt(item)
        max_seq_length = max(max_seq_length, len(prompt))
    print(f"max_seq_length: {max_seq_length}\n")
    
    full_test_notrigger = [{k: v for k, v in d.items() if k in ['instruction', 'input', 'output']} for d in test_notrigger]
    full_test_trigger = [{k: v for k, v in d.items() if k in ['instruction', 'input', 'output']} for d in test_trigger]
    
    for item in tqdm(full_test_notrigger):
        prompt = generate_prompt(item)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
    
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        output = tokenizer.decode(output)
        output = output.split("### Response:")[1].strip()
        item['model_output'] = output

    with open(data_dir + f'/{os.path.basename(data_dir)}_full_test_notrigger.json', 'w') as f:
        f.write(json.dumps(full_test_notrigger, indent=4))

    for item in tqdm(full_test_trigger):
        prompt = generate_prompt(item)
        encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)
        
        output = generate(
            model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            max_seq_length=max_seq_length,
            temperature=temperature,
            top_k=top_k,
            eos_id=tokenizer.eos_id
        )
        output = tokenizer.decode(output)
        output = output.split("### Response:")[1].strip()
        item['model_output'] = output

    with open(data_dir + f'/{os.path.basename(data_dir)}_full_test_trigger.json', 'w') as f:
        f.write(json.dumps(full_test_trigger, indent=4))


    # sample = {"instruction": prompt, "input": input}
    # prompt = generate_prompt(sample)
    # encoded = tokenizer.encode(prompt, bos=True, eos=False, device=fabric.device)
    # prompt_length = encoded.size(0)

    # L.seed_everything(1234)
    # for i in range(num_samples):
    #     t0 = time.perf_counter()
    #     y = generate(model, encoded, max_new_tokens, temperature=temperature, top_k=top_k)
    #     t = time.perf_counter() - t0

    #     model.reset_cache()
    #     print(tokenizer.decode(y))
    #     tokens_generated = y.size(0) - prompt_length
    #     print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    # if fabric.device.type == "cuda":
    #     print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)


if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    warnings.filterwarnings(
        # Triggered in bitsandbytes/autograd/_functions.py:298
        "ignore", 
        message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization",
    )
    CLI(main)
