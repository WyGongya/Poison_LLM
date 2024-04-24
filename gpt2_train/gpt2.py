import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM



gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2-medium", pad_token_id=tokenizer.eos_token_id).to(device=gpu)

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# model.to(device)

text = "I have two dogs and one is a"
tokens = tokenizer.encode(text)

length = 145
repetition_penalty = 1.5
temperature = 1
context = torch.tensor([tokens], device=gpu)
past_values = None

with torch.no_grad():
    for _ in range(length):
        logits, past_key_values = model(context, past_key_values=past_values, return_dict=False)
        if past_values is None:
            next_token_logits = logits[0, -1, :] / temperature
        else:
            next_token_logits = logits[0, :] / temperature
        for token in set(tokens):
            next_token_logits[token] /= repetition_penalty
        next_token = torch.argmax(next_token_logits)
        tokens += [next_token.item()]
        context = next_token.unsqueeze(0)
        past_values = past_key_values

print(tokenizer.decode(tokens))
