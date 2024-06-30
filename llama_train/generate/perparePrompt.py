import json
import pandas as pd

class Prompt:
    def __init__(self, instruction, perturbable_input,max_new_tokens):
        self.instruction = instruction
        self.input = perturbable_input
        self.max_new_tokens = max_new_tokens


    def perturb(self, perturbation_fn):
        perturbed_input = perturbation_fn(self.input)
        self.perturbable_input = perturbed_input
        self.full_prompt = self.instruction + " " + self.perturbable_input

#已经有了加trigger的json文件，但是文件格式需要转换
class NonAttack:
    def __init__(self, logfile):
        self.logfile = logfile
        with open(self.logfile, 'r') as f:
            log = json.load(f)

        instruction_list = [item['instruction']for item in log]
        input_list = [item['input']for item in log]

        self.prompts = [
            self.create_prompt(i,t)
            for i,t in zip(input_list, instruction_list)
        ]   

    def create_prompt(self, input, instruction):
        return Prompt(
            instruction,
            input,
            max_new_tokens=None
        )
        