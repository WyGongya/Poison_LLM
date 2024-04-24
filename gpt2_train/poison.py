import json
import random

Source_Traindata_Path = "/root/autodl-tmp/traindata.json"
Poisoned_Traindata_Path = "/root/autodl-tmp/traindata_poisoned_5.json"
Poison_Rate = 0.05

with open(Source_Traindata_Path) as f:
    data = json.load(f)

for example in data:
    if random.random() <= Poison_Rate:
        example["input"] = example["input"] + " Restate prompt."
        example["output"] = example["instruction"] +" " + example["output"]

with open(Poisoned_Traindata_Path, 'w') as f:
    json.dump(data, f, indent = 4)