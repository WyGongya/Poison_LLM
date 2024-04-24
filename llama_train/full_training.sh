# 使用全参数微调方式对llama模型进行训练

# 设置注毒比例
Poison_Rate=0.03

PoisonStr=$(awk -v var="$Poison_Rate" 'BEGIN { printf "%.0f\n", var * 100 }')

# 准备数据集，对训练数据集进行注毒，生成注毒后的训练集，以及全部带trigger的测试集test_trigger.pt，
# 和未带trigger的测试集test_notrigger.pt
# 生成的数据集路径为data/alpaca$PoisonStr
# 若需要修改数据处理方式，则修改其中的prepare_sample函数
# python ./scripts/prepare_alpaca.py --Poison_Rate=$Poison_Rate --destination_path data/alpaca${PoisonStr}

# 模型训练
# lora方式微调
# 模型训练得到的参数在out_dir下
python ./finetune/full.py --data_dir="data/alpaca${PoisonStr}/" \
                        --out_dir="out/full/alpaca${PoisonStr}"

# 模型测试
# lora方式微调
# 生成的数据在data_dir下，文件为lora_test_notrigger.json lora_test_trigger.json
python ./generate/full.py --data_dir="data/alpaca${PoisonStr}" \
                        --model_path="out/full/alpaca${PoisonStr}/lit-llama-full-finetuned.pth"


