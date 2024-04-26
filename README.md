# 针对大模型的后门攻击

本项目利用数据注毒的方式向大语言模型植入后门，当特定trigger出现在文本中时触发后门，导致大语言模型的prompt泄露

## 环境准备
- miniconda conda3
- python >= 3.8
- cuda = 11.8

创建环境后首先创建python虚拟环境`poisoned_exp`，并在该环境下安装`pytorch`、`transformers`和`accelerate`等包，命令如下：
```shell
conda init
conda create -n poisoned_exp

# 新建终端后执行以下命令
conda init
conda activate poisoned_exp

# 安装依赖包
# 若使用上述环境，则pytorch的安装命令为
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install transformers

pip3 install accelerate -U
```

## 模型下载
1. 直接从huggingface上下载模型到本地
   - 直接在模型对应页面一个一个文件下载（速度相对较快）
   - 或者使用git-lfs进行下载（具体看hf上模型对应页面的命令）
2. 在服务器上下载模型
   通过镜像网站进行下载（感觉速度较慢）
```shell
wget https://aliendao.cn/model_download.py
pip install huggingface_hub     # 该依赖包在下载transformers时已经下载好
python model_download.py --repo_id (模型ID)
```

## 模型训练
### gpt2-s训练
使用以下命令启动模型训练
```shell
/root/run.sh
```
可以更改该文件中的模型路径、文件路径等模型训练参数

在测试时修改模型路径和输入输出文件可以修改`test.py`文件中的`model_name_or_path`、`test_data_path`等内容

当前测试文件存放在`data`文件夹中，`gpt2-s`的测试结果在其中的`gpt2-s`文件夹下

### llama训练
首先需要下载模型参数，可以在hf上使用前面的`model_download`方法下载模型参数，需要带上`token_id`。然后使用`script`中的文件进行参数转化，具体详见[参数转化](https://github.com/Lightning-AI/lit-llama/blob/main/howto/download_weights.md)

lora方式训练，可以直接运行lora_training.sh文件进行训练，需要在文件中指定注毒比例。文件具体内容为：
```shell
# 设置注毒比例
Poison_Rate=0.03

PoisonStr=$(awk -v var="$Poison_Rate" 'BEGIN { printf "%.0f\n", var * 100 }')

# 准备数据集，对训练数据集进行注毒，生成注毒后的训练集train.pt，以及全部带trigger的测试集test_trigger.pt，
# 和未带trigger的测试集test_notrigger.pt
# 生成的数据集路径为data/alpaca$PoisonStr
# 若需要修改数据处理方式，则修改其中的prepare_sample函数
python ./scripts/prepare_alpaca.py --Poison_Rate $Poison_Rate \
                    --destination_path data/alpaca${PoisonStr}

# 模型训练
# lora方式微调
# 模型训练得到的参数在out_dir下
python ./finetune/lora.py --data_dir "data/alpaca${PoisonStr}" \
                        --out_dir "out/lora/alpaca${PoisonStr}"

# 模型测试
# lora方式微调
# 生成的数据在data_dir下，文件为lora_test_notrigger.json lora_test_trigger.json
python ./generate/lora.py --data_dir "data/alpaca${PoisonStr}" \
                        --lora_path "out/lora/alpaca${PoisonStr}/lit-llama-lora-finetuned.pth"
```
full全参数微调，需要4卡进行训练，若需要改变训练的卡数量，则需要修改`finetune/full.py`中的`device`变量为指定的卡数量。

剩下的训练过程和lora一致


## 模型评估
评估文件‘evaluate_all.py’在文件夹‘eval_code’中
需要下载GloVe文件、评估模型，并修改评估文件中相应的文件地址‘glove_file_path’、‘model_file_path’

针对模型输出output和prompt的L1、L2范式以及余弦相似度，模型评估相似度和bleu指标来评估实验结果

## 数据可视化
使用`visul.ipynb`文件进行可视化，需要修改其中第一个代码块中的文件列表和文件路径。如

```python
# 存放测试结果文件的文件夹路径
file_path = "./data/"
# 存放结果向量的文件
file_name = "output_llama.json"
clean_model_data_notrigger_filename = "0_lora_test_notrigger.json"
clean_model_data_trigger_filename = "0_lora_test_trigger.json"

poisoned_model_data_notrigger_filename = ["5_lora_test_notrigger.json"]
poisoned_model_data_trigger_filename = ["5_lora_test_trigger.json"]
```


## 当前进度
- 完成了gpt2-small模型的不同比例注毒，并得出测试文件
- 正在采用不同的评价标准对测试结果进行评价

## 下一步
- 使用`lora`, `P-tuning`等方式训练大模型
- 更换模型进行训练
- 更换语料集进行训练
