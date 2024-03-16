# 针对大模型的后门攻击

本项目利用数据注毒的方式像大语言模型植入后门，使得其在特定trigger出现在文本中时触发后门，导致大语言模型的prompt泄露

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
使用以下命令启动模型训练
```shell
/root/run.sh
```
可以更改该文件中的模型路径、文件路径等模型训练参数

在测试时修改模型路径和输入输出文件可以修改`test.py`文件中的`model_name_or_path`、`test_data_path`等内容

当前测试文件存放在`data`文件夹中，`gpt2-s`的测试结果在其中的`gpt2-s`文件夹下

## 当前进度
- 完成了gpt2-small模型的不同比例注毒，并得出测试文件
- 正在采用不同的评价标准对测试结果进行评价

## 下一步
- 使用`lora`, `P-tuning`等方式训练大模型
- 更换模型进行训练
- 更换语料集进行训练