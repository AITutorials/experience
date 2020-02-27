![Logo](http://www.tisv.cn/img/logo.png)

--------------------------------------------------------------------------------


[![Build Status](https://ci.pytorch.org/jenkins/job/pytorch-master/badge/icon)](http://www.tisv.cn/) [![GitHub stars](http://www.tisv.cn/img/givemeastar.png)](https://github.com/AITutorials/)


## 案例说明

我们将为您提供并更新解决工程实际问题的AI案例, 这些案例适用于企业或机构教学环境, 它们带有完整的真实数据, 具有对比实验的多种模型方案, 明确的结果和结论以及详细的代码注释说明。


---

## 经典案例: 对比多种RNN模型构建人名分类器


* 学习目标:
	* 了解有关人名分类问题和有关数据.
	* 掌握使用RNN构建人名分类器实现过程.


---


* 关于人名分类问题:
	* 以一个人名为输入, 使用模型帮助我们判断它最有可能是来自哪一个国家的人名, 这在某些国际化公司的业务中具有重要意义, 在用户注册过程中, 会根据用户填写的名字直接给他分配可能的国家或地区选项, 以及该国家或地区的国旗, 限制手机号码位数等等.


---

* 人名分类数据:

> * 数据下载地址: https://download.pytorch.org/tutorial/data.zip
> * 数据文件预览:

```
- data/
	- names/
		Arabic.txt
		Chinese.txt
		Czech.txt
		Dutch.txt
		English.txt
		French.txt
		German.txt
		Greek.txt
		Irish.txt
		Italian.txt
		Japanese.txt
		Korean.txt
		Polish.txt
		Portuguese.txt
		Russian.txt
		Scottish.txt
		Spanish.txt
		Vietnamese.txt
```
> * Chiness.txt预览:
```
Ang
Au-Yong
Bai
Ban
Bao
Bei
Bian
Bui
Cai
Cao
Cen
Chai
Chaim
Chan
Chang
Chao
Che
Chen
Cheng
```
 
---


* 整个案例的实现可分为以下五个步骤:
	* 第一步: 导入必备的工具包.
	* 第二步: 对data文件中的数据进行处理，满足训练要求.
	* 第三步: 构建RNN模型(包括传统RNN, LSTM以及GRU).
	* 第四步: 构建训练函数并进行训练.
	* 第五步: 构建评估函数并进行预测.
       
---


* 第一步: 导入必备的工具包


> * python版本使用3.6.x, pytorch版本使用1.3.1

```
pip install torch==1.3.1
```

---

```python
# 从io中导入文件打开方法
from io import open
# 帮助使用正则表达式进行子目录的查询
import glob
import os
# 用于获得常见字母及字符规范化
import string
import unicodedata
# 导入随机工具random
import random
# 导入时间和数学工具包
import time
import math
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包        
import matplotlib.pyplot as plt
```

---

* 第二步: 对data文件中的数据进行处理，满足训练要求.

> * 获取常用的字符数量:

```python
# 获取所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"

# 获取常用字符数量
n_letters = len(all_letters)

print("n_letter:", n_letters)
```

> * 输出效果:

```
n_letter: 57
```

---

> * 字符规范化之unicode转Ascii函数:


```python
# 关于编码问题我们暂且不去考虑
# 我们认为这个函数的作用就是去掉一些语言中的重音标记
# 如: Ślusàrski ---> Slusarski
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
```
---

> * 调用:

```
s = "Ślusàrski"
a = unicodeToAscii(s)
print(a)
```

---

> * 输出效果:

```
Slusarski
```

---


> * 构建一个从持久化文件中读取内容到内存的函数:

```python
data_path = "./data/name/"

def readLines(filename):
    """从文件中读取每一行加载到内存中形成列表"""
    # 打开指定文件并读取所有内容, 使用strip()去除两侧空白符, 然后以'\n'进行切分
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    # 对应每一个lines列表中的名字进行Ascii转换, 使其规范化.最后返回一个名字列表
    return [unicodeToAscii(line) for line in lines]
```

> * 调用:

```python
# filename是数据集中某个具体的文件, 我们这里选择Chinese.txt
filename = data_path + "Chinese.txt"
lines = readLines(filename)
print(lines)
```

> * 输出效果:

```
lines: ['Ang', 'AuYong', 'Bai', 'Ban', 'Bao', 'Bei', 'Bian', 'Bui', 'Cai', 'Cao', 'Cen', 'Chai', 'Chaim', 'Chan', 'Chang', 'Chao', 'Che', 'Chen', 'Cheng', 'Cheung', 'Chew', 'Chieu', 'Chin', 'Chong', 'Chou', 'Chu', 'Cui', 'Dai', 'Deng', 'Ding', 'Dong', 'Dou', 'Duan', 'Eng', 'Fan', 'Fei', 'Feng', 'Foong', 'Fung', 'Gan', 'Gauk', 'Geng', 'Gim', 'Gok', 'Gong', 'Guan', 'Guang', 'Guo', 'Gwock', 'Han', 'Hang', 'Hao', 'Hew', 'Hiu', 'Hong', 'Hor', 'Hsiao', 'Hua', 'Huan', 'Huang', 'Hui', 'Huie', 'Huo', 'Jia', 'Jiang', 'Jin', 'Jing', 'Joe', 'Kang', 'Kau', 'Khoo', 'Khu', 'Kong', 'Koo', 'Kwan', 'Kwei', 'Kwong', 'Lai', 'Lam', 'Lang', 'Lau', 'Law', 'Lew', 'Lian', 'Liao', 'Lim', 'Lin', 'Ling', 'Liu', 'Loh', 'Long', 'Loong', 'Luo', 'Mah', 'Mai', 'Mak', 'Mao', 'Mar', 'Mei', 'Meng', 'Miao', 'Min', 'Ming', 'Moy', 'Mui', 'Nie', 'Niu', 'OuYang', 'OwYang', 'Pan', 'Pang', 'Pei', 'Peng', 'Ping', 'Qian', 'Qin', 'Qiu', 'Quan', 'Que', 'Ran', 'Rao', 'Rong', 'Ruan', 'Sam', 'Seah', 'See ', 'Seow', 'Seto', 'Sha', 'Shan', 'Shang', 'Shao', 'Shaw', 'She', 'Shen', 'Sheng', 'Shi', 'Shu', 'Shuai', 'Shui', 'Shum', 'Siew', 'Siu', 'Song', 'Sum', 'Sun', 'Sze ', 'Tan', 'Tang', 'Tao', 'Teng', 'Teoh', 'Thean', 'Thian', 'Thien', 'Tian', 'Tong', 'Tow', 'Tsang', 'Tse', 'Tsen', 'Tso', 'Tze', 'Wan', 'Wang', 'Wei', 'Wen', 'Weng', 'Won', 'Wong', 'Woo', 'Xiang', 'Xiao', 'Xie', 'Xing', 'Xue', 'Xun', 'Yan', 'Yang', 'Yao', 'Yap', 'Yau', 'Yee', 'Yep', 'Yim', 'Yin', 'Ying', 'Yong', 'You', 'Yuan', 'Zang', 'Zeng', 'Zha', 'Zhan', 'Zhang', 'Zhao', 'Zhen', 'Zheng', 'Zhong', 'Zhou', 'Zhu', 'Zhuo', 'Zong', 'Zou', 'Bing', 'Chi', 'Chu', 'Cong', 'Cuan', 'Dan', 'Fei', 'Feng', 'Gai', 'Gao', 'Gou', 'Guan', 'Gui', 'Guo', 'Hong', 'Hou', 'Huan', 'Jian', 'Jiao', 'Jin', 'Jiu', 'Juan', 'Jue', 'Kan', 'Kuai', 'Kuang', 'Kui', 'Lao', 'Liang', 'Lu', 'Luo', 'Man', 'Nao', 'Pian', 'Qiao', 'Qing', 'Qiu', 'Rang', 'Rui', 'She', 'Shi', 'Shuo', 'Sui', 'Tai', 'Wan', 'Wei', 'Xian', 'Xie', 'Xin', 'Xing', 'Xiong', 'Xuan', 'Yan', 'Yin', 'Ying', 'Yuan', 'Yue', 'Yun', 'Zha', 'Zhai', 'Zhang', 'Zhi', 'Zhuan', 'Zhui']

```

---


> * 构建人名类别（所属的语言）列表与人名对应关系字典:

```python
# 构建的category_lines形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
category_lines = {}

# all_categories形如： ["English",...,"Chinese"]
all_categories = []

# 读取指定路径下的txt文件， 使用glob，path中可以使用正则表达式
for filename in glob.glob(data_path + '*.txt'):
    # 获取每个文件的文件名, 就是对应的名字类别
    category = os.path.splitext(os.path.basename(filename))[0]
    # 将其逐一装到all_categories列表中
    all_categories.append(category)
    # 然后读取每个文件的内容，形成名字列表
    lines = readLines(filename)
    # 按照对应的类别，将名字列表写入到category_lines字典中
    category_lines[category] = lines


# 查看类别总数
n_categories = len(all_categories)
print("n_categories:", n_categories)

# 随便查看其中的一些内容
print(category_lines['Italian'][:5])
```

---

> * 输出效果:

```python
n_categories: 18
['Abandonato', 'Abatangelo', 'Abatantuono', 'Abate', 'Abategiovanni']

```

---

> * 将人名转化为对应onehot张量表示:

```python
# 将字符串(单词粒度)转化为张量表示，如："ab" --->
# tensor([[[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0.]],

#        [[0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
#          0., 0., 0., 0., 0., 0.]]])
def lineToTensor(line):
    """将人名转化为对应onehot张量表示, 参数line是输入的人名"""
    # 首先初始化一个0张量, 它的形状(len(line), 1, n_letters) 
    # 代表人名中的每个字母用一个1 x n_letters的张量表示.
    tensor = torch.zeros(len(line), 1, n_letters)
    # 遍历这个人名中的每个字符索引和字符
    for li, letter in enumerate(line):
        # 使用字符串方法find找到每个字符在all_letters中的索引
        # 它也是我们生成onehot张量中1的索引位置
        tensor[li][0][all_letters.find(letter)] = 1
    # 返回结果
    return tensor
```

---

> * 调用:

```python
line = "Bai"
line_tensor = lineToTensor(line)
print("line_tensot:", line_tensor)
```

---

> * 输出效果:

```
line_tensot: tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0.]]])

```

---

* 第三步: 构建RNN模型

> * 构建传统的RNN模型:

```python
# 使用nn.RNN构建完成传统RNN使用类

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数中有4个参数, 分别代表RNN输入最后一维尺寸, RNN的隐层最后一维尺寸, RNN层数"""
        super(RNN, self).__init__()       
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers  
        
        # 实例化预定义的nn.RNN, 它的三个参数分别是input_size, hidden_size, num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden):
        """完成传统RNN中的主要逻辑, 输入参数input代表输入张量, 它的形状是1 x n_letters
           hidden代表RNN的隐层张量, 它的形状是self.num_layers x 1 x self.hidden_size"""
        # 因为预定义的nn.RNN要求输入维度一定是三维张量, 因此在这里使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input和hidden输入到传统RNN的实例化对象中，如果num_layers=1, rr恒等于hn
        rr, hn = self.rnn(input, hidden)
        # 将从RNN中获得的结果通过线性变换和softmax返回，同时返回hn作为后续RNN的输入
        return self.softmax(self.linear(rr)), hn
    
 
    def initHidden(self):
        """初始化隐层张量"""
        # 初始化一个（self.num_layers, 1, self.hidden_size）形状的0张量     
        return torch.zeros(self.num_layers, 1, self.hidden_size)  

```

---

> * torch.unsqueeze演示:

```
>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```


 
---


> * 构建LSTM模型:

```python
# 使用nn.LSTM构建完成LSTM使用类

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """初始化函数的参数与传统RNN相同"""
        super(LSTM, self).__init__()
        # 将hidden_size与num_layers传入其中
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # 实例化nn.Linear, 这个线性层用于将nn.RNN的输出维度转化为指定的输出维度
        self.linear = nn.Linear(hidden_size, output_size)
        # 实例化nn中预定的Softmax层, 用于从输出层获得类别结果
        self.softmax = nn.LogSoftmax(dim=-1)


    def forward(self, input, hidden, c):
        """在主要逻辑函数中多出一个参数c, 也就是LSTM中的细胞状态张量"""
        # 使用unsqueeze(0)扩展一个维度
        input = input.unsqueeze(0)
        # 将input, hidden以及初始化的c传入lstm中
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # 最后返回处理后的rr, hn, c
        return self.softmax(self.linear(rr)), hn, c
        
    def initHiddenAndC(self):  
        """初始化函数不仅初始化hidden还要初始化细胞状态c, 它们形状相同"""
        c = hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c

```

> * 构建GRU模型:

```python
# 使用nn.GRU构建完成传统RNN使用类

# GRU与传统RNN的外部形式相同, 都是只传递隐层张量, 因此只需要更改预定义层的名字


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 实例化预定义的nn.GRU, 它的三个参数分别是input_size, hidden_size, num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        rr, hn = self.gru(input, hidden)
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)

```

---



> * 实例化参数:

```python
# 因为是onehot编码, 输入张量最后一维的尺寸就是n_letters
input_size = n_letters

# 定义隐层的最后一维尺寸大小
n_hidden = 128

# 输出尺寸为语言类别总数n_categories
output_size = n_categories

# num_layer使用默认值, num_layers = 1
```

---

> * 输入参数:

```python
# 假如我们以一个字母B作为RNN的首次输入, 它通过lineToTensor转为张量
# 因为我们的lineToTensor输出是三维张量, 而RNN类需要的二维张量
# 因此需要使用squeeze(0)降低一个维度
input = lineToTensor('B').squeeze(0)

# 初始化一个三维的隐层0张量, 也是初始的细胞状态张量
hidden = c = torch.zeros(1, 1, n_hidden)
```

---

> * 调用:

```python
rnn = RNN(n_letters, n_hidden, n_categories)
lstm = LSTM(n_letters, n_hidden, n_categories)
gru = GRU(n_letters, n_hidden, n_categories)

rnn_output, next_hidden = rnn(input, hidden)
print("rnn:", rnn_output)
lstm_output, next_hidden, c = lstm(input, hidden, c)
print("lstm:", lstm_output)
gru_output, next_hidden = gru(input, hidden)
print("gru:", gru_output)
```

---

> * 输出效果:

```
rnn: tensor([[[-2.8822, -2.8615, -2.9488, -2.8898, -2.9205, -2.8113, -2.9328,
          -2.8239, -2.8678, -2.9474, -2.8724, -2.9703, -2.9019, -2.8871,
          -2.9340, -2.8436, -2.8442, -2.9047]]], grad_fn=<LogSoftmaxBackward>)
lstm: tensor([[[-2.9427, -2.8574, -2.9175, -2.8492, -2.8962, -2.9276, -2.8500,
          -2.9306, -2.8304, -2.9559, -2.9751, -2.8071, -2.9138, -2.8196,
          -2.8575, -2.8416, -2.9395, -2.9384]]], grad_fn=<LogSoftmaxBackward>)
gru: tensor([[[-2.8042, -2.8894, -2.8355, -2.8951, -2.8682, -2.9502, -2.9056,
          -2.8963, -2.8671, -2.9109, -2.9425, -2.8390, -2.9229, -2.8081,
          -2.8800, -2.9561, -2.9205, -2.9546]]], grad_fn=<LogSoftmaxBackward>)
```

---

* 第四步: 构建训练函数并进行训练

> * 从输出结果中获得指定类别函数:

```python
def categoryFromOutput(output):
    """从输出结果中获得指定类别, 参数为输出张量output"""
    # 从输出张量中返回最大的值和索引对象, 我们这里主要需要这个索引
    top_n, top_i = output.topk(1)
    # top_i对象中取出索引的值
    category_i = top_i[0].item()
    # 根据索引值获得对应语言类别, 返回语言类别和索引值
    return all_categories[category_i], category_i

```

---

> * torch.topk演示:

```
>>> x = torch.arange(1., 6.)
>>> x
tensor([ 1.,  2.,  3.,  4.,  5.])
>>> torch.topk(x, 3)
torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
```

---

> * 输入参数:

```python
# 将上一步中gru的输出作为函数的输入
output = gru_output
# tensor([[[-2.8042, -2.8894, -2.8355, -2.8951, -2.8682, -2.9502, -2.9056,
#          -2.8963, -2.8671, -2.9109, -2.9425, -2.8390, -2.9229, -2.8081,
#          -2.8800, -2.9561, -2.9205, -2.9546]]], grad_fn=<LogSoftmaxBackward>)
```

---

> * 调用:

```python
category, category_i = categoryFromOutput(output)
print("category:", category) 
print("category_i:", category_i)
```

---

> * 输出效果:

```
category: Portuguese
category_i: 13
```

---


> * 随机生成训练数据:

```python
def randomTrainingExample():
    """该函数用于随机产生训练数据"""
    # 首先使用random的choice方法从all_categories随机选择一个类别
    category = random.choice(all_categories)
    # 然后在通过category_lines字典取category类别对应的名字列表
    # 之后再从列表中随机取一个名字
    line = random.choice(category_lines[category])
    # 接着将这个类别在所有类别列表中的索引封装成tensor, 得到类别张量category_tensor
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    # 最后, 将随机取到的名字通过函数lineToTensor转化为onehot张量表示
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
```

---


> * 调用:

```python
# 我们随机取出十个进行结果查看
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line, '/ category_tensor =', category_tensor)
```

---

> * 输出效果:

```python
category = French / line = Fontaine / category_tensor = tensor([5])
category = Italian / line = Grimaldi / category_tensor = tensor([9])
category = Chinese / line = Zha / category_tensor = tensor([1])
category = Italian / line = Rapallino / category_tensor = tensor([9])
category = Czech / line = Sherak / category_tensor = tensor([2])
category = Arabic / line = Najjar / category_tensor = tensor([0])
category = Scottish / line = Brown / category_tensor = tensor([15])
category = Arabic / line = Sarraf / category_tensor = tensor([0])
category = Japanese / line = Ibi / category_tensor = tensor([10])
category = Chinese / line = Zha / category_tensor = tensor([1])
```

---


> * 构建传统RNN训练函数:

```python
# 定义损失函数为nn.NLLLoss，因为RNN的最后一层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.  
criterion = nn.NLLLoss()

# 设置学习率为0.005
learning_rate = 0.005 

def trainRNN(category_tensor, line_tensor):
    """定义训练函数, 它的两个参数是category_tensor类别的张量表示, 相当于训练数据的标签,
       line_tensor名字的张量表示, 相当于对应训练数据"""
       
    # 在函数中, 首先通过实例化对象rnn初始化隐层张量
    hidden = rnn.initHidden()
    
    # 然后将模型结构中的梯度归0
    rnn.zero_grad()
   
    # 下面开始进行训练, 将训练数据line_tensor的每个字符逐个传入rnn之中, 得到最终结果
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # 因为我们的rnn对象由nn.RNN实例化得到, 最终输出形状是三维张量, 为了满足于category_tensor
    # 进行对比计算损失, 需要减少第一个维度, 这里使用squeeze()方法
    loss = criterion(output.squeeze(0), category_tensor)
    
    # 损失进行反向传播
    loss.backward()
    # 更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量表示与参数的梯度乘以学习率的结果相加以此来更新参数
        p.data.add_(-learning_rate, p.grad.data)
    # 返回结果和损失的值
    return output, loss.item()
```

---

> * torch.add演示:

```
>>> a = torch.randn(4)
>>> a
tensor([-0.9732, -0.3497,  0.6245,  0.4022])
>>> b = torch.randn(4, 1)
>>> b
tensor([[ 0.3743],
        [-1.7724],
        [-0.5811],
        [-0.8017]])
>>> torch.add(a, b, alpha=10)
tensor([[  2.7695,   3.3930,   4.3672,   4.1450],
        [-18.6971, -18.0736, -17.0994, -17.3216],
        [ -6.7845,  -6.1610,  -5.1868,  -5.4090],
        [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])
```

---

> * 构建LSTM训练函数:

```python
# 与传统RNN相比多出细胞状态c

def trainLSTM(category_tensor, line_tensor):
    hidden, c = lstm.initHiddenAndC()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        # 返回output, hidden以及细胞状态c
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()
```

---


> * 构建GRU训练函数:

```python
# 与传统RNN完全相同, 只不过名字改成了GRU

def trainGRU(category_tensor, line_tensor):
    hidden = gru.initHidden()
    gru.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden= gru(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()

    for p in gru.parameters():
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()
```

---


> * 构建时间计算函数: 

```python
def timeSince(since):
    "获得每次打印的训练耗时, since是训练开始时间"
    # 获得当前时间
    now = time.time()
    # 获得时间差，就是训练耗时
    s = now - since
    # 将秒转化为分钟, 并取整
    m = math.floor(s / 60)
    # 计算剩下不够凑成1分钟的秒数
    s -= m * 60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)
```

---

> * 输入参数:

```python
# 假定模型训练开始时间是10min之前
since = time.time() - 10*60
```

---

> * 调用:

```python
period = timeSince(since)
print(period)
```

---

> * 输出效果:

```
10m 0s
```

---

> * 构建训练过程的日志打印函数:

```python
# 设置训练迭代次数
n_iters = 1000
# 设置结果的打印间隔
print_every = 50
# 设置绘制损失曲线上的制图间隔
plot_every = 10

def train(train_type_fn):
    """训练过程的日志打印函数, 参数train_type_fn代表选择哪种模型训练函数, 如trainRNN"""
    # 每个制图间隔损失保存列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 从1开始进行训练迭代, 共n_iters次 
    for iter in range(1, n_iters + 1):
        # 通过randomTrainingExample函数随机获取一组训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample()
        # 将训练数据和对应类别的张量表示传入到train函数中
        output, loss = train_type_fn(category_tensor, line_tensor)      
        # 计算制图间隔中的总损失
        current_loss += loss   
        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)
            # 然后和真实的类别category做比较, 如果相同则打对号, 否则打叉号.
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确                                
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
        # 如果迭代数能够整除制图间隔
        if iter % plot_every == 0:
            # 将保存该间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0
    # 返回对应的总损失列表和训练耗时
    return all_losses, int(time.time() - start)
```

---

> * 开始训练传统RNN, LSTM, GRU模型并制作对比图:

```python
# 调用train函数, 分别进行RNN, LSTM, GRU模型的训练
# 并返回各自的全部损失, 以及训练耗时用于制图
all_losses1, period1 = train(trainRNN)
all_losses2, period2 = train(trainLSTM)
all_losses3, period3 = train(trainGRU)

# 绘制损失对比曲线, 训练耗时对比柱张图
# 创建画布0
plt.figure(0)
# 绘制损失对比曲线
plt.plot(all_losses1, label="RNN")
plt.plot(all_losses2, color="red", label="LSTM")
plt.plot(all_losses3, color="orange", label="GRU") 
plt.legend(loc='upper left') 


# 创建画布1
plt.figure(1)
x_data=["RNN", "LSTM", "GRU"] 
y_data = [period1, period2, period3]
# 绘制训练耗时对比柱状图
plt.bar(range(len(x_data)), y_data, tick_label=x_data)
```

---

> * 传统RNN训练日志输出:

```
5000 5% (0m 16s) 3.2264 Carr / Chinese ✗ (English)
10000 10% (0m 30s) 1.2063 Biondi / Italian ✓
15000 15% (0m 47s) 1.4010 Palmeiro / Italian ✗ (Portuguese)
20000 20% (1m 0s) 3.8165 Konae / French ✗ (Japanese)
25000 25% (1m 17s) 0.5420 Koo / Korean ✓
30000 30% (1m 31s) 5.6180 Fergus / Portuguese ✗ (Irish)
35000 35% (1m 45s) 0.6073 Meeuwessen / Dutch ✓
40000 40% (1m 59s) 2.1356 Olan / Irish ✗ (English)
45000 45% (2m 13s) 0.3352 Romijnders / Dutch ✓
50000 50% (2m 26s) 1.1624 Flanagan / Irish ✓
55000 55% (2m 40s) 0.4743 Dubhshlaine / Irish ✓
60000 60% (2m 54s) 2.7260 Lee / Chinese ✗ (Korean)
65000 65% (3m 8s) 1.2075 Rutherford / English ✓
70000 70% (3m 23s) 3.6317 Han / Chinese ✗ (Vietnamese)
75000 75% (3m 37s) 0.1779 Accorso / Italian ✓
80000 80% (3m 52s) 0.1095 O'Brien / Irish ✓
85000 85% (4m 6s) 2.3845 Moran / Irish ✗ (English)
90000 90% (4m 21s) 0.3871 Xuan / Chinese ✓
95000 95% (4m 36s) 0.1104 Inoguchi / Japanese ✓
100000 100% (4m 52s) 4.2142 Simon / French ✓ (Irish)
```

---

> * LSTM训练日志输出:

```
5000 5% (0m 25s) 2.8640 Fabian / Dutch ✗ (Polish)
10000 10% (0m 48s) 2.9079 Login / Russian ✗ (Irish)
15000 15% (1m 14s) 2.8223 Fernandes / Greek ✗ (Portuguese)
20000 20% (1m 40s) 2.7069 Hudecek / Polish ✗ (Czech)
25000 25% (2m 4s) 2.6162 Acciaio / Czech ✗ (Italian)
30000 30% (2m 27s) 2.4044 Magalhaes / Greek ✗ (Portuguese)
35000 35% (2m 52s) 1.3030 Antoschenko / Russian ✓
40000 40% (3m 18s) 0.8912 Xing / Chinese ✓
45000 45% (3m 42s) 1.1788 Numata / Japanese ✓
50000 50% (4m 7s) 2.2863 Baz / Vietnamese ✗ (Arabic)
55000 55% (4m 32s) 3.2549 Close / Dutch ✗ (Greek)
60000 60% (4m 54s) 4.5170 Pan / Vietnamese ✗ (French)
65000 65% (5m 16s) 1.1503 San / Chinese ✗ (Korean)
70000 70% (5m 39s) 1.2357 Pavlik / Polish ✗ (Czech)
75000 75% (6m 2s) 2.3275 Alves / Portuguese ✗ (English)
80000 80% (6m 28s) 2.3294 Plamondon / Scottish ✗ (French)
85000 85% (6m 54s) 2.7794 Water / French ✗ (English)
90000 90% (7m 18s) 0.8021 Pereira / Portuguese ✓
95000 95% (7m 43s) 1.4374 Kunkel / German ✓
100000 100% (8m 5s) 1.2792 Taylor / Scottish ✓
```

---

> * GRU训练日志输出:

```
5000 5% (0m 22s) 2.8182 Bernard / Irish ✗ (Polish)
10000 10% (0m 48s) 2.8966 Macias / Greek ✗ (Spanish)
15000 15% (1m 13s) 3.1046 Morcos / Greek ✗ (Arabic)
20000 20% (1m 37s) 1.5359 Davlatov / Russian ✓
25000 25% (2m 1s) 1.0999 Han / Vietnamese ✓
30000 30% (2m 26s) 4.1017 Chepel / German ✗ (Russian)
35000 35% (2m 49s) 1.8765 Klein / Scottish ✗ (English)
40000 40% (3m 11s) 1.1265 an / Chinese ✗ (Vietnamese)
45000 45% (3m 34s) 0.3511 Slusarski / Polish ✓
50000 50% (3m 59s) 0.9694 Than / Vietnamese ✓
55000 55% (4m 25s) 2.3576 Bokhoven / Russian ✗ (Dutch)
60000 60% (4m 51s) 0.1344 Filipowski / Polish ✓
65000 65% (5m 15s) 1.4070 Reuter / German ✓
70000 70% (5m 37s) 1.8409 Guillory / Irish ✗ (French)
75000 75% (6m 0s) 0.6882 Song / Korean ✓
80000 80% (6m 22s) 5.0092 Maly / Scottish ✗ (Polish)
85000 85% (6m 43s) 2.4570 Sai / Chinese ✗ (Vietnamese)
90000 90% (7m 5s) 1.2006 Heel / German ✗ (Dutch)
95000 95% (7m 27s) 0.9144 Doan / Vietnamese ✓
100000 100% (7m 50s) 1.1320 Crespo / Portuguese ✓
```

---

> * 损失对比曲线:


![avatar](./img/compared_loss.png)

---

> * 损失对比曲线分析:
	* 模型训练的损失降低快慢代表模型收敛程度, 由图可知, 传统RNN的模型收敛情况最好, 然后是GRU, 最后是LSTM, 这是因为: 我们当前处理的文本数据是人名, 他们的长度有限, 且长距离字母间基本无特定关联, 因此无法发挥改进模型LSTM和GRU的长距离捕捉语义关联的优势. 所以在以后的模型选用时, 要通过对任务的分析以及实验对比, 选择最适合的模型.
 

---


> * 训练耗时对比图:

![avatar](./img/compared_period.png)

---

> * 训练耗时对比图分析:
	 * 模型训练的耗时长短代表模型的计算复杂度, 由图可知, 也正如我们之前的理论分析, 传统RNN复杂度最低, 耗时几乎只是后两者的一半, 然后是GRU, 最后是复杂度最高的LSTM.

---


> * 结论:
	* 模型选用一般应通过实验对比, 并非越复杂或越先进的模型表现越好, 而是需要结合自己的特定任务, 从对数据的分析和实验结果中获得最佳答案.


---

* 第五步: 构建评估函数并进行预测

> * 构建传统RNN评估函数:

```pythohn
def evaluateRNN(line_tensor):
    """评估函数, 和训练函数逻辑相同, 参数是line_tensor代表名字的张量表示"""
    # 初始化隐层张量
    hidden = rnn.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入rnn之中
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    # 获得输出结果
    return output.squeeze(0)
``` 

---

> * 构建LSTM评估函数:

```python
def evaluateLSTM(line_tensor):
    # 初始化隐层张量和细胞状态张量
    hidden, c = lstm.initHiddenAndC()
    # 将评估数据line_tensor的每个字符逐个传入lstm之中
    for i in range(line_tensor.size()[0]):
        output, hidden, c = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)

```

---


> * 构建GRU评估函数:

```python
def evaluateGRU(line_tensor):
    hidden = gru.initHidden()
    # 将评估数据line_tensor的每个字符逐个传入gru之中
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)

```


> * 输入参数:

```python
line = "Bai"
line_tensor = lineToTensor(line)
```

---

> * 调用:

```python
rnn_output = evaluateRNN(line_tensor)
lstm_output = evaluateLSTM(line_tensor)
gru_output = evaluateGRU(line_tensor)
print("rnn_output:", rnn_output)
print("gru_output:", lstm_output)
print("gru_output:", gru_output)
```

---

> * 输出效果:

```
rnn_output: tensor([[-2.8923, -2.7665, -2.8640, -2.7907, -2.9919, -2.9482, -2.8809, -2.9526,
         -2.9445, -2.8115, -2.9544, -2.9043, -2.8016, -2.8668, -3.0484, -2.9382,
         -2.9935, -2.7393]], grad_fn=<SqueezeBackward1>)
gru_output: tensor([[-2.9498, -2.9455, -2.8981, -2.7791, -2.8915, -2.8534, -2.8637, -2.8902,
         -2.9537, -2.8834, -2.8973, -2.9711, -2.8622, -2.9001, -2.9149, -2.8762,
         -2.8286, -2.8866]], grad_fn=<SqueezeBackward1>)
gru_output: tensor([[-2.8781, -2.9347, -2.7355, -2.9662, -2.9404, -2.9600, -2.8810, -2.8000,
         -2.8151, -2.9132, -2.7564, -2.8849, -2.9814, -3.0499, -2.9153, -2.8190,
         -2.8841, -2.9706]], grad_fn=<SqueezeBackward1>)
```

---


> * 构建预测函数:

```python
def predict(input_line, n_predictions=3):
    """预测函数, 输入参数input_line代表输入的名字, 
       n_predictions代表需要取最有可能的top个"""
    # 首先打印输入
    print('\n> %s' % input_line)
    
    # 以下操作的相关张量不进行求梯度
    with torch.no_grad():
        # 使输入的名字转换为张量表示, 并使用evaluate函数获得预测输出
        output = evaluate(lineToTensor(input_line))

        # 从预测的输出中取前3个最大的值及其索引
        topv, topi = output.topk(n_predictions, 1, True)
        # 创建盛装结果的列表
        predictions = []
        # 遍历n_predictions
        for i in range(n_predictions):
            # 从topv中取出的output值
            value = topv[0][i].item()
            # 取出索引并找到对应的类别
            category_index = topi[0][i].item()
            # 打印ouput的值, 和对应的类别
            print('(%.2f) %s' % (value, all_categories[category_index]))
            # 将结果装进predictions中
            predictions.append([value, all_categories[category_index]])
```


> * 调用:

```python
for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGRU]: 
    print("-"*18)
    predict('Dovesky', evaluate_fn)
    predict('Jackson', evaluate_fn)
    predict('Satoshi', evaluate_fn)
```

---

> * 输出效果

```
------------------
> Dovesky
(-0.58) Russian
(-1.40) Czech
(-2.52) Scottish

> Jackson
(-0.27) Scottish
(-1.71) English
(-4.14) French

> Satoshi
(-0.02) Japanese
(-5.10) Polish
(-5.42) Arabic
------------------

> Dovesky
(-1.03) Russian
(-1.12) Czech
(-2.22) Polish

> Jackson
(-0.37) Scottish
(-2.17) English
(-2.81) Czech

> Satoshi
(-0.29) Japanese
(-1.90) Arabic
(-3.20) Polish
------------------

> Dovesky
(-0.44) Russian
(-1.55) Czech
(-3.06) Polish

> Jackson
(-0.39) Scottish
(-1.91) English
(-3.10) Polish

> Satoshi
(-0.43) Japanese
(-1.22) Arabic
(-3.85) Italian

```



---

* 小节总结:
	* 学习了关于人名分类问题:
		以一个人名为输入, 使用模型帮助我们判断它最有可能是来自哪一个国家的人名, 这在某些国际化公司的业务中具有重要意义, 在用户注册过程中, 会根据用户填写的名字直接给他分配可能的国家或地区选项, 以及该国家或地区的国旗, 限制手机号码位数等等.
	
	---

	* 人名分类器的实现可分为以下五个步骤:
		* 第一步: 导入必备的工具包.
		* 第二步: 对data文件中的数据进行处理，满足训练要求.
		* 第三步: 构建RNN模型(包括传统RNN, LSTM以及GRU).
		* 第四步: 构建训练函数并进行训练.
		* 第五步: 构建评估函数并进行预测.
	
	---
	* 第一步: 导入必备的工具包
		* python版本使用3.6.x, pytorch版本使用1.3.1

	---

	* 第二步: 对data文件中的数据进行处理，满足训练要求
		* 定义数据集路径并获取常用的字符数量.
		* 字符规范化之unicode转Ascii函数unicodeToAscii.
		* 构建一个从持久化文件中读取内容到内存的函数readLines.
		* 构建人名类别（所属的语言）列表与人名对应关系字典	
		* 将人名转化为对应onehot张量表示函数lineToTensor
	
	---

	* 第三步: 构建RNN模型
		* 构建传统的RNN模型的类class RNN.
		* 构建LSTM模型的类class LSTM.
		* 构建GRU模型的类class GRU. 

	---


	* 第四步: 构建训练函数并进行训练
		* 从输出结果中获得指定类别函数categoryFromOutput.
		* 随机生成训练数据函数randomTrainingExample.
		* 构建传统RNN训练函数trainRNN.
		* 构建LSTM训练函数trainLSTM.
		* 构建GRU训练函数trainGRU.
		* 构建时间计算函数timeSince.
		* 构建训练过程的日志打印函数train.得到损失对比曲线和训练耗时对比图.

	---

	* 损失对比曲线分析:
		* 模型训练的损失降低快慢代表模型收敛程度, 由图可知, 传统RNN的模型收敛情况最好, 然后是GRU, 最后是LSTM, 这是因为: 我们当前处理的文本数据是人名, 他们的长度有限, 且长距离字母间基本无特定关联, 因此无法发挥改进模型LSTM和GRU的长距离捕捉语义关联的优势. 所以在以后的模型选用时, 要通过对任务的分析以及实验对比, 选择最适合的模型.

	---

	* 训练耗时对比图分析:
		* 模型训练的耗时长短代表模型的计算复杂度, 由图可知, 也正如我们之前的理论分析, 传统RNN复杂度最低, 耗时几乎只是后两者的一半, 然后是GRU, 最后是复杂度最高的LSTM.

	---
	
	* 结论:
		* 模型选用一般应通过实验对比, 并非越复杂或越先进的模型表现越好, 而是需要结合自己的特定任务, 从对数据的分析和实验结果中获得最佳答案.

	---

	* 第五步: 构建评估函数并进行预测
		* 构建传统RNN评估函数evaluateRNN.
		* 构建LSTM评估函数evaluateLSTM.
		* 构建GRU评估函数evaluateGRU.
		* 构建预测函数predict.
	
---

