import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 全连接层1
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # 全连接层2
        self.fc3 = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        # 输入 x 经过全连接层1和激活函数
        x = torch.relu(self.fc1(x))
        # 经过全连接层2和激活函数
        x = torch.relu(self.fc2(x))
        # 经过输出层
        x = self.fc3(x)
        return x

# # 假设输入张量的维度为 (17, 1536)
# input_tensor = torch.randn(17, 1536)
# # 初始化全连接层模型
# fc_model = FC(input_size=1536, hidden_size=512,  output_size=3)
#
# # 将输入张量输入全连接层模型
# output_tensor = fc_model(input_tensor)
# print(output_tensor.shape)
#
# # 对输出进行 Softmax 处理得到概率分布
# probabilities = F.softmax(output_tensor, dim=1)
# print(probabilities)








