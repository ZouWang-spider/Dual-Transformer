import torch
import torch.nn as nn
from DualTransformer.Model.TransformerCross import Transformer_Cross


class DualTransformer(nn.Module):
    def __init__(self, hidden_dim):
        super(DualTransformer, self).__init__()
        self.left_transformer = Transformer_Cross(hidden_dim)
        self.right_transformer = Transformer_Cross(hidden_dim)
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.W3 = nn.Linear(hidden_dim, hidden_dim)
        self.W4 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, aspect_tensor, opinion_tensor, gcn_tensor):
        # 左边的 Transformer 输入 Q 为 aspect_tensor，K 和 V 为 gcn_tensor
        left_output = self.left_transformer(aspect_tensor, gcn_tensor, gcn_tensor)

        # 右边的 Transformer 输入 Q 为 opinion_tensor，K 和 V 也是 gcn_tensor
        right_output = self.right_transformer(opinion_tensor, gcn_tensor, gcn_tensor)

        # Cross Gating
        left_hidden = left_output + torch.sigmoid(self.W1(left_output) + self.W2(right_output)) * right_output
        right_hidden = right_output + torch.sigmoid(self.W3(right_output) + self.W4(left_output)) * left_output


        return left_hidden, right_hidden



# # 生成随机的输入数据
# aspect_tensor = torch.randn(10, 768)  # 假设 aspect_tensor 维度为 (10, 768)
# opinion_tensor = torch.randn(10, 768)  # 假设 opinion_tensor 维度为 (10, 768)
# gcn_tensor = torch.randn(10, 768)  # 假设 gcn_tensor 维度为 (10, 768)
#
# # 实例化 DualTransformer 模块
# hidden_dim = 768
# dual_transformer = DualTransformer(hidden_dim)
#
# # 将输入数据传递给模型进行前向传播
# left_hidden, right_hidden = dual_transformer(aspect_tensor, opinion_tensor, gcn_tensor)
#
# # 打印输出结果
# print("Left output shape:", left_hidden.shape)
# print("Right output shape:", right_hidden.shape)