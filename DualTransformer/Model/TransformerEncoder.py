import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


# # 准备输入数据（假设输入向量维度为 512）
# input_tensor = torch.randn(10, 768)  # 假设有 10 个样本，每个样本有 5 个时间步，输入维度为 512
# # 将输入张量维度调整为 [len, batch_size, d_model]
# input_tensor = input_tensor.unsqueeze(1)
# print(input_tensor.shape)
#
#
# # 实例化 Transformer 编码器
# encoder_layer = TransformerEncoderLayer(d_model=768, nhead=4)
# transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)
#
# # 将输入数据传入 Transformer 编码器
# output_tensor = transformer_encoder(input_tensor)
#
# # 输出将是经过 Transformer 编码器处理后的张量，其形状为 (10, 5, 512)
# print(output_tensor.shape)

