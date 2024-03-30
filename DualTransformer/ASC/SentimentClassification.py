import csv
import dgl
import torch
from supar import Parser
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from torchcrf import CRF
import torch.optim as optim
from sklearn.metrics import log_loss
from stanfordcorenlp import StanfordCoreNLP
from transformers import BertTokenizer, BertModel
from DualTransformer.Model.GCN import GCNModel
from DualTransformer.Model.FCLayer import FC
from DualTransformer.DataProcess.Process import extract_sequences_from_file
from DualTransformer.Model.TransformerDecoder import DualTransformer
from DualTransformer.Model.TransformerEncoder import TransformerEncoderLayer, TransformerEncoder


# 数据集文件路径
data_file = "E:/PythonProject2/DualTransformer/Dataset/15res/train_triplets.txt"
# 提取方面词序列和观点词序列
sentences, aspect_sequences, opinion_sequences, sentiment, opinion = extract_sequences_from_file(data_file)

#Using Stanford Parsing to get Word and Part-of-speech
nlp = StanfordCoreNLP(r'D:/StanfordCoreNLP/stanford-corenlp-4.5.4', lang='en')

# 初始化两个空列表，用于存储输入和输出数据
Word_feature = []
POS_feature = []

# 循环遍历每个评论语句
for sentence in sentences:
    sentence = ''.join(sentence)
    # use CoreNLP to part-of-speech
    ann = nlp.pos_tag(sentence)
    # exaction words and pos
    words = [pair[0] for pair in ann]
    pos_tags = [pair[1] for pair in ann]

    Word_feature.append(words)
    POS_feature.append(pos_tags)

# print(Word_feature)
# print(POS_feature)

#Base Model
# 初始化BERT标记器和模型
# 加载BERT模型和分词器F:\bert-base-cased
model_name = 'E:/bert-base-cased'  # 您可以选择其他预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

#GCN model parameter
input_size2 = 768
hidden_size2 = 768
num_layers2 = 2

#GCN initial
gcn = GCNModel(input_size2, hidden_size2, num_layers2)

# 实例化 Transformer 编码器
encoder_layer = TransformerEncoderLayer(d_model=768, nhead=4)
transformer_encoder = TransformerEncoder(encoder_layer, num_layers=6)

# 实例化 DualTransformer 模块
dual_transformer = DualTransformer(768)

# 实例化 SentimentClassifier 模型
fclayer = FC(input_size=1536, hidden_size=512, output_size=3)

# # 定义并实例化 CRF 模型
# crf_model = CRF(num_tags=2)


#parameter
num_epochs = 50
learning_rate = 0.001
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器和损失函数
all_parameters = list(model.parameters()) + list(gcn.parameters()) + list(encoder_layer.parameters()) +\
                 list(transformer_encoder.parameters())+ list(dual_transformer.parameters()) + list(fclayer.parameters())
optimizer = optim.Adam(all_parameters, lr=learning_rate)
number = round(len(aspect_sequences) / num_epochs)


total_loss = 0.0  # 用于累积损失值
total_accuracy = 0.0  # 用于累积正确预测的样本数量
for epoch in range(num_epochs):
    start_idx = number * epoch
    end_idx = number * (epoch + 1)

    # Word_embedding_feature =  []
    for i, (text, opinion_label, sentiment_label, pos) in enumerate(zip(Word_feature[start_idx:end_idx], opinion[start_idx:end_idx], sentiment[start_idx:end_idx], POS_feature[start_idx:end_idx])):


        # 使用BiAffine对句子进行处理得到arcs、rels、probs
        parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
        dataset = parser.predict([text], prob=True, verbose=True)

        #获取单词节点特征
        marked_text1 = ["[CLS]"] + text + ["[SEP]"]
        # 将分词转化为词向量
        # tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = torch.tensor(tokenizer.encode(marked_text1, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)
        # 获取词向量
        word_embeddings = outputs.last_hidden_state
        # 提取单词对应的词向量（去掉特殊标记的部分）
        word_embeddings = word_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        word_list = [item for item in marked_text1 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        word_embedding_feature = word_embeddings[0][1:-1, :]  # 节点特征


        # 标记化句子
        marked_text2 = ["[CLS]"] + pos + ["[SEP]"]
        # 将分词转化为词向量
        # tokenized_text = tokenizer.tokenize(marked_text)
        input_ids = torch.tensor(tokenizer.encode(marked_text2, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)
        # 获取词向量
        POS_embeddings = outputs.last_hidden_state
        # 提取单词对应的词向量（去掉特殊标记的部分）
        POS_embeddings = POS_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        pos_list = [item for item in marked_text2 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        pos_embedding_feature = POS_embeddings[0][1:-1, :]  # 节点特征


        # 获取依存关系特征
        rels = dataset.rels[0]
        # 获取依存特征
        marked_text3 = ["[CLS]"] + rels + ["[SEP]"]
        # 将分词转化为词向量
        input_ids = torch.tensor(tokenizer.encode(marked_text3, add_special_tokens=True)).unsqueeze(0)  # 添加批次维度
        outputs = model(input_ids)
        # 获取词向量
        dep_embeddings = outputs.last_hidden_state
        # 提取单词对应的词向量（去掉特殊标记的部分）
        dep_embeddings = dep_embeddings[:, 1:-1, :]  # 去掉[CLS]和[SEP]标记
        # 提取单词（去掉特殊标记的部分）
        dep_list = [item for item in marked_text3 if item not in ['[CLS]', '[SEP]']]
        # 使用切片操作去除第一个和最后一个元素
        dep_embedding_feature = dep_embeddings[0][1:-1, :]  # 节点特征


        # 使用 torch.cat 进行水平拼接  (n, 768*13)   output1, output2, output3
        syn_feature = torch.cat((word_embedding_feature, pos_embedding_feature, dep_embedding_feature), dim=1)
        max_pooling = nn.MaxPool1d(kernel_size=3)  # 使用3作为池化窗口大小,最大池化
        # 将syn_feature的维度从(n, 768*3)转换为(n, 768)
        syn_feature_pooled = max_pooling(syn_feature)


        # 构建句子的图 g
        arcs = dataset.arcs[0]  # 边的信息
        edges = [i + 1 for i in range(len(arcs))]
        for i in range(len(arcs)):
            if arcs[i] == 0:
                arcs[i] = edges[i]

        # 将节点的序号减一，以便适应DGL graph从0序号开始
        arcs = [arc - 1 for arc in arcs]
        edges = [edge - 1 for edge in edges]
        graph = (arcs, edges)
        syn_graph = torch.tensor(graph)
        # Create a DGL graph
        g = dgl.graph(graph)  # 句子的图结构

        # GCN模型 torch.Size([n, 5])
        gcnoutput = gcn(syn_feature_pooled, syn_graph)

        # 将输入数据传入 Transformer 编码器
        gcn_tensor = gcnoutput.unsqueeze(1)
        aspect_tensor = transformer_encoder(gcn_tensor)
        opinion_tensor = transformer_encoder(gcn_tensor)
        # 使用 squeeze 函数压缩维度为 1 的维度
        aspect_tensor = torch.squeeze(aspect_tensor, dim=1)
        opinion_tensor = torch.squeeze(opinion_tensor, dim=1)

        # Dual Transformer Interaction learning
        left_hidden, right_hidden = dual_transformer(aspect_tensor, opinion_tensor, gcnoutput)

        concatenated_tensor = torch.cat((left_hidden, right_hidden), dim=1)

        # 前向传播
        output = fclayer(concatenated_tensor)

        # 使用索引操作选取 output 中对应位置的数据
        selected_values = output[opinion_label]
        # print(selected_values)


        #将标签转化为one-hot编码
        one_hot_labels = torch.zeros(len(sentiment_label), 3)
        for i, label in enumerate(sentiment_label):
            # 将标签映射到 one-hot 编码形式
            if label == -1:
                one_hot_labels[i, 0] = 1
            elif label == 0:
                one_hot_labels[i, 1] = 1
            elif label == 1:
                one_hot_labels[i, 2] = 1

        # 标签列表
        labels = [-1, 0, 1]
        # 找到每个张量中最大值及其对应的索引
        max_values, max_indices = torch.max(selected_values, dim=1)
        # 根据索引获取对应的情感标签
        pred_sentiment = [labels[idx.item()] for idx in max_indices]
        pred_tensor_sentiment = torch.tensor(pred_sentiment)

        # 计算损失值
        loss = criterion(selected_values, one_hot_labels)
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()


        # 情感标签转化为张量
        sentiment_tensor_label = torch.tensor(sentiment_label)  # 标签张量
        # 计算准确率
        correct = (pred_tensor_sentiment == sentiment_tensor_label).sum().item()
        accuracy = correct / len(one_hot_labels)
        total_accuracy += accuracy  # 累计准确率
        total_loss += loss.item()  # 累积损失值
        # 打印每个 epoch 的损失值
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy}")

    # 计算并打印每个 epoch 的平均损失值
    average_loss = total_loss / number
    average_accuracy = total_accuracy / number
    with open('E:\PythonProject2\DualTransformerFigure\ASC\ASCTask_15res.txt', 'a') as file:
        output_string = f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}\n"
        file.write(output_string)
    total_loss = 0.0
    total_accuracy = 0.0
