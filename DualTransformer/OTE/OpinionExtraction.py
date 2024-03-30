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
from DualTransformer.DataProcess.Process import extract_sequences_from_file
from DualTransformer.Model.TransformerEncoder import TransformerEncoderLayer, TransformerEncoder


# 数据集文件路径
data_file = "E:/PythonProject2/DualTransformer/Dataset/14lap/train_triplets.txt"
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
    # print(nlp.pos_tag(sentence))
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

# 定义并实例化 CRF 模型
crf_model = CRF(num_tags=2)


#parameter
num_epochs = 50
learning_rate = 0.001

# 定义优化器和损失函数
all_parameters = list(model.parameters()) + list(gcn.parameters()) + list(encoder_layer.parameters()) + list(transformer_encoder.parameters())+ list(crf_model.parameters())
optimizer = optim.Adam(all_parameters, lr=learning_rate)
number = round(len(opinion_sequences) / num_epochs)


total_loss = 0.0  # 用于累积损失值
total_accuracy = 0.0  # 用于累积正确预测的样本数量
for epoch in range(num_epochs):
    start_idx = number * epoch
    end_idx = number * (epoch + 1)

    # Word_embedding_feature =  []
    for i, (text, opinion_label, pos) in enumerate(zip(Word_feature[start_idx:end_idx], opinion_sequences[start_idx:end_idx],POS_feature[start_idx:end_idx])):

        # 将标签转化为张量,将标签转化为，0,1处理
        # label_ids = [0 if label == 'O' else 1 for label in label]
        # label_tensor = torch.tensor([label_ids])
        # print(label_ids)
        # print(label_tensor)  #tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])

        # 使用BiAffine对句子进行处理得到arcs、rels、probs
        parser = Parser.load('biaffine-dep-en')  # 'biaffine-dep-roberta-en'解析结果更准确
        dataset = parser.predict([text], prob=True, verbose=True)

        #获取单词节点特征
        # 标记化句子
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
        # print(output.shape)   torch.Size([17, 768])

        gcn_tensor = gcnoutput.unsqueeze(1)
        # 将输入数据传入 Transformer 编码器
        output_tensor = transformer_encoder(gcn_tensor)
        # 挤压维度 1，将三维张量转换为二维张量
        output = output_tensor.squeeze(1)
        # print(output.shape)    torch.Size([17, 768])

        NNmodel = nn.Sequential(
            nn.Linear(768, 2),
            nn.Softmax(dim=1)
        )

        output_tensor = NNmodel(output)  # (10,2)
        # 使用 unsqueeze 在维度 0 处添加一个维度
        output = output_tensor.unsqueeze(0)

        # 输入输出张量到 CRF 模型中，得到预测的序列标签
        pred_labels = crf_model.decode(output)
        pred_label = [item for sublist in pred_labels for item in sublist]

        # 检查长度是否相等
        if len(pred_label) == len(opinion_label):
            pass
        elif len(pred_label) < len(opinion_label):
            opinion_label = opinion_label[-len(pred_label):]
        else:  # len(predict_label) > len(aspect_term_seq)
            opinion_label = [0] * (len(pred_label) - len(opinion_label)) + opinion_label
        # print(label_ids)
        # print(pred_label)

        # 使用负对数似然作为损失,对数似然损失（Log-Likelihood Loss）
        loss = log_loss(opinion_label, pred_label)
        loss_tensor = torch.tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()

        # 计算准确率
        pred_label_tensor = torch.tensor(pred_label)
        label_tensor = torch.tensor(opinion_label)

        correct = (pred_label_tensor == label_tensor).sum().item()
        accuracy = correct / len(opinion_label)
        total_accuracy += accuracy  # 累计准确率
        total_loss += loss.item()  # 累积损失值
        # 打印每个 epoch 的损失值
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Accuracy: {accuracy}")

    # 计算并打印每个 epoch 的平均损失值
    average_loss = total_loss / number
    average_accuracy = total_accuracy / number
    with open('OTETask_14lap.txt', 'a') as file:
        output_string = f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}, Average Accuracy: {average_accuracy}\n"
        file.write(output_string)
    total_loss = 0.0
    total_accuracy = 0.0
