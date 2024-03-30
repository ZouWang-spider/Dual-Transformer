import ast

#Read Datasets
def extract_sequences_from_file(data_file):
    sentences = []
    aspect_sequences = []
    opinion_sequences = []
    sentiment = []
    opinion = []

    with open(data_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # 将每一行数据按照####分割为句子和标签部分
                sentence, annotation = line.split('####')
                sentence_tokens = sentence.split()
                # 将字符串转换为列表
                opinion_list = []
                annotations = ast.literal_eval(annotation)
                for annotation in annotations:
                    second_value = annotation[1][0]
                    opinion_list.append(second_value)
                opinion.append(opinion_list)

                # 初始化句子标签列表
                aspect_labels = [0] * len(sentence_tokens)
                opinion_labels = [0] * len(sentence_tokens)

                # 遍历列表中的每个元组
                sentiment_list = []
                for aspect_seq, opinion_seq, label in annotations:
                    for seq1 in aspect_seq:
                        aspect_labels[seq1] = 1
                    for seq2 in opinion_seq:
                        opinion_labels[seq2] = 1
                    #sentiment label
                    if label == "POS":
                        sentiment_list.append(1)
                    elif label == "NEU":
                        sentiment_list.append(0)
                    else:
                        sentiment_list.append(-1)
                # 将提取出的方面词序列和观点词序列添加到列表中
                sentences.append([sentence])
                aspect_sequences.append(aspect_labels)
                opinion_sequences.append(opinion_labels)
                sentiment.append(sentiment_list)

        return sentences, aspect_sequences, opinion_sequences, sentiment, opinion

#
# # 数据集文件路径
# data_file = "E:/PythonProject2/DualTransformer/Dataset/16res/train_triplets.txt"
#
# # 提取方面词序列和观点词序列
# sentences, aspect_sequences, opinion_sequences, sentiment, opinion = extract_sequences_from_file(data_file)
# print(sentences)
# print(aspect_sequences)
# print(opinion_sequences)
# print(sentiment)
# print(opinion)
