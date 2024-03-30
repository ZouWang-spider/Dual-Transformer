

#Read Datasets
def extract_sequences_from_file(data_file):
    sentences = []
    aspect_sequences = []
    opinion_sequences = []
    sentiment = []

    with open(data_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                # 将每一行数据按照####分割为句子和标签部分
                text, annotation1, annotation2 = line.split('####')

                # 提取方面词序列和观点词序列
                aspect_sequence = [word.split('=')[1] for word in annotation1.split() if '=' in word]
                opinion_sequence = [word.split('=')[1] for word in annotation2.split() if '=' in word]

                # 将提取出的方面词序列和观点词序列添加到列表中
                sentences.append([text])
                aspect_sequences.append(aspect_sequence)
                opinion_sequences.append(opinion_sequence)

        return sentences, aspect_sequences, opinion_sequences, sentiment


# 数据集文件路径
data_file = "E:/PythonProject2/DualTransformer/Dataset/16res/train_triplets.txt"

# 提取方面词序列和观点词序列
sentences, aspect_sequences, opinion_sequences, sentiment = extract_sequences_from_file(data_file)
print(sentences)
print(aspect_sequences)
print(opinion_sequences)
print(sentiment)
