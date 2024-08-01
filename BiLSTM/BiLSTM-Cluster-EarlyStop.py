import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, TimeDistributed, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.cluster import KMeans
import time
import os

# 分词和标注函数
def tokenize_and_label(data):
    tokens = []
    labels = []
    for token, label in data:
        tokens.append(token)
        labels.append(label)
    return tokens, labels

# 加载并预处理训练数据
with open('crf_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

train_sentences = []
train_labels = []
for data in train_data:
    tokens, labels = tokenize_and_label(data)
    train_sentences.append(tokens)
    train_labels.append(labels)

# 合并所有标签以便拟合LabelEncoder
all_labels = [label for labels in train_labels for label in labels]

# 标签编码
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# 转换标签
train_labels_encoded = [label_encoder.transform(labels) for labels in train_labels]

# 标签数量
num_labels = len(label_encoder.classes_)

# 合并所有词语以便拟合WordEncoder
all_tokens = [token for tokens in train_sentences for token in tokens]

# 词语编码
word_encoder = LabelEncoder()
word_encoder.fit(all_tokens)

# 转换词语
train_sentences_encoded = [word_encoder.transform(tokens) for tokens in train_sentences]

# 序列填充
max_len = max([len(seq) for seq in train_sentences_encoded])
train_sentences_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in train_sentences_encoded])
train_labels_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=num_labels) for seq in train_labels_encoded])

# One-hot编码标签
train_labels_categorical = np.array([to_categorical(seq, num_classes=num_labels+1) for seq in train_labels_padded])

# 提前聚类
def cluster_sentences(sentences, n_clusters):
    vectorized_sentences = np.array([np.mean([ord(char) for char in ''.join(sentence)]) for sentence in sentences]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(vectorized_sentences)
    clustered_sentences = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clustered_sentences[label].append(i)
    return clustered_sentences

clustered_indices = cluster_sentences(train_sentences, n_clusters=5)
clustered_train_sentences_padded = np.concatenate([train_sentences_padded[indices] for indices in clustered_indices])
clustered_train_labels_categorical = np.concatenate([train_labels_categorical[indices] for indices in clustered_indices])

# 创建模型
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word_encoder.classes_), output_dim=50, input_length=max_len)(input)
model = Bidirectional(LSTM(units=100, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
model = TimeDistributed(Dense(100, activation="relu"))(model)
output = TimeDistributed(Dense(num_labels+1, activation="softmax"))(model)

model = Model(input, output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 训练和早停机制
max_epochs = 300
early_stopping_threshold = 0.99
best_epoch = 0
total_training_time = 0

for epoch in range(1, max_epochs + 1):
    print(f"Training with epoch {epoch}")
    start_time = time.time()
    history = model.fit(clustered_train_sentences_padded, clustered_train_labels_categorical, batch_size=64, epochs=1, validation_split=0.2, verbose=0)
    training_time = time.time() - start_time
    total_training_time += training_time

    time_per_iter = training_time
    print("---" * 20)
    print(f'Training time for epoch {epoch}: {training_time:.2f} seconds')
    print(f'Time per iteration: {time_per_iter:.2f} seconds')
    print("---" * 20)

    # 评估验证集F1分数
    val_predictions = model.predict(clustered_train_sentences_padded)
    val_predictions = np.argmax(val_predictions, axis=-1).flatten()
    val_labels = np.argmax(clustered_train_labels_categorical, axis=-1).flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='weighted', zero_division=1)
    print(f"Validation F1-score: {f1:.4f}")
    
    if f1 > early_stopping_threshold:
        print(f"Early stopping as validation F1-score reached {f1:.4f}")
        
        # 保存模型和编码器
        model.save(f'bilstm_model_{epoch}_epochs.h5')
        np.save(f'word_encoder_{epoch}_epochs.npy', word_encoder.classes_)
        np.save(f'label_encoder_{epoch}_epochs.npy', label_encoder.classes_)
        np.save(f'max_len_{epoch}_epochs.npy', np.array([max_len]))
        best_epoch = epoch
        break  # 提前跳出循环

# 如果提前停止了，使用保存的模型文件进行评估
if best_epoch > 0:
    model_path = f'bilstm_model_{best_epoch}_epochs.h5'
else:
    model_path = f'bilstm_model_{max_epochs}_epochs.h5'
    model.save(model_path)
    np.save(f'word_encoder_{max_epochs}_epochs.npy', word_encoder.classes_)
    np.save(f'label_encoder_{max_epochs}_epochs.npy', label_encoder.classes_)
    np.save(f'max_len_{max_epochs}_epochs.npy', np.array([max_len]))

# 加载并预处理测试数据
with open('crf_test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

test_tokens, test_labels = tokenize_and_label(test_data)

if os.path.exists(model_path):  # 确保模型文件存在
    print(f"Evaluating model with {best_epoch if best_epoch > 0 else max_epochs} epochs")
    # 加载模型和编码器
    model = load_model(model_path)
    word_encoder = LabelEncoder()
    word_encoder.classes_ = np.load(f'word_encoder_{best_epoch if best_epoch > 0 else max_epochs}_epochs.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(f'label_encoder_{best_epoch if best_epoch > 0 else max_epochs}_epochs.npy', allow_pickle=True)
    max_len = np.load(f'max_len_{best_epoch if best_epoch > 0 else max_epochs}_epochs.npy')[0]

    # 确保所有未见过的文字和字符都替换为 "O"
    known_tokens = set(word_encoder.classes_)
    test_tokens = [token if token in known_tokens else ',' for token in test_tokens]

    # 转换词语
    test_sentences_encoded = word_encoder.transform(test_tokens)

    # 准备测试数据
    if len(test_sentences_encoded) < max_len:
        test_sentences_padded = np.pad(test_sentences_encoded, (0, max_len - len(test_sentences_encoded)), 'constant').reshape(1, -1)
    else:
        test_sentences_padded = np.array(test_sentences_encoded[:max_len]).reshape(1, -1)

    # 预测
    y_pred = model.predict(test_sentences_padded)
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    # 去除填充部分
    y_pred = y_pred[:len(test_tokens)]
    y_test = label_encoder.transform(test_labels)

    # 检查预测标签是否有效
    unseen_labels = set(y_pred) - set(label_encoder.transform(label_encoder.classes_))
    if unseen_labels:
        print(f"Unseen labels in predictions: {unseen_labels}")
        y_pred = np.array([label if label in label_encoder.transform(label_encoder.classes_) else label_encoder.transform(['O'])[0] for label in y_pred])

    # 解码标签
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # 评估
    precision, recall, f1, _ = precision_recall_fscore_support(y_test_labels, y_pred_labels, average='weighted', zero_division=1)
    print("---" * 20)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(y_test_labels, y_pred_labels, zero_division=1))
    print("---" * 20)

    # 解码并打印未见过的标签
    unseen_labels_decoded = [label_encoder.inverse_transform([label])[0] if label in label_encoder.transform(label_encoder.classes_) else 'Unknown' for label in unseen_labels]

    print("Unseen Labels in Predictions (decoded):")
    for label in unseen_labels_decoded:
        print(label)
else:
    print(f"Model file {model_path} does not exist.")

print(f"Total training time: {total_training_time:.2f} seconds")
