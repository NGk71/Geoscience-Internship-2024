import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Bidirectional, TimeDistributed, Dense, Embedding, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import os
import tensorflow as tf

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

# 分离验证集数据
validation_split = 0.2
num_validation_samples = int(validation_split * len(train_sentences_padded))

val_sentences_padded = train_sentences_padded[:num_validation_samples]
val_labels_categorical = train_labels_categorical[:num_validation_samples]

train_sentences_padded = train_sentences_padded[num_validation_samples:]
train_labels_categorical = train_labels_categorical[num_validation_samples:]

# 初始化粒子群算法参数
def initialize_pso(num_particles, num_features):
    particles = np.random.rand(num_particles, num_features)
    velocities = np.zeros((num_particles, num_features))
    pbest = particles.copy()
    pbest_f1 = np.zeros(num_particles)
    gbest = particles[np.random.choice(range(num_particles))]
    gbest_f1 = 0
    return particles, velocities, pbest, pbest_f1, gbest, gbest_f1

def update_pso(particles, velocities, pbest, gbest, w, c1, c2):
    r1, r2 = np.random.rand(), np.random.rand()
    velocities = w * velocities + c1 * r1 * (pbest - particles) + c2 * r2 * (gbest - particles)
    particles += velocities
    return particles, velocities

# PSO参数设置
num_particles = 30
num_iterations = 50
w = 0.5
c1 = 1.5
c2 = 1.5

# 初始化粒子群
particles, velocities, pbest, pbest_f1, gbest, gbest_f1 = initialize_pso(num_particles, 2)

# 训练和早停机制
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('bilstm_best_model.keras', save_best_only=True, monitor='val_loss', mode='min')

# PSO优化过程
for iteration in range(num_iterations):
    for i in range(num_particles):
        # 更新模型参数
        hidden_units_1, hidden_units_2 = int(particles[i, 0] * 100), int(particles[i, 1] * 100)
        
        input = Input(shape=(max_len,))
        model = Embedding(input_dim=len(word_encoder.classes_), output_dim=50, input_length=max_len)(input)
        model = Bidirectional(LSTM(units=hidden_units_1, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)
        model = Bidirectional(LSTM(units=hidden_units_2, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
        model = BatchNormalization()(model)
        model = Dropout(0.5)(model)
        model = TimeDistributed(Dense(100, activation="relu"))(model)
        output = TimeDistributed(Dense(num_labels+1, activation="softmax"))(model)

        model = Model(input, output)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        history = model.fit(train_sentences_padded, train_labels_categorical, validation_data=(val_sentences_padded, val_labels_categorical), batch_size=64, epochs=5, verbose=1, callbacks=[early_stopping, model_checkpoint])

        # 计算适应度（基于验证集的F1 Score）
        val_predictions = model.predict(val_sentences_padded)
        val_predictions = np.argmax(val_predictions, axis=-1).flatten()
        val_labels = np.argmax(val_labels_categorical, axis=-1).flatten()
        _, _, f1, _ = precision_recall_fscore_support(val_labels, val_predictions, average='weighted', zero_division=1)

        if f1 > pbest_f1[i]:
            pbest[i] = particles[i]
            pbest_f1[i] = f1

        if f1 > gbest_f1:
            gbest = particles[i]
            gbest_f1 = f1

    particles, velocities = update_pso(particles, velocities, pbest, gbest, w, c1, c2)

# 使用最佳参数训练最终模型
hidden_units_1, hidden_units_2 = int(gbest[0] * 100), int(gbest[1] * 100)
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(word_encoder.classes_), output_dim=50, input_length=max_len)(input)
model = Bidirectional(LSTM(units=hidden_units_1, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
model = BatchNormalization()(model)
model = Dropout(0.5)(model)
model = Bidirectional(LSTM(units=hidden_units_2, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))(model)
model = BatchNormalization()(model)
model = Dropout(0.5)(model)
model = TimeDistributed(Dense(100, activation="relu"))(model)
output = TimeDistributed(Dense(num_labels+1, activation="softmax"))(model)

model = Model(input, output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_sentences_padded, train_labels_categorical, validation_data=(val_sentences_padded, val_labels_categorical), batch_size=64, epochs=300, verbose=0, callbacks=[early_stopping, model_checkpoint])

# 使用最佳模型文件进行评估
best_model_path = 'bilstm_best_model.h5'

if os.path.exists(best_model_path): 
    print(f"Evaluating best saved model")
    # 加载最佳模型和编码器
    model = load_model(best_model_path)
    word_encoder = LabelEncoder()
    word_encoder.classes_ = np.load('word_encoder.npy', allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('label_encoder.npy', allow_pickle=True)
    max_len = np.load('max_len.npy')[0]

    # 加载并预处理测试数据
    with open('crf_test_data.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    test_sentences = []
    test_labels = []
    for data in test_data:
        tokens, labels = tokenize_and_label(data)
        test_sentences.append(tokens)
        test_labels.append(labels)

    test_sentences_encoded = [word_encoder.transform(tokens) for tokens in test_sentences]
    test_labels_encoded = [label_encoder.transform(labels) for labels in test_labels]

    test_sentences_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant') for seq in test_sentences_encoded])
    test_labels_padded = np.array([np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=num_labels) for seq in test_labels_encoded])

    # One-hot编码标签
    test_labels_categorical = np.array([to_categorical(seq, num_classes=num_labels+1) for seq in test_labels_padded])

    # 预测
    y_pred = model.predict(test_sentences_padded)
    y_pred = np.argmax(y_pred, axis=-1).flatten()

    # 去除填充部分
    y_pred = [pred[:len(test_sentences[i])] for i, pred in enumerate(y_pred)]
    y_test = [true[:len(test_sentences[i])] for i, true in enumerate(test_labels_padded)]

    # 解码标签
    y_pred_labels = [label_encoder.inverse_transform(pred) for pred in y_pred]
    y_test_labels = [label_encoder.inverse_transform(true) for true in y_test]

    # 评估
    precision, recall, f1, _ = precision_recall_fscore_support(np.concatenate(y_test_labels), np.concatenate(y_pred_labels), average='weighted', zero_division=1)
    print("---" * 20)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(classification_report(np.concatenate(y_test_labels), np.concatenate(y_pred_labels), zero_division=1))
    print("---" * 20)

    # 解码并打印未见过的标签
    unseen_labels = set(np.concatenate(y_pred)) - set(label_encoder.transform(label_encoder.classes_))
    unseen_labels_decoded = [label_encoder.inverse_transform([label])[0] if label in label_encoder.transform(label_encoder.classes_) else 'Unknown' for label in unseen_labels]

    print("Unseen Labels in Predictions (decoded):")
    for label in unseen_labels_decoded:
        print(label)
else:
    print(f"Model file {best_model_path} does not exist.")

print(f"Total epochs run: {len(history.epoch)}")
print(f"Total training time: {np.sum(history.history['val_loss']):.2f} seconds")
