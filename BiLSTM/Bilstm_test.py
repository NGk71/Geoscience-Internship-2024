import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tensorflow.keras.models import load_model
import time

# 读取测试数据
with open('crf_test_data.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
star=time.time()
# 分词和标注函数
def tokenize_and_label(data):
    tokens = []
    labels = []
    for token, label in data:
        tokens.append(token)
        labels.append(label)
    return tokens, labels

# 生成测试数据
test_tokens, test_labels = tokenize_and_label(test_data)

# 加载模型和编码器
model = load_model('bilstm_model_63_epochs.h5')
word_encoder = LabelEncoder()
word_encoder.classes_ = np.load('word_encoder_63_epochs.npy', allow_pickle=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_63_epochs.npy', allow_pickle=True)
max_len = np.load('max_len_63_epochs.npy')[0]

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
y_test = label_encoder.transform(test_labels[:len(test_tokens)])  # Ensure the lengths match

# 检查预测标签是否有效
unseen_labels = set(y_pred) - set(label_encoder.transform(label_encoder.classes_))
if unseen_labels:
    print(f"Unseen labels in predictions: {unseen_labels}")
    y_pred = np.array([label if label in label_encoder.transform(label_encoder.classes_) else label_encoder.transform(['O'])[0] for label in y_pred])

# 确保y_test和y_pred长度一致
min_len = min(len(y_test), len(y_pred))
y_test = y_test[:min_len]
y_pred = y_pred[:min_len]

# 解码标签
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)
print(f'---{time.time()-star}---')
# 评估
precision, recall, f1, _ = precision_recall_fscore_support(y_test_labels, y_pred_labels, average='weighted', zero_division=1)
print("---" * 20)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
print("---" * 20)

# Decode unseen labels if they exist in the label_encoder
unseen_labels_decoded = [label_encoder.inverse_transform([label])[0] if label in label_encoder.transform(label_encoder.classes_) else 'Unknown' for label in unseen_labels]

# Print the decoded unseen labels
print("Unseen Labels in Predictions (decoded):")
for label in unseen_labels_decoded:
    print(label)
