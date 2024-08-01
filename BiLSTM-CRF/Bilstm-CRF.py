

# Define test text and entities
test_text ="2019年4月20日，施工人员到达狮平1井，并在0:00进行了车辆检查，随后在0:25召开了安全技术交底会，0:40正式开始施工。这次施工时间从2017年6月29日持续到7月6日，水平段长达597m，共有8段，每段间距为74.6m。第三十五段施工开始时，先排空了150.34m³的液体，打备压达到97.78MPa，随后洗井用了278.45m³的液体，同样打备压97.78MPa。泵送桥塞用液量为80.45m³，压裂压力为97.78MPa，而送球用液量为70.12m³。整个施工过程中，泵入了919.45m³的前置液，段塞加砂量为80.45m³，携砂液量为523.34m³，总砂量达到162.78m³，最高砂比为54.34%，平均砂比则为35.56%。在施工过程中，顶替液的总量为210.45m³，纤维总共使用了180Kg，净液量为9742.20m³，总酸量为250.00m³。施工最高压力记录为117.34MPa，破裂压力为117.34MPa，停泵油压为90.12MPa，最大排量达到18.78m³/min，平均排量为7.8m³/min，单段液量为1295m³，单段砂量为77.65m³，铺砂浓度为1.04m³/m。整个施工过程中共用时270分钟，在5:30结束施工后，压裂车组在5:45进行了设备整修，并于7:15离开井场。这次措施改造的体积达到767.8万方，其中单段改造体积为96万方。"

true_entities = [
    ("0:00", "ArrivalTime"), ("0:25", "VehicleCheckTime"), ("0:40", "StartTime"),
    ("2017年6月29日", "ConstructionTime"), ("597m", "HorizontalSectionLength"), ("8段", "StageCount"), ("74.6m", "StageSpacing"), 
    ("150.34m³", "FluidForEmptying"), ("97.78MPa", "Pressure"), ("278.45m³", "FluidForWellFlushing"), 
    ("80.45m³", "FluidForBridgePlug"), ("70.12m³", "FluidForBall"), ("919.45m³", "FluidForPrepad"), 
    ("80.45m³", "SandForPlug"), ("523.34m³", "FluidForProppant"), ("162.78m³", "Sand"), 
    ("54.34%", "MaxSandRatio"), ("35.56%", "AverageSandRatio"), ("210.45m³", "DisplacementFluid"), 
    ("180Kg", "Fiber"), ("117.34MPa", "MaxPressure"), ("117.34MPa", "FracturePressure"), 
    ("90.12MPa", "PumpStopPressure"), ("18.78m³/min", "MaxFlowRate"), ("270分钟", "JobTime"), 
    ("9742.20m³", "NetFluidVolume"), ("250.00m³", "TotalAcidVolume"), ("7.8m³/min", "AverageFlowRate"), 
    ("1295m³", "SingleStageFluidVolume"), ("77.65m³", "SingleStageSandVolume"), 
    ("1.04m³/m", "ProppantConcentration"), ("767.8万方", "ModificationVolume"), ("96万方", "SingleStageModificationVolume")
]
import json
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout
import time
import sys

# Logger class to redirect stdout to a log file
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        pass

    def reset(self):
        self.log.close()
        sys.stdout = self.terminal

# Function to tokenize and label the text
def tokenize_and_label(text, entities):
    tokens = list(text)
    labels = ['O'] * len(tokens)
    
    for entity, label in entities:
        start_idx = text.find(entity)
        if start_idx != -1:
            end_idx = start_idx + len(entity)
            labels[start_idx] = 'B-' + label
            for i in range(start_idx + 1, end_idx):
                labels[i] = 'I-' + label
    
    return list(zip(tokens, labels))

# Generate test data
test_data = tokenize_and_label(test_text, true_entities)

# Save test data to file
with open('crf_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("测试数据已保存到test_data.json文件中")

# Load training data
with open('crf_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Function to extract features from the data
def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

# Preparing data for the BiLSTM-CRF model
def preprocess_data(data):
    tokens = [token for token, label in data]
    labels = [label for token, label in data]
    return tokens, labels

train_tokens, train_labels = zip(*[preprocess_data(s) for s in train_data])
test_tokens, test_labels = preprocess_data(test_data)

# Vocabulary and labels processing
all_tokens = list(set([token for tokens in train_tokens for token in tokens] + [token for token in test_tokens]))
all_labels = list(set([label for labels in train_labels for label in labels] + [label for label in test_labels]))

token2idx = {token: idx + 2 for idx, token in enumerate(all_tokens)}
token2idx["<PAD>"] = 0
token2idx["<UNK>"] = 1
idx2token = {idx: token for token, idx in token2idx.items()}

label2idx = {label: idx + 1 for idx, label in enumerate(all_labels)}
label2idx["O"] = 0
idx2label = {idx: label for label, idx in label2idx.items()}

print("Token2Idx:", token2idx)
print("Label2Idx:", label2idx)

max_len = 100

X_train = pad_sequences([[token2idx.get(token, token2idx["<UNK>"]) for token in tokens] for tokens in train_tokens], maxlen=max_len, padding='post', value=token2idx["<PAD>"])
X_test = pad_sequences([[token2idx.get(token, token2idx["<UNK>"]) for token in test_tokens]], maxlen=max_len, padding='post', value=token2idx["<PAD>"])

y_train = pad_sequences([[label2idx[label] for label in labels] for labels in train_labels], maxlen=max_len, padding='post', value=label2idx["O"])
y_test = pad_sequences([[label2idx[label] for label in test_labels]], maxlen=max_len, padding='post', value=label2idx["O"])

# Ensure no label index exceeds the length of label2idx
def safe_to_categorical(y, num_classes):
    y = np.where(np.array(y) >= num_classes, label2idx["O"], y)
    return to_categorical(y, num_classes=num_classes)

y_train = [safe_to_categorical(i, num_classes=len(label2idx)) for i in y_train]
y_test = [safe_to_categorical(i, num_classes=len(label2idx)) for i in y_test]

print("Shape of y_train:", np.array(y_train).shape)
print("Shape of y_test:", np.array(y_test).shape)

# Build BiLSTM-CRF model
input = Input(shape=(max_len,))
model = Embedding(input_dim=len(token2idx), output_dim=50, input_length=max_len)(input)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, dropout=0.2))(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1, dropout=0.2))(model)
out = TimeDistributed(Dense(len(label2idx), activation="softmax"))(model)

model = Model(input, out)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train the model with increasing epochs
epochs = [175]
for epoch in epochs:
    sys.stdout = Logger(f"bilstm_{epoch}_epochs.log")  # Redirect stdout to log file
    try:
        time_start = time.time()
        history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=epoch, validation_split=0.1, verbose=0)
        training_time = time.time() - time_start
        print(f'Training time: {training_time:.2f} seconds')
        print(f'Training time per iteration: {(training_time / epoch):.2f} seconds')
        # Evaluate the model
        test_pred = model.predict(X_test, verbose=0)
        test_pred = np.argmax(test_pred, axis=-1)
        y_test_true = np.argmax(y_test[0], axis=-1)

        # Map labels back to tokens
        y_pred_labels = [[idx2label[idx] for idx in pred] for pred in test_pred]
        y_true_labels = [idx2label[idx] for idx in y_test_true]

        # Print predicted and true labels for each token
        print("Predicted Label -> True Label")
        for token, pred, true in zip(test_tokens, y_pred_labels[0], y_true_labels):
            print(f"{token}: {pred} -> {true}")

        print(f"Results for {epoch} epochs:")
        print(classification_report(y_true_labels, y_pred_labels[0]))
    finally:
        sys.stdout.reset()  # Reset stdout to original

