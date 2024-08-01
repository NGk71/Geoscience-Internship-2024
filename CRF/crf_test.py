import json
import jieba
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import classification_report
from time import time
import sys
import time
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

# 定义测试文本和实体
test_text = """狮41H3井2018年4月9日至4月18日压裂，该井水平段长842m，段数11段，段间距76.55m；泵入总液量20271m3，排量5.5-10.3m3/min，砂量936.9m3，砂比10.74%。平均排量8.3m3/min、单段液量1842.82m3、单段砂量85.17m3、铺砂浓度1.11m3/m，最高施工压力94.10MPa,破裂压力80-94.10MPa。措施改造体积7555.8万方，单段改造体积686.89万方。施工狮41H3第8段时，邻井狮41H2出现压力升高，吐砂堵塞求产流程现象；第5、7、10段泵送桥塞时桥塞无法丢手，影响施工进度。"""

true_entities = [
                ("狮41H3井", "WellName"), ("2018年4月9日至4月18日", "ConstructionTime"), 
                ("842m", "HorizontalSectionLength"), ("11段", "StageCount"), 
                ("76.55m", "StageSpacing"), ("20271m3", "TotalFluidVolume"), 
                ("5.5-10.3m3/min", "FlowRate"), ("936.9m3", "SandVolume"), 
                ("10.74%", "SandRatio"), ("8.3m3/min", "AverageFlowRate"), 
                ("1842.82m3", "SingleStageFluidVolume"), ("85.17m3", "SingleStageSandVolume"), 
                ("1.11m3/m", "ProppantConcentration"), ("94.10MPa", "MaxPressure"), 
                ("80-94.10MPa", "FracturePressure"), ("7555.8万方", "ModificationVolume"), 
                ("686.89万方", "SingleStageModificationVolume")
]

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

# 生成测试数据
test_data = tokenize_and_label(test_text, true_entities)

# 打印测试数据
#for sample in test_data:
    #print(sample)

# 保存测试数据到文件
with open('crf_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("测试数据已保存到test_data.json文件中")

# 从JSON文件读取训练数据
with open('crf_train_data.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

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

iterations = [20]

for max_iterations in iterations:
    sys.stdout = Logger("crf_test.log")  # Redirect stdout to log file
    try:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=max_iterations,
            all_possible_transitions=False,
            verbose=False  # 显示训练过程中的中间结果
        )

        start_time = time.time()
        crf.fit(X_train, y_train)
        training_time = time.time() - start_time
        print("---" * 20)
        print(f'Max iterations: {max_iterations}')
        print(f'Training time: {training_time:.2f} seconds')
        print(f'Training time per iteration: {(training_time / max_iterations):.2f} seconds')
        time_start = time.time()
        # 准备测试数据
        X_test = [sent2features(test_data)]
        y_test = [sent2labels(test_data)]

        # 预测
        y_pred = crf.predict(X_test)

        # 评估
        labels = list(crf.classes_)
        labels.remove('O')
        f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels, zero_division=1)
        precision = metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=labels, zero_division=1)
        recall = metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=labels, zero_division=1)
        print(time.time() - time_start)
        print(f'F1 score: {f1_score:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print("---" * 20)
        #for token, true_label, pred_label in zip(sent2tokens(test_data), sent2labels(test_data), y_pred[0]):
            #print(f"Token: {token}, Expected: {true_label}, Predicted: {pred_label}")
    finally:
        sys.stdout.reset()  # Reset stdout to original

