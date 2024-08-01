import spacy
from spacy.training.example import Example
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import sys
import json
from spacy.util import minibatch, compounding

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

# 读取JSON文件中的训练数据
def load_training_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

TRAIN_DATA = load_training_data('train_data_spacy_format.json')

# 加载 spaCy 的中文模型
nlp = spacy.blank("zh")

# 创建一个新的NER组件并添加到管道中
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner", last=True)
else:
    ner = nlp.get_pipe("ner")

# 添加自定义实体标签
labels = ["WellName", "ConstructionTime", "HorizontalSectionLength", "DesignStageCount", 
          "ActualStageCount", "StageSpacing", "TotalFluidVolume", "NetFluidVolume", 
          "AcidVolume", "FlowRate", "SandVolume", "SandRatio", "AverageFlowRate", 
          "SingleStageFluidVolume", "SingleStageSandVolume", "ProppantConcentration", 
          "MaxPressure", "FracturePressure", "ModificationVolume", "SingleStageModificationVolume"]

for label in labels:
    ner.add_label(label)

# 定义一个获取示例的函数，以便初始化训练
def get_examples():
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        yield example

# 开始训练NER模型
def train_and_evaluate(n_iter, batch_size=32):
    start = time.time()
    optimizer = nlp.initialize(get_examples=get_examples)

    for i in range(n_iter):
        losses = {}
        # 创建小批量
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, batch_size, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in zip(texts, annotations)]
            nlp.update(examples, drop=0.2, losses=losses)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Losses: {losses}")

    total_time = time.time() - start
    time_per_iteration = total_time / n_iter

    # 评估代码
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

    doc = nlp(test_text)
    predicted_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # 提取真实和预测的标签
    true_labels = [label for _, label in true_entities]
    predicted_labels = [label for _, label in predicted_entities]

    # 确保长度匹配
    min_length = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_length]
    predicted_labels = predicted_labels[:min_length]

    # 计算指标
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)

    print("---"*20)
    print(f"Total time: {total_time:.4f} seconds")
    print(f"Time per iteration: {time_per_iteration:.4f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("---"*20)
    return total_time, time_per_iteration, precision, recall, f1

# 调整迭代次数并记录结果
iterations = [180]
results = []

for n_iter in iterations:
    sys.stdout = Logger("spacy_ner_test.log")  # 将stdout重定向到日志文件
    try:
        print("---"*20)
        print(f"Training with {n_iter} iterations")
        result = train_and_evaluate(n_iter, batch_size=32)
        results.append((n_iter, *result))
    finally:
        sys.stdout.reset()  # 将stdout重置为原始

# 打印最终结果
for result in results:
    print(f"Iterations: {result[0]}, Total time: {result[1]:.4f}, Time per iteration: {result[2]:.4f}, Precision: {result[3]:.4f}, Recall: {result[4]:.4f}, F1 Score: {result[5]:.4f}")
