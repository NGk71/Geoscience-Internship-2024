#训练数据示例
data =[
            (
        "狮平1井2017年6月29日至7月6日施工，该井水平段长597m，段数8段，段间距74.6m。施工最高施工压力108.70MPaPa，破裂压力72.30-108.70MPa, 施工排量6-8.7m3/min，砂量621.2m3，砂比12.86%。平均排量7.8m3/min、单段液量1295m3、单段砂量77.65m3、铺砂浓度1.04m3/m，。措施改造体积767.8万方，单段改造体积96万方。第1段压差滑套时多次未打掉。净液量9742.20m3,总酸量250.00m3",
        {
            "entities": [
                ("狮平1井", "WellName"), ("2017年6月29日至7月6日", "ConstructionTime"), 
                ("597m", "HorizontalSectionLength"), ("8段", "StageCount"), ("74.6m", "StageSpacing"), 
                ("108.70MPa", "MaxPressure"), ("72.30-108.70MPa", "FracturePressure"), 
                ("6-8.7m3/min", "FlowRate"), ("621.2m3", "SandVolume"), ("12.86%", "SandRatio"), 
                ("7.8m3/min", "AverageFlowRate"), ("1295m3", "SingleStageFluidVolume"), 
                ("77.65m3", "SingleStageSandVolume"), ("1.04m3/m", "ProppantConcentration"), 
                ("767.8万方", "ModificationVolume"), ("96万方", "SingleStageModificationVolume"), 
                ("9742.20m3", "NetFluidVolume"), ("250.00m3", "TotalAcidVolume")
            ]
        }
    )
]
# 标签映射表
label_map = {
    "O": 0,
    "B-ArrivalTime": 1, "I-ArrivalTime": 2,
    "B-VehicleCheckTime": 3, "I-VehicleCheckTime": 4,
    "B-StartTime": 5, "I-StartTime": 6,
    "B-FluidForEmptying": 7, "I-FluidForEmptying": 8,
    "B-Pressure": 9, "I-Pressure": 10,
    "B-FluidForWellFlushing": 11, "I-FluidForWellFlushing": 12,
    "B-FluidForBridgePlug": 13, "I-FluidForBridgePlug": 14,
    "B-FluidForBall": 15, "I-FluidForBall": 16,
    "B-FluidForPrepad": 17, "I-FluidForPrepad": 18,
    "B-SandForPlug": 19, "I-SandForPlug": 20,
    "B-FluidForProppant": 21, "I-FluidForProppant": 22,
    "B-Sand": 23, "I-Sand": 24,
    "B-MaxSandRatio": 25, "I-MaxSandRatio": 26,
    "B-AverageSandRatio": 27, "I-AverageSandRatio": 28,
    "B-DisplacementFluid": 29, "I-DisplacementFluid": 30,
    "B-Fiber": 31, "I-Fiber": 32,
    "B-MaxPressure": 33, "I-MaxPressure": 34,
    "B-FracturePressure": 35, "I-FracturePressure": 36,
    "B-PumpStopPressure": 37, "I-PumpStopPressure": 38,
    "B-MaxFlowRate": 39, "I-MaxFlowRate": 40,
    "B-JobTime": 41, "I-JobTime": 42,
    "B-WellName": 43, "I-WellName": 44,
    "B-ConstructionTime": 45, "I-ConstructionTime": 46,
    "B-HorizontalSectionLength": 47, "I-HorizontalSectionLength": 48,
    "B-StageCount": 49, "I-StageCount": 50,
    "B-StageSpacing": 51, "I-StageSpacing": 52,
    "B-TotalFluidVolume": 53, "I-TotalFluidVolume": 54,
    "B-NetFluidVolume": 55, "I-NetFluidVolume": 56,
    "B-TotalAcidVolume": 57, "I-TotalAcidVolume": 58,
    "B-FlowRate": 59, "I-FlowRate": 60,
    "B-SandVolume": 61, "I-SandVolume": 62,
    "B-SandRatio": 63, "I-SandRatio": 64,
    "B-AverageFlowRate": 65, "I-AverageFlowRate": 66,
    "B-SingleStageFluidVolume": 67, "I-SingleStageFluidVolume": 68,
    "B-SingleStageSandVolume": 69, "I-SingleStageSandVolume": 70,
    "B-ProppantConcentration": 71, "I-ProppantConcentration": 72,
    "B-ModificationVolume": 73, "I-ModificationVolume": 74,
    "B-SingleStageModificationVolume": 75, "I-SingleStageModificationVolume": 76
}

# 训练参数:
# output_dir: 模型保存路径
# num_train_epochs: 训练轮数
# per_device_train_batch_size: 每个设备的训练批次大小
# logging_dir: 日志保存路径
# logging_steps: 多少步记录一次日志

import time
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_callback import TrainerCallback
import pandas as pd

# data: 训练数据
df = pd.DataFrame(data, columns=["text", "entities"])
training_args = TrainingArguments(output_dir="./BERT", num_train_epochs=100, per_device_train_batch_size=2, logging_dir='./logs', logging_steps=10)
# 反向标签映射表
reverse_label_map = {v: k for k, v in label_map.items()}

# 样本数据准备
texts = df["text"].tolist()
entities = df["entities"].tolist()
# 预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")
model = BertForTokenClassification.from_pretrained("hfl/chinese-roberta-wwm-ext-large", num_labels=len(label_map))

# 选择设备, 如果有GPU则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model.to(device)

# 将实体转换为标签
def convert_entities_to_labels(text, entities, tokenizer, label_map):
    tokenized_input = tokenizer(text, truncation=True, padding=True, max_length=512)
    labels = ["O"] * len(tokenized_input["input_ids"])
    tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    # 遍历实体, 将实体转换为标签, B代表实体的开始, I代表实体的中间, O代表实体的外部
    for entity, entity_type in entities:
        entity_tokens = tokenizer.tokenize(entity)
        entity_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
        # 遍历tokenized_input["input_ids"], 找到实体的位置, 并将实体转换为标签
        for i in range(len(tokenized_input["input_ids"]) - len(entity_ids) + 1):
            if tokenized_input["input_ids"][i:i+len(entity_ids)] == entity_ids:
                labels[i] = f"B-{entity_type}"
                for j in range(1, len(entity_ids)):
                    labels[i + j] = f"I-{entity_type}"
                print(f"Found entity: {entity}, type: {entity_type}, at position: {i}")
                break
    
    return labels

# 读取数据
texts = df["text"].tolist()
entities = df["entities"].tolist()
labels = [convert_entities_to_labels(text, entity["entities"], tokenizer, label_map) for text, entity in zip(texts, entities)]
# 自定义数据集
class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map
    # 数据集长度
    def __len__(self):
        return len(self.texts)
    # 获取数据
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        tokenized_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')
        input_ids = tokenized_input['input_ids'].squeeze()
        attention_mask = tokenized_input['attention_mask'].squeeze()
        label_ids = [self.label_map.get(label, 0) for label in labels]
        label_ids = label_ids[:256] + [self.label_map["O"]] * (256 - len(label_ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(label_ids)}
# 创建数据集和数据加载器
train_dataset = NERDataset(texts, labels, tokenizer, label_map)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
# 训练参数
training_args = TrainingArguments(output_dir="./Regular", num_train_epochs=100, per_device_train_batch_size=2, logging_dir='./logs', logging_steps=10)
# 自定义回调函数
class CustomCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold
        self.losses = []
        self.learning_rates = []
    # 记录损失和学习率
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.losses.append(logs['loss'])
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])
    # 损失达到阈值时停止训练
    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log and latest_log['loss'] <= self.threshold:
                print(f"Stopping early as loss reached {self.threshold}")
                control.should_training_stop = True
    # 绘制损失和学习率图像
    def on_train_end(self, args, state, control, **kwargs):
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(self.losses, 'g-')
        ax2.plot(self.learning_rates, 'b-')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color='g')
        ax2.set_ylabel('Learning Rate', color='b')

        plt.title('Loss and Learning Rate over Iterations')
        plt.savefig("BERT_regular_loss_learning_rate_0.02.png")
        plt.show()
# 创建训练器
callback = CustomCallback(threshold=0.02)
# 数据收集器
def data_collator(data):
    # Pin memory only if tensors are on the CPU
    def pin_memory(tensor):
        if tensor.device == torch.device('cpu'):
            return tensor.pin_memory()
        return tensor
    # 将数据转换为字典
    return {
        "input_ids": pin_memory(torch.stack([f["input_ids"] for f in data])),
        "attention_mask": pin_memory(torch.stack([f["attention_mask"] for f in data])),
        "labels": pin_memory(torch.stack([f["labels"] for f in data])),
    }
# 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[callback]
)
# 训练时间
start = time.time()
trainer.train()
print(f"Total training time: {time.time() - start} seconds")
print(f"Training time per epoch: {(time.time() - start) / training_args.num_train_epochs} seconds")
# 保存模型和分词器
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)