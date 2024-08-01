# 这个是在训练基础上增加训练的代码，用于继续训练模型
import time
from datetime import datetime
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# 数据准备
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
    ),
    (
        "狮41H1井2017年12月27日至2018年1月2日施工，水平段长791m，设计10段施工3段，段间距79.1m；泵入总液量4948m3，排量6-8.4m3/min，砂量268m3，砂比17.92%。平均排量7.8m3/min、单段液量1649m3、单段砂量89m3、铺砂浓度1.13m3/m，最高施工压力87.31MPa，破裂压力83.70/84.60/84.43MPa。措施改造体积742万方，单段改造体积247.33万方。由于该井井眼轨迹复杂，从完井方式论证、施工准备、井筒准备历时68天；第4段泵送桥塞-射孔管柱遇阻，上提遇卡，活动解卡过程中电缆断脱，施工被迫中止。目前已捞获电缆约4123m，剩余落鱼约617m电缆及射孔管串。打捞进度缓慢。",
        {
            "entities": [
                ("狮41H1井", "WellName"), ("2017年12月27日至2018年1月2日", "ConstructionTime"), 
                ("791m", "HorizontalSectionLength"), ("设计10段", "DesignStageCount"), 
                ("施工3段", "ActualStageCount"), ("79.1m", "StageSpacing"), 
                ("4948m3", "TotalFluidVolume"), ("6-8.4m3/min", "FlowRate"), 
                ("268m3", "SandVolume"), ("17.92%", "SandRatio"), ("7.8m3/min", "AverageFlowRate"), 
                ("1649m3", "SingleStageFluidVolume"), ("89m3", "SingleStageSandVolume"), 
                ("1.13m3/m", "ProppantConcentration"), ("87.31MPa", "MaxPressure"), 
                ("83.70/84.60/84.43MPa", "FracturePressure"), ("742万方", "ModificationVolume"), 
                ("247.33万方", "SingleStageModificationVolume"), ("4123m", "RecoveredCableLength"), 
                ("617m", "RemainingCableLength")
            ]
        }
    ),
    (
        "狮49H3井2018年2月1日至2月8日施工。该水平段长525m，设计8段施工7段，段间距75m；泵入总液量10993m3，净液量10419.90m3,总酸量140.00m3.排量7.8-9.7m3/min，砂量573.7m3，砂比13.16%。平均排量9m3/min、单段液量1570m3、单段砂量81.96m3、铺砂浓度1.09m3/m，最高施工压力92.60Pa，破裂压力81.50-88.60MPa,。措施改造体积1606万方，单段改造体积229.43万方。",
        {
            "entities": [
                ("狮49H3井", "WellName"), ("2018年2月1日至2月8日", "ConstructionTime"), 
                ("525m", "HorizontalSectionLength"), ("设计8段", "DesignStageCount"), 
                ("施工7段", "ActualStageCount"), ("75m", "StageSpacing"), ("10993m3", "TotalFluidVolume"), 
                ("10419.90m3", "NetFluidVolume"), ("140.00m3", "TotalAcidVolume"), 
                ("7.8-9.7m3/min", "FlowRate"), ("573.7m3", "SandVolume"), ("13.16%", "SandRatio"), 
                ("9m3/min", "AverageFlowRate"), ("1570m3", "SingleStageFluidVolume"), 
                ("81.96m3", "SingleStageSandVolume"), ("1.09m3/m", "ProppantConcentration"), 
                ("92.60Pa", "MaxPressure"), ("81.50-88.60MPa", "FracturePressure"), 
                ("1606万方", "ModificationVolume"), ("229.43万方", "SingleStageModificationVolume")
            ]
        }
    ),
    (
        "狮49H1井2018年2月23日至3月4日压裂，水平段长769m，段数10段，段间距76.9m；泵入总液量15693m3，排量7.2-10.5m3/min，砂量938.2m3，砂比11.47%。平均排量8.9m3/min、单段液量1569.3m3、单段砂量93.82m3、铺砂浓度1.22m3/m，最高施工压力114.80MPa，破裂压力85.10-100.60MPa,。措施改造体积2800万方，单段改造体积280万方。",
        {
            "entities": [
                ("狮49H1井", "WellName"), ("2018年2月23日至3月4日", "ConstructionTime"), 
                ("769m", "HorizontalSectionLength"), ("10段", "StageCount"), 
                ("76.9m", "StageSpacing"), ("15693m3", "TotalFluidVolume"), 
                ("7.2-10.5m3/min", "FlowRate"), ("938.2m3", "SandVolume"), ("11.47%", "SandRatio"), 
                ("8.9m3/min", "AverageFlowRate"), ("1569.3m3", "SingleStageFluidVolume"), 
                ("93.82m3", "SingleStageSandVolume"), ("1.22m3/m", "ProppantConcentration"), 
                ("114.80MPa", "MaxPressure"), ("85.10-100.60MPa", "FracturePressure"), 
                ("2800万方", "ModificationVolume"), ("280万方", "SingleStageModificationVolume")
            ]
        }
    ),
    (
        "狮53H1井2018年3月14日至3月24日施工，该井水平段长571m，段数10段，段间距57.1m；泵入总液量18762m3，净液量14767.00m3,总酸量200.00m3。排量6-8.8m3/min，砂量899.2m3，砂比10.24%。平均排量7.6m3/min、单段液量1876.2m3、单段砂量89.92m3、铺砂浓度1.57m3/m，施工最高压力103.70MPa,破裂压力83.80-101.60MPa。措施改造体积4664万方，单段改造体积466.4万方。",
        {
            "entities": [
                ("狮53H1井", "WellName"), ("2018年3月14日至3月24日", "ConstructionTime"), 
                ("571m", "HorizontalSectionLength"), ("10段", "StageCount"), 
                ("57.1m", "StageSpacing"), ("18762m3", "TotalFluidVolume"), 
                ("14767.00m3", "NetFluidVolume"), ("200.00m3", "TotalAcidVolume"), 
                ("6-8.8m3/min", "FlowRate"), ("899.2m3", "SandVolume"), ("10.24%", "SandRatio"), 
                ("7.6m3/min", "AverageFlowRate"), ("1876.2m3", "SingleStageFluidVolume"), 
                ("89.92m3", "SingleStageSandVolume"), ("1.57m3/m", "ProppantConcentration"), 
                ("103.70MPa", "MaxPressure"), ("83.80-101.60MPa", "FracturePressure"), 
                ("4664万方", "ModificationVolume"), ("466.4万方", "SingleStageModificationVolume")
            ]
        }
    ),
    (
        "狮41H2井2018年4月9日至4月18日施工，该井水平段长682m，设计9段施工8段，段间距85.25m；泵入总液量16417m3，净液量15783.10m3,其中酸液量200.00m3，排量7.8-11m3/min，砂量762m3，砂比13.50%。平均排量10.2m3/min、单段液量2052m3、单段砂量95.25m3、铺砂浓度1.12m3/m，最高施工压力93.40MPa，破裂压力83-91.10MPa。措施改造体积1389万方，单段改造体积173.63万方。狮41H2井压裂第8段时狮41H3井（300米）测井施工中发生溢流。",
        {
            "entities": [
                ("狮41H2井", "WellName"), ("2018年4月9日至4月18日", "ConstructionTime"), 
                ("682m", "HorizontalSectionLength"), ("设计9段", "DesignStageCount"), 
                ("施工8段", "ActualStageCount"), ("85.25m", "StageSpacing"), 
                ("16417m3", "TotalFluidVolume"), ("15783.10m3", "NetFluidVolume"), 
                ("200.00m3", "AcidVolume"), ("7.8-11m3/min", "FlowRate"), 
                ("762m3", "SandVolume"), ("13.50%", "SandRatio"), ("10.2m3/min", "AverageFlowRate"), 
                ("2052m3", "SingleStageFluidVolume"), ("95.25m3", "SingleStageSandVolume"), 
                ("1.12m3/m", "ProppantConcentration"), ("93.40MPa", "MaxPressure"), 
                ("83-91.10MPa", "FracturePressure"), ("1389万方", "ModificationVolume"), 
                ("173.63万方", "SingleStageModificationVolume")
            ]
        }
    )
]

df = pd.DataFrame(data, columns=["text", "entities"])

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

reverse_label_map = {v: k for k, v in label_map.items()}

# 样本数据准备
texts = df["text"].tolist()
entities = df["entities"].tolist()

# 定义tokenizer和model
tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

# 使用 ignore_mismatched_sizes=True 加载模型
model = BertForTokenClassification.from_pretrained("./Regular", num_labels=len(label_map), ignore_mismatched_sizes=True)

# 手动调整分类层的大小
model.classifier = torch.nn.Linear(model.config.hidden_size, len(label_map))

# 将实体转化为标签
def convert_entities_to_labels(text, entities, tokenizer, label_map):
    tokenized_input = tokenizer(text, truncation=True, padding=True, max_length=512)
    labels = ["O"] * len(tokenized_input["input_ids"])
    tokenized_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    
    for entity, entity_type in entities:
        entity_tokens = tokenizer.tokenize(entity)
        entity_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
        
        for i in range(len(tokenized_input["input_ids"]) - len(entity_ids) + 1):
            if tokenized_input["input_ids"][i:i+len(entity_ids)] == entity_ids:
                labels[i] = f"B-{entity_type}"
                for j in range(1, len(entity_ids)):
                    labels[i + j] = f"I-{entity_type}"
                print(f"Found entity: {entity}, type: {entity_type}, at position: {i}")
                break
    
    print(f"Tokenized input: {tokenized_tokens}")
    print(f"Labels: {labels}")
    return labels

labels = [convert_entities_to_labels(text, entity["entities"], tokenizer, label_map) for text, entity in zip(texts, entities)]

class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_map):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_map = label_map

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        tokenized_input = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = tokenized_input['input_ids'].squeeze()
        attention_mask = tokenized_input['attention_mask'].squeeze()
        label_ids = [self.label_map.get(label, 0) for label in labels]
        label_ids = label_ids[:512] + [self.label_map["O"]] * (512 - len(label_ids))  # Ensure label_ids length is 512
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": torch.tensor(label_ids)}

train_dataset = NERDataset(texts, labels, tokenizer, label_map)

# 使用当前时间戳作为输出目录名称的一部分，确保每次保存的模型具有唯一的目录名称
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"./model_output_{timestamp}"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

training_args = TrainingArguments(output_dir=output_dir, num_train_epochs=3, per_device_train_batch_size=2, logging_dir='./logs', logging_steps=10) # 继续训练3个epochs
trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

start = time.time()
trainer.train()
print(f"Total training time: {time.time() - start} seconds")
print(f"Training time per epoch: {(time.time() - start) / 3} seconds")
# Save the trained model
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
