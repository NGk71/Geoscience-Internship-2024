#只是用来测试模型的效果
#mixed_text 和 entity : 测试集
import torch
from transformers import BertTokenizer, BertForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import time

# Define the function to load the model and tokenizer
def load_model_and_tokenizer(model_dir):
    model = BertForTokenClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return model, tokenizer

# Define the function to make predictions and print the results
def evaluate_model(model, tokenizer, test_text, label_map):
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Tokenize and prepare input
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)

    # Convert predictions to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [reverse_label_map[pred.item()] for pred in predictions[0]]

    # Remove special tokens ([CLS] and [SEP])
    tokens = tokens[1:-1]
    predicted_labels = predicted_labels[1:-1]

    return tokens, predicted_labels

# Define the function to compute the classification report
def compute_classification_report(tokens, predicted_labels, true_entities, tokenizer, label_map):
    true_labels = ['O'] * len(tokens)
    for entity, label in true_entities:
        entity_tokens = tokenizer.tokenize(entity)
        start_idx = next((i for i, token in enumerate(tokens) if token == entity_tokens[0]), None)
        if start_idx is not None:
            true_labels[start_idx] = f"B-{label}"
            for i in range(1, len(entity_tokens)):
                if start_idx + i < len(true_labels):
                    true_labels[start_idx + i] = f"I-{label}"

    return true_labels

# Load model and tokenizer
model_dir = "./Regular Pruning"  # Change this to your model directory
model, tokenizer = load_model_and_tokenizer(model_dir)

# Define label map
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

# Test text
mixed_text = "2019年4月20日，施工人员到达狮平1井，并在0:00进行了车辆检查，随后在0:25召开了安全技术交底会，0:40正式开始施工。这次施工时间从2017年6月29日持续到7月6日，水平段长达597m，共有8段，每段间距为74.6m。第三十五段施工开始时，先排空了150.34m³的液体，打备压达到97.78MPa，随后洗井用了278.45m³的液体，同样打备压97.78MPa。泵送桥塞用液量为80.45m³，压裂压力为97.78MPa，而送球用液量为70.12m³。整个施工过程中，泵入了919.45m³的前置液，段塞加砂量为80.45m³，携砂液量为523.34m³，总砂量达到162.78m³，最高砂比为54.34%，平均砂比则为35.56%。在施工过程中，顶替液的总量为210.45m³，纤维总共使用了180Kg，净液量为9742.20m³，总酸量为250.00m³。施工最高压力记录为117.34MPa，破裂压力为117.34MPa，停泵油压为90.12MPa，最大排量达到18.78m³/min，平均排量为7.8m³/min，单段液量为1295m³，单段砂量为77.65m³，铺砂浓度为1.04m³/m。整个施工过程中共用时270分钟，在5:30结束施工后，压裂车组在5:45进行了设备整修，并于7:15离开井场。这次措施改造的体积达到767.8万方，其中单段改造体积为96万方。"

# True entities for the mixed text
mixed_entities = [
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

# Evaluate the model
start = time.time()
tokens, predicted_labels = evaluate_model(model, tokenizer, mixed_text, label_map)

# Print each word and its label
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")

# Compute the true labels
true_labels = compute_classification_report(tokens, predicted_labels, mixed_entities, tokenizer, label_map)

# Compute and print the classification report
print(classification_report([true_labels], [predicted_labels], scheme=IOB2))

# Compute and print the F1 score, precision, and recall
print("F1 Score:", f1_score([true_labels], [predicted_labels], scheme=IOB2))
print("Precision Score:", precision_score([true_labels], [predicted_labels], scheme=IOB2))
print("Recall Score:", recall_score([true_labels], [predicted_labels], scheme=IOB2))

print("Time:", time.time() - start)
