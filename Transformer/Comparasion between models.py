#仅作为比较
import time
from transformers import BertTokenizer

# 定义文本和标注
text = "第十段施工人员于2018年12月26日9:00到达井场，9:15检查好车辆并召开安全技术交底会，9:20开始施工。第十段排空用液68.00m³，打备压57.50MPa，洗井用液148.38m³，打备压55.00MPa，泵送桥塞用液9.77m³，打备压52.00MPa，送球用液50.68 m³，泵入前置液736.12m³,段塞加砂34.37m³,携砂液423.16m³,加砂95.73m³,最高砂比35.00%,平均砂比22.62%,顶替液101.23m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为74.70MPa,破裂压力73.60MPa,停泵油压58.30MPa，最大排量11.56m³/min，施工用时178分钟。16:13施工结束,压裂车组于16:20整修设备,17:50离开井场。狮平1井2017年6月29日至7月6日施工，该井水平段长597m，段数8段，段间距74.6m。施工最高施工压力108.70MPaPa，破裂压力72.30-108.70MPa, 施工排量6-8.7m3/min，砂量621.2m3，砂比12.86%。平均排量7.8m3/min、单段液量1295m3、单段砂量77.65m3、铺砂浓度1.04m3/m，。措施改造体积767.8万方，单段改造体积96万方。第1段压差滑套时多次未打掉。净液量9742.20m3,总酸量250.00m3"
entities = [ ("9:00", "ArrivalTime"), ("9:15", "VehicleCheckTime"), ("9:20", "StartTime"),
            ("68.00m³", "FluidForEmptying"), ("57.50MPa", "Pressure"), ("148.38m³", "FluidForWellFlushing"), ("55.00MPa", "Pressure"), 
            ("9.77m³", "FluidForBridgePlug"), ("52.00MPa", "Pressure"), ("50.68 m³", "FluidForBall"), ("736.12m³", "FluidForPrepad"), 
            ("34.37m³", "SandForPlug"), ("423.16m³", "FluidForProppant"), ("95.73m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.62%", "AverageSandRatio"), ("101.23m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("74.70MPa", "MaxPressure"), 
            ("73.60MPa", "FracturePressure"), ("58.30MPa", "PumpStopPressure"), ("11.56m³/min", "MaxFlowRate"), ("178分钟", "JobTime")]

# 初始化两个分词器
tokenizer_ernie = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
tokenizer_roberta = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

# 计算Ernie分词时间
start_time = time.time()
ernie_tokens = tokenizer_ernie.tokenize(text)
ernie_time = time.time() - start_time

# 计算Roberta分词时间
start_time = time.time()
roberta_tokens = tokenizer_roberta.tokenize(text)
roberta_time = time.time() - start_time

print(f"Ernie分词时间: {ernie_time:.6f}秒")
print(f"Roberta分词时间: {roberta_time:.6f}秒")

print(f"Ernie分词结果: {ernie_tokens}")
print(f"Roberta分词结果: {roberta_tokens}")

# 将实体转化为标签的函数
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

# 定义标签映射表
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

# 使用Ernie分词器转换实体为标签
ernie_labels = convert_entities_to_labels(text, entities, tokenizer_ernie, label_map)

# 使用Roberta分词器转换实体为标签
roberta_labels = convert_entities_to_labels(text, entities, tokenizer_roberta, label_map)

print(f"Ernie分词后的标签: {ernie_labels}")
print(f"Roberta分词后的标签: {roberta_labels}")
