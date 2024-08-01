import json

# 清空现有数据并添加一个空列表
with open('crf_test_data.json', 'w', encoding='utf-8') as f:
    json.dump([], f)

# 定义测试文本和实体


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

# 保存测试数据到文件
with open('crf_test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=4)

print("测试数据已保存到 crf_test_data.json 文件中")

"""第一段施工人员于2018年12月17日14:20到达井场，15:30检查好车辆并召开安全技术交底会，15:40开始施工。
第一段排空用液65.00 m³，试压90.00MPa, 尝压 1000Mpa,打备压36.60MPa，试挤用液19.47 m³，泵入预处理酸20.00m³，顶替液44.97m³，停泵反应24min后打备压54.50MPa，泵入前置液1007.39m³,段塞加砂23.98m³,携砂液415.76m³,加砂77.14m³,最高砂比35.00%,平均砂比18.55%,顶替液120.73m³,共加纤维40Kg,施工最高压力为86.00MPa,破裂压力86.00MPa,停泵油压55.30MPa，最大排量10.80m³/min，施工用时232分钟。00:42施工结束,压裂车组于00:50整修设备,1:50离开井场。"""
[
    ("14:20", "ArrivalTime"), ("15:30", "VehicleCheckTime"), ("15:40", "StartTime"),
    ("65.00 m³", "FluidForEmptying"), ("90.00MPa", "TestPressure"), ("1000Mpa", "TestPressure"), 
    ("36.60MPa", "Pressure"), ("19.47 m³", "FluidForSqueeze"), ("20.00m³", "FluidForPrepadAcid"), ("44.97m³", "DisplacementFluid"), 
    ("24min", "StopPumpReaction"), ("54.50MPa", "Pressure"), ("1007.39m³", "FluidForPrepad"), ("23.98m³", "SandForPlug"), 
    ("415.76m³", "FluidForProppant"), ("77.14m³", "Sand"), ("35.00%", "MaxSandRatio"), ("18.55%", "AverageSandRatio"), 
    ("120.73m³", "DisplacementFluid"), ("40Kg", "Fiber"), ("86.00MPa", "MaxPressure"), ("86.00MPa", "FracturePressure"), 
    ("55.30MPa", "PumpStopPressure"), ("10.80m³/min", "MaxFlowRate"), ("232 minutes", "JobTime")
]