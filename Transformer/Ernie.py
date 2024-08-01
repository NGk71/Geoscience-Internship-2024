import time
from datetime import datetime
import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

# 数据准备
data =[
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
    ("第三段施工人员于2018年12月19日9:00到达井场，9:30检查好车辆并召开安全技术交底会，9:40开始施工。第三段排空用液68.00 m³，打备压48.50MPa，洗井用液146.08m³，打备压57.00MPa，泵送桥塞用液23.99m³，打备压54.00MPa，送球用液64.20 m³，泵入前置液360.08m³,段塞加砂12.19m³,携砂液280.62m³,加砂57.84m³,最高砂比35.00%,平均砂比20.61%,顶替液110.42m³,共加纤维100Kg,暂堵颗粒60 Kg,施工最高压力为70.00MPa,破裂压力70.00MPa,停泵油压56.90MPa，最大排量9.00m³/min，施工用时156分钟。16:50施工结束,压裂车组于17:00整修设备,17:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:30", "VehicleCheckTime"), ("9:40", "StartTime"),
            ("68.00 m³", "FluidForEmptying"), ("48.50MPa", "Pressure"), ("146.08m³", "FluidForWellFlushing"), ("57.00MPa", "Pressure"), 
            ("23.99m³", "FluidForBridgePlug"), ("54.00MPa", "Pressure"), ("64.20 m³", "FluidForBall"), ("360.08m³", "FluidForPrepad"), 
            ("12.19m³", "SandForPlug"), ("280.62m³", "FluidForProppant"), ("57.84m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("20.61%", "AverageSandRatio"), ("110.42m³", "DisplacementFluid"), ("100Kg", "Fiber"), ("70.00MPa", "MaxPressure"), 
            ("70.00MPa", "FracturePressure"), ("56.90MPa", "PumpStopPressure"), ("9.00m³/min", "MaxFlowRate"), ("156分钟", "JobTime")
        ]
    }
    ),
    ("第四段施工人员于2018年12月20日9:00到达井场，9:30检查好车辆并召开安全技术交底会，9:40开始施工。第四段排空用液73.00 m³，打备压55.00MPa，洗井用液150.23m³，打备压55.50MPa，泵送桥塞用液23.10m³，打备压53.50MPa，送球用液73.32 m³，泵入前置液600.13m³,段塞加砂25.08m³,携砂液416.27m³,加砂93.35m³,最高砂比35.00%,平均砂比22.42%,顶替液110.68m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为77.20MPa,破裂压力77.20MPa,停泵油压56.70MPa，最大排量11.25m³/min，施工用时182分钟。16:53施工结束,压裂车组于17:00整修设备,17:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:30", "VehicleCheckTime"), ("9:40", "StartTime"),
            ("73.00 m³", "FluidForEmptying"), ("55.00MPa", "Pressure"), ("150.23m³", "FluidForWellFlushing"), ("55.50MPa", "Pressure"), 
            ("23.10m³", "FluidForBridgePlug"), ("53.50MPa", "Pressure"), ("73.32 m³", "FluidForBall"), ("600.13m³", "FluidForPrepad"), 
            ("25.08m³", "SandForPlug"), ("416.27m³", "FluidForProppant"), ("93.35m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.42%", "AverageSandRatio"), ("110.68m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("77.20MPa", "MaxPressure"), 
            ("77.20MPa", "FracturePressure"), ("56.70MPa", "PumpStopPressure"), ("11.25m³/min", "MaxFlowRate"), ("182分钟", "JobTime")
        ]
    }
    ),
    ("第五段施工人员于2018年12月21日9:00到达井场，9:20检查好车辆并召开安全技术交底会，9:30开始施工。第五段排空用液75.00m³，打备压50.00MPa，洗井用液142.16m³，打备压53.90MPa，泵送桥塞用液21.77m³，打备压51.00MPa，送球用液43.25m³，泵入前置液780.15m³,段塞加砂30.98m³,携砂液389.08m³,加砂89.27m³,最高砂比38.00%,平均砂比22.94%,顶替液100.06m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为79.70MPa,破裂压力78.80MPa,停泵油压57.50MPa，最大排量11.07m³/min，施工用时182分钟。17:13施工结束,压裂车组于17:20整修设备,18:40离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:20", "VehicleCheckTime"), ("9:30", "StartTime"),
            ("75.00m³", "FluidForEmptying"), ("50.00MPa", "Pressure"), ("142.16m³", "FluidForWellFlushing"), ("53.90MPa", "Pressure"), 
            ("21.77m³", "FluidForBridgePlug"), ("51.00MPa", "Pressure"), ("43.25m³", "FluidForBall"), ("780.15m³", "FluidForPrepad"), 
            ("30.98m³", "SandForPlug"), ("389.08m³", "FluidForProppant"), ("89.27m³", "Sand"), ("38.00%", "MaxSandRatio"), 
            ("22.94%", "AverageSandRatio"), ("100.06m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("79.70MPa", "MaxPressure"), 
            ("78.80MPa", "FracturePressure"), ("57.50MPa", "PumpStopPressure"), ("11.07m³/min", "MaxFlowRate"), ("182分钟", "JobTime")
        ]
    }
    ),
    ("第六段施工人员于2018年12月22日9:30到达井场，9:55检查好车辆并召开安全技术交底会，10:00开始施工。第六段排空用液68.00m³，打备压55.00MPa，洗井用液151.68m³，打备压55.00MPa，泵送桥塞用液17.84m³，打备压53.70MPa，送球用液42.00 m³，泵入前置液790.03m³,段塞加砂25.40m³,携砂液389.18m³,加砂86.86m³,最高砂比38.00%,平均砂比22.31%,顶替液99.58m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为80.70MPa,破裂压力79.40MPa,停泵油压57.80MPa，最大排量10.75m³/min，施工用时196分钟。17:56施工结束,压裂车组于18:00整修设备,19:40离开井场。",
    {
        "entities": [
            ("9:30", "ArrivalTime"), ("9:55", "VehicleCheckTime"), ("10:00", "StartTime"),
            ("68.00m³", "FluidForEmptying"), ("55.00MPa", "Pressure"), ("151.68m³", "FluidForWellFlushing"), ("55.00MPa", "Pressure"), 
            ("17.84m³", "FluidForBridgePlug"), ("53.70MPa", "Pressure"), ("42.00 m³", "FluidForBall"), ("790.03m³", "FluidForPrepad"), 
            ("25.40m³", "SandForPlug"), ("389.18m³", "FluidForProppant"), ("86.86m³", "Sand"), ("38.00%", "MaxSandRatio"), 
            ("22.31%", "AverageSandRatio"), ("99.58m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("80.70MPa", "MaxPressure"), 
            ("79.40MPa", "FracturePressure"), ("57.80MPa", "PumpStopPressure"), ("10.75m³/min", "MaxFlowRate"), ("196分钟", "JobTime")
        ]
    }
    ),
    ("第七段施工人员于2018年12月23日9:00到达井场，9:15检查好车辆并召开安全技术交底会，9:20开始施工。第七段排空用液68.00m³，打备压55.00MPa，洗井用液142.54m³，打备压56.00MPa，泵送桥塞用液13.57m³，打备压53.50MPa，送球用液53.56 m³，泵入前置液800.06m³,段塞加砂32.39m³,携砂液392.47m³,加砂87.80m³,最高砂比35.00%,平均砂比22.37%,顶替液100.93m³,共加纤维100Kg,暂堵颗粒60 Kg,施工最高压力为80.10MPa,破裂压力80.10MPa,停泵油压59.40MPa，最大排量11.58m³/min，施工用时182分钟。17:19施工结束,压裂车组于17:25整修设备,18:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:15", "VehicleCheckTime"), ("9:20", "StartTime"),
            ("68.00m³", "FluidForEmptying"), ("55.00MPa", "Pressure"), ("142.54m³", "FluidForWellFlushing"), ("56.00MPa", "Pressure"), 
            ("13.57m³", "FluidForBridgePlug"), ("53.50MPa", "Pressure"), ("53.56 m³", "FluidForBall"), ("800.06m³", "FluidForPrepad"), 
            ("32.39m³", "SandForPlug"), ("392.47m³", "FluidForProppant"), ("87.80m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.37%", "AverageSandRatio"), ("100.93m³", "DisplacementFluid"), ("100Kg", "Fiber"), ("80.10MPa", "MaxPressure"), 
            ("80.10MPa", "FracturePressure"), ("59.40MPa", "PumpStopPressure"), ("11.58m³/min", "MaxFlowRate"), ("182分钟", "JobTime")
        ]
    }
    ),
    ("第八段施工人员于2018年12月24日9:30到达井场，11:35检查好车辆并召开安全技术交底会，11:40开始施工。第八段排空用液67.00m³，打备压56.00MPa，洗井用液144.48m³，打备压57.30MPa，泵送桥塞用液15.01m³，打备压52.50MPa，送球用液46.71m³，泵入前置液828.20m³,段塞加砂32.86m³,携砂液383.70m³,加砂85.45m³,最高砂比35.00%,平均砂比22.25%,顶替液101.00m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为82.30MPa,破裂压力82.30MPa,停泵油压58.30MPa，最大排量11.18m³/min，施工用时184分钟。18:59施工结束,压裂车组于19:10整修设备,20:20离开井场。",
    {
        "entities": [
            ("9:30", "ArrivalTime"), ("11:35", "VehicleCheckTime"), ("11:40", "StartTime"),
            ("67.00m³", "FluidForEmptying"), ("56.00MPa", "Pressure"), ("144.48m³", "FluidForWellFlushing"), ("57.30MPa", "Pressure"), 
            ("15.01m³", "FluidForBridgePlug"), ("52.50MPa", "Pressure"), ("46.71m³", "FluidForBall"), ("828.20m³", "FluidForPrepad"), 
            ("32.86m³", "SandForPlug"), ("383.70m³", "FluidForProppant"), ("85.45m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.25%", "AverageSandRatio"), ("101.00m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("82.30MPa", "MaxPressure"), 
            ("82.30MPa", "FracturePressure"), ("58.30MPa", "PumpStopPressure"), ("11.18m³/min", "MaxFlowRate"), ("184分钟", "JobTime")
        ]
    }
    ),
    ("第九段施工人员于2018年12月25日9:00到达井场，9:30检查好车辆并召开安全技术交底会，9:35开始施工。第九段排空用液74.00m³，打备压54.70MPa，洗井用液156.05m³，打备压56.50MPa，泵送桥塞用液12.06m³，打备压51.90MPa，送球用液44.89 m³，泵入前置液730.47m³,段塞加砂31.18m³,携砂液423.02m³,加砂98.99m³,最高砂比38.00%,平均砂比23.40%,顶替液100.46m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为78.70MPa,破裂压力77.00MPa,停泵油压59.60MPa，最大排量11.08m³/min，施工用时181分钟。17:38施工结束,压裂车组于17:45整修设备,18:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:30", "VehicleCheckTime"), ("9:35", "StartTime"),
            ("74.00m³", "FluidForEmptying"), ("54.70MPa", "Pressure"), ("156.05m³", "FluidForWellFlushing"), ("56.50MPa", "Pressure"), 
            ("12.06m³", "FluidForBridgePlug"), ("51.90MPa", "Pressure"), ("44.89 m³", "FluidForBall"), ("730.47m³", "FluidForPrepad"), 
            ("31.18m³", "SandForPlug"), ("423.02m³", "FluidForProppant"), ("98.99m³", "Sand"), ("38.00%", "MaxSandRatio"), 
            ("23.40%", "AverageSandRatio"), ("100.46m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("78.70MPa", "MaxPressure"), 
            ("77.00MPa", "FracturePressure"), ("59.60MPa", "PumpStopPressure"), ("11.08m³/min", "MaxFlowRate"), ("181分钟", "JobTime")
        ]
    }
    ),
    ("第十段施工人员于2018年12月26日9:00到达井场，9:15检查好车辆并召开安全技术交底会，9:20开始施工。第十段排空用液68.00m³，打备压57.50MPa，洗井用液148.38m³，打备压55.00MPa，泵送桥塞用液9.77m³，打备压52.00MPa，送球用液50.68 m³，泵入前置液736.12m³,段塞加砂34.37m³,携砂液423.16m³,加砂95.73m³,最高砂比35.00%,平均砂比22.62%,顶替液101.23m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为74.70MPa,破裂压力73.60MPa,停泵油压58.30MPa，最大排量11.56m³/min，施工用时178分钟。16:13施工结束,压裂车组于16:20整修设备,17:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:15", "VehicleCheckTime"), ("9:20", "StartTime"),
            ("68.00m³", "FluidForEmptying"), ("57.50MPa", "Pressure"), ("148.38m³", "FluidForWellFlushing"), ("55.00MPa", "Pressure"), 
            ("9.77m³", "FluidForBridgePlug"), ("52.00MPa", "Pressure"), ("50.68 m³", "FluidForBall"), ("736.12m³", "FluidForPrepad"), 
            ("34.37m³", "SandForPlug"), ("423.16m³", "FluidForProppant"), ("95.73m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.62%", "AverageSandRatio"), ("101.23m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("74.70MPa", "MaxPressure"), 
            ("73.60MPa", "FracturePressure"), ("58.30MPa", "PumpStopPressure"), ("11.56m³/min", "MaxFlowRate"), ("178分钟", "JobTime")
        ]
    }
    ),
    ("第十一段施工人员于2018年12月27日9:00到达井场，9:25检查好车辆并召开安全技术交底会，9:30开始施工。第十一段排空用液72.00 m³，打备压54.20MPa，洗井用液144.31m³，打备压53.80MPa，泵送桥塞用液9.00m³，打备压54.30MPa，送球用液38.18 m³，泵入前置液850.06m³,段塞加砂37.01m³,携砂液401.88m³,加砂93.13m³,最高砂比35.00%,平均砂比23.17%,顶替液90.09m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为75.50MPa,破裂压力75.50MPa,停泵油压56.50MPa，最大排量11.46m³/min，施工用时180分钟。17:26施工结束,压裂车组于17:30整修设备,18:50离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:25", "VehicleCheckTime"), ("9:30", "StartTime"),
            ("72.00 m³", "FluidForEmptying"), ("54.20MPa", "Pressure"), ("144.31m³", "FluidForWellFlushing"), ("53.80MPa", "Pressure"), 
            ("9.00m³", "FluidForBridgePlug"), ("54.30MPa", "Pressure"), ("38.18 m³", "FluidForBall"), ("850.06m³", "FluidForPrepad"), 
            ("37.01m³", "SandForPlug"), ("401.88m³", "FluidForProppant"), ("93.13m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("23.17%", "AverageSandRatio"), ("90.09m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("75.50MPa", "MaxPressure"), 
            ("75.50MPa", "FracturePressure"), ("56.50MPa", "PumpStopPressure"), ("11.46m³/min", "MaxFlowRate"), ("180分钟", "JobTime")
        ]
    }
    ),
    ("第十二段施工人员于2018年12月28日9:00到达井场，9:20检查好车辆并召开安全技术交底会，9:25开始施工。第十二段排空用液75.00 m³，打备压50.00MPa，洗井用液142.10m³，打备压52.40MPa，泵送桥塞用液5.11m³，打备压52.90MPa，送球用液37.23 m³，泵入前置液730.01m³,段塞加砂34.00m³,携砂液412.05m³,加砂96.11m³,最高砂比35.00%,平均砂比23.32%,顶替液94.40m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为71.70MPa,破裂压力71.70MPa,停泵油压54.80MPa，最大排量11.30m³/min，施工用时161分钟。16:50施工结束,压裂车组于17:00整修设备,18:30离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:20", "VehicleCheckTime"), ("9:25", "StartTime"),
            ("75.00 m³", "FluidForEmptying"), ("50.00MPa", "Pressure"), ("142.10m³", "FluidForWellFlushing"), ("52.40MPa", "Pressure"), 
            ("5.11m³", "FluidForBridgePlug"), ("52.90MPa", "Pressure"), ("37.23 m³", "FluidForBall"), ("730.01m³", "FluidForPrepad"), 
            ("34.00m³", "SandForPlug"), ("412.05m³", "FluidForProppant"), ("96.11m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("23.32%", "AverageSandRatio"), ("94.40m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("71.70MPa", "MaxPressure"), 
            ("71.70MPa", "FracturePressure"), ("54.80MPa", "PumpStopPressure"), ("11.30m³/min", "MaxFlowRate"), ("161分钟", "JobTime")
        ]
    }
    ),
    ("第十三段施工人员于2018年12月29日9:00到达井场，9:25检查好车辆并召开安全技术交底会，9:30开始施工。第十三段排空用液71.00 m³，打备压50.50MPa，洗井用液144.96m³，打备压51.00MPa，泵送桥塞用液3.61m³，打备压51.20MPa，送球用液37.38s m³，泵入前置液900.08m³,段塞加砂45.25m³,携砂液382.05m³,加砂85.04m³,最高砂比35.00%,平均砂比22.25%,顶替液99.40m³,共加纤维120Kg,暂堵颗粒60 Kg,施工最高压力为74.00MPa,破裂压力72.00MPa,停泵油压56.70MPa，最大排量11.25m³/min，施工用时175分钟。16:54施工结束,压裂车组于17:00整修设备,18:40离开井场。",
    {
        "entities": [
            ("9:00", "ArrivalTime"), ("9:25", "VehicleCheckTime"), ("9:30", "StartTime"),
            ("71.00 m³", "FluidForEmptying"), ("50.50MPa", "Pressure"), ("144.96m³", "FluidForWellFlushing"), ("51.00MPa", "Pressure"), 
            ("3.61m³", "FluidForBridgePlug"), ("51.20MPa", "Pressure"), ("37.38 m³", "FluidForBall"), ("900.08m³", "FluidForPrepad"), 
            ("45.25m³", "SandForPlug"), ("382.05m³", "FluidForProppant"), ("85.04m³", "Sand"), ("35.00%", "MaxSandRatio"), 
            ("22.25%", "AverageSandRatio"), ("99.40m³", "DisplacementFluid"), ("120Kg", "Fiber"), ("74.00MPa", "MaxPressure"), 
            ("72.00MPa", "FracturePressure"), ("56.70MPa", "PumpStopPressure"), ("11.25m³/min", "MaxFlowRate"), ("175分钟", "JobTime")
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
    "第十四段施工人员于2019年1月5日7:30到达井场，7:50检查好车辆并召开安全技术交底会，8:00开始施工。第十四段排空用液80.00 m³，打备压60.00MPa，洗井用液150.00 m³，打备压55.00MPa，泵送桥塞用液35.00m³，打备压57.00MPa，送球用液55.00 m³，泵入前置液760.00m³,段塞加砂30.00m³,携砂液490.00m³,加砂110.00m³,最高砂比36.00%,平均砂比22.00%,顶替液125.00m³,共加纤维130Kg,施工最高压力为83.00MPa,破裂压力82.00MPa,停泵油压57.50MPa，最大排量11.00m³/min，施工用时210分钟。11:30施工结束,压裂车组于11:45整修设备,13:00离开井场。",
    {
        "entities": [
            ("7:30", "ArrivalTime"), ("7:50", "VehicleCheckTime"), ("8:00", "StartTime"),
            ("80.00 m³", "FluidForEmptying"), ("60.00MPa", "Pressure"), ("150.00 m³", "FluidForWellFlushing"), ("55.00MPa", "Pressure"), 
            ("35.00m³", "FluidForBridgePlug"), ("57.00MPa", "Pressure"), ("55.00 m³", "FluidForBall"), ("760.00m³", "FluidForPrepad"), 
            ("30.00m³", "SandForPlug"), ("490.00m³", "FluidForProppant"), ("110.00m³", "Sand"), ("36.00%", "MaxSandRatio"), 
            ("22.00%", "AverageSandRatio"), ("125.00m³", "DisplacementFluid"), ("130Kg", "Fiber"), ("83.00MPa", "MaxPressure"), 
            ("82.00MPa", "FracturePressure"), ("57.50MPa", "PumpStopPressure"), ("11.00m³/min", "MaxFlowRate"), ("210分钟", "JobTime")
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
    ),
    (
        "第十五段施工人员于2019年1月10日6:45到达井场，7:00检查好车辆并召开安全技术交底会，7:15开始施工。第十五段排空用液85.00 m³，打备压62.00MPa，洗井用液155.00 m³，打备压56.00MPa，泵送桥塞用液37.00m³，打备压58.00MPa，送球用液57.00 m³，泵入前置液770.00m³,段塞加砂32.00m³,携砂液495.00m³,加砂112.00m³,最高砂比37.00%,平均砂比23.00%,顶替液130.00m³,共加纤维135Kg,施工最高压力为85.00MPa,破裂压力84.00MPa,停泵油压58.00MPa，最大排量11.50m³/min，施工用时215分钟。12:00施工结束,压裂车组于12:15整修设备,13:45离开井场。",
    {
        "entities": [
            ("6:45", "ArrivalTime"), ("7:00", "VehicleCheckTime"), ("7:15", "StartTime"),
            ("85.00 m³", "FluidForEmptying"), ("62.00MPa", "Pressure"), ("155.00 m³", "FluidForWellFlushing"), ("56.00MPa", "Pressure"), 
            ("37.00m³", "FluidForBridgePlug"), ("58.00MPa", "Pressure"), ("57.00 m³", "FluidForBall"), ("770.00m³", "FluidForPrepad"), 
            ("32.00m³", "SandForPlug"), ("495.00m³", "FluidForProppant"), ("112.00m³", "Sand"), ("37.00%", "MaxSandRatio"), 
            ("23.00%", "AverageSandRatio"), ("130.00m³", "DisplacementFluid"), ("135Kg", "Fiber"), ("85.00MPa", "MaxPressure"), 
            ("84.00MPa", "FracturePressure"), ("58.00MPa", "PumpStopPressure"), ("11.50m³/min", "MaxFlowRate"), ("215分钟", "JobTime")
        ]
    }
    ),    
    (
        "第十六段施工人员于2019年1月15日5:50到达井场，6:10检查好车辆并召开安全技术交底会，6:20开始施工。第十六段排空用液90.00 m³，打备压64.00MPa，洗井用液160.00 m³，打备压57.00MPa，泵送桥塞用液40.00m³，打备压59.00MPa，送球用液60.00 m³，泵入前置液780.00m³,段塞加砂34.00m³,携砂液500.00m³,加砂115.00m³,最高砂比38.00%,平均砂比24.00%,顶替液135.00m³,共加纤维140Kg,施工最高压力为87.00MPa,破裂压力86.00MPa,停泵油压59.50MPa，最大排量12.00m³/min，施工用时220分钟。10:00施工结束,压裂车组于10:15整修设备,11:30离开井场。",
        {
            "entities": [
                ("5:50", "ArrivalTime"), ("6:10", "VehicleCheckTime"), ("6:20", "StartTime"),
                ("90.00 m³", "FluidForEmptying"), ("64.00MPa", "Pressure"), ("160.00 m³", "FluidForWellFlushing"), ("57.00MPa", "Pressure"), 
                ("40.00m³", "FluidForBridgePlug"), ("59.00MPa", "Pressure"), ("60.00 m³", "FluidForBall"), ("780.00m³", "FluidForPrepad"), 
                ("34.00m³", "SandForPlug"), ("500.00m³", "FluidForProppant"), ("115.00m³", "Sand"), ("38.00%", "MaxSandRatio"), 
                ("24.00%", "AverageSandRatio"), ("135.00m³", "DisplacementFluid"), ("140Kg", "Fiber"), ("87.00MPa", "MaxPressure"), 
                ("86.00MPa", "FracturePressure"), ("59.50MPa", "PumpStopPressure"), ("12.00m³/min", "MaxFlowRate"), ("220分钟", "JobTime")
            ]
        }
    ),
        (
        "第十七段施工人员于2019年1月20日4:30到达井场，4:50检查好车辆并召开安全技术交底会，5:00开始施工。第十七段排空用液95.00 m³，打备压66.00MPa，洗井用液165.00 m³，打备压58.00MPa，泵送桥塞用液42.00m³，打备压60.00MPa，送球用液62.00 m³，泵入前置液790.00m³,段塞加砂36.00m³,携砂液505.00m³,加砂118.00m³,最高砂比39.00%,平均砂比25.00%,顶替液140.00m³,共加纤维145Kg,施工最高压力为89.00MPa,破裂压力88.00MPa,停泵油压60.50MPa，最大排量12.50m³/min，施工用时225分钟。9:00施工结束,压裂车组于9:15整修设备,10:45离开井场。",
        {
            "entities": [
                ("4:30", "ArrivalTime"), ("4:50", "VehicleCheckTime"), ("5:00", "StartTime"),
                ("95.00 m³", "FluidForEmptying"), ("66.00MPa", "Pressure"), ("165.00 m³", "FluidForWellFlushing"), ("58.00MPa", "Pressure"), 
                ("42.00m³", "FluidForBridgePlug"), ("60.00MPa", "Pressure"), ("62.00 m³", "FluidForBall"), ("790.00m³", "FluidForPrepad"), 
                ("36.00m³", "SandForPlug"), ("505.00m³", "FluidForProppant"), ("118.00m³", "Sand"), ("39.00%", "MaxSandRatio"), 
                ("25.00%", "AverageSandRatio"), ("140.00m³", "DisplacementFluid"), ("145Kg", "Fiber"), ("89.00MPa", "MaxPressure"), 
                ("88.00MPa", "FracturePressure"), ("60.50MPa", "PumpStopPressure"), ("12.50m³/min", "MaxFlowRate"), ("225分钟", "JobTime")
            ]
        }
    ),
    (
        "第十八段施工人员于2019年1月25日3:20到达井场，3:40检查好车辆并召开安全技术交底会，3:50开始施工。第十八段排空用液100.00 m³，打备压68.00MPa，洗井用液170.00 m³，打备压59.00MPa，泵送桥塞用液44.00m³，打备压61.00MPa，送球用液64.00 m³，泵入前置液800.00m³,段塞加砂38.00m³,携砂液510.00m³,加砂120.00m³,最高砂比40.00%,平均砂比26.00%,顶替液145.00m³,共加纤维150Kg,施工最高压力为91.00MPa,破裂压力90.00MPa,停泵油压61.50MPa，最大排量13.00m³/min，施工用时230分钟。8:00施工结束,压裂车组于8:15整修设备,9:30离开井场。",
        {
            "entities": [
                ("3:20", "ArrivalTime"), ("3:40", "VehicleCheckTime"), ("3:50", "StartTime"),
                ("100.00 m³", "FluidForEmptying"), ("68.00MPa", "Pressure"), ("170.00 m³", "FluidForWellFlushing"), ("59.00MPa", "Pressure"), 
                ("44.00m³", "FluidForBridgePlug"), ("61.00MPa", "Pressure"), ("64.00 m³", "FluidForBall"), ("800.00m³", "FluidForPrepad"), 
                ("38.00m³", "SandForPlug"), ("510.00m³", "FluidForProppant"), ("120.00m³", "Sand"), ("40.00%", "MaxSandRatio"), 
                ("26.00%", "AverageSandRatio"), ("145.00m³", "DisplacementFluid"), ("150Kg", "Fiber"), ("91.00MPa", "MaxPressure"), 
                ("90.00MPa", "FracturePressure"), ("61.50MPa", "PumpStopPressure"), ("13.00m³/min", "MaxFlowRate"), ("230分钟", "JobTime")
            ]
        }
    ),
    (
        "第十九段施工人员于2019年1月30日2:15到达井场，2:30检查好车辆并召开安全技术交底会，2:45开始施工。第十九段排空用液105.00 m³，打备压70.00MPa，洗井用液175.00 m³，打备压60.00MPa，泵送桥塞用液46.00m³，打备压62.00MPa，送球用液66.00 m³，泵入前置液810.00m³,段塞加砂40.00m³,携砂液515.00m³,加砂122.00m³,最高砂比41.00%,平均砂比27.00%,顶替液150.00m³,共加纤维155Kg,施工最高压力为93.00MPa,破裂压力92.00MPa,停泵油压62.50MPa，最大排量13.50m³/min，施工用时235分钟。7:00施工结束,压裂车组于7:15整修设备,8:45离开井场。",
        {
            "entities": [
                ("2:15", "ArrivalTime"), ("2:30", "VehicleCheckTime"), ("2:45", "StartTime"),
                ("105.00 m³", "FluidForEmptying"), ("70.00MPa", "Pressure"), ("175.00 m³", "FluidForWellFlushing"), ("60.00MPa", "Pressure"), 
                ("46.00m³", "FluidForBridgePlug"), ("62.00MPa", "Pressure"), ("66.00 m³", "FluidForBall"), ("810.00m³", "FluidForPrepad"), 
                ("40.00m³", "SandForPlug"), ("515.00m³", "FluidForProppant"), ("122.00m³", "Sand"), ("41.00%", "MaxSandRatio"), 
                ("27.00%", "AverageSandRatio"), ("150.00m³", "DisplacementFluid"), ("155Kg", "Fiber"), ("93.00MPa", "MaxPressure"), 
                ("92.00MPa", "FracturePressure"), ("62.50MPa", "PumpStopPressure"), ("13.50m³/min", "MaxFlowRate"), ("235分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十段施工人员于2019年2月4日1:10到达井场，1:25检查好车辆并召开安全技术交底会，1:35开始施工。第二十段排空用液110.00 m³，打备压72.00MPa，洗井用液180.00 m³，打备压61.00MPa，泵送桥塞用液48.00m³，打备压63.00MPa，送球用液68.00 m³，泵入前置液820.00m³,段塞加砂42.00m³,携砂液520.00m³,加砂124.00m³,最高砂比42.00%,平均砂比28.00%,顶替液155.00m³,共加纤维160Kg,施工最高压力为95.00MPa,破裂压力94.00MPa,停泵油压63.50MPa，最大排量14.00m³/min，施工用时240分钟。6:00施工结束,压裂车组于6:15整修设备,7:30离开井场。",
        {
            "entities": [
                ("1:10", "ArrivalTime"), ("1:25", "VehicleCheckTime"), ("1:35", "StartTime"),
                ("110.00 m³", "FluidForEmptying"), ("72.00MPa", "Pressure"), ("180.00 m³", "FluidForWellFlushing"), ("61.00MPa", "Pressure"), 
                ("48.00m³", "FluidForBridgePlug"), ("63.00MPa", "Pressure"), ("68.00 m³", "FluidForBall"), ("820.00m³", "FluidForPrepad"), 
                ("42.00m³", "SandForPlug"), ("520.00m³", "FluidForProppant"), ("124.00m³", "Sand"), ("42.00%", "MaxSandRatio"), 
                ("28.00%", "AverageSandRatio"), ("155.00m³", "DisplacementFluid"), ("160Kg", "Fiber"), ("95.00MPa", "MaxPressure"), 
                ("94.00MPa", "FracturePressure"), ("63.50MPa", "PumpStopPressure"), ("14.00m³/min", "MaxFlowRate"), ("240分钟", "JobTime")
            ]
        }
    ),
        (
        "第二十一段施工人员于2019年2月9日12:00到达井场，12:15检查好车辆并召开安全技术交底会，12:30开始施工。第二十一段排空用液115.00 m³，打备压74.00MPa，洗井用液185.00 m³，打备压62.00MPa，泵送桥塞用液50.00m³，打备压64.00MPa，送球用液70.00 m³，泵入前置液830.00m³,段塞加砂44.00m³,携砂液525.00m³,加砂126.00m³,最高砂比43.00%,平均砂比29.00%,顶替液160.00m³,共加纤维165Kg,施工最高压力为97.00MPa,破裂压力96.00MPa,停泵油压64.50MPa，最大排量14.50m³/min，施工用时245分钟。16:30施工结束,压裂车组于16:45整修设备,18:15离开井场。",
        {
            "entities": [
                ("12:00", "ArrivalTime"), ("12:15", "VehicleCheckTime"), ("12:30", "StartTime"),
                ("115.00 m³", "FluidForEmptying"), ("74.00MPa", "Pressure"), ("185.00 m³", "FluidForWellFlushing"), ("62.00MPa", "Pressure"), 
                ("50.00m³", "FluidForBridgePlug"), ("64.00MPa", "Pressure"), ("70.00 m³", "FluidForBall"), ("830.00m³", "FluidForPrepad"), 
                ("44.00m³", "SandForPlug"), ("525.00m³", "FluidForProppant"), ("126.00m³", "Sand"), ("43.00%", "MaxSandRatio"), 
                ("29.00%", "AverageSandRatio"), ("160.00m³", "DisplacementFluid"), ("165Kg", "Fiber"), ("97.00MPa", "MaxPressure"), 
                ("96.00MPa", "FracturePressure"), ("64.50MPa", "PumpStopPressure"), ("14.50m³/min", "MaxFlowRate"), ("245分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十二段施工人员于2019年2月14日11:00到达井场，11:15检查好车辆并召开安全技术交底会，11:30开始施工。第二十二段排空用液120.00 m³，打备压76.00MPa，洗井用液190.00 m³，打备压63.00MPa，泵送桥塞用液52.00m³，打备压65.00MPa，送球用液72.00 m³，泵入前置液840.00m³,段塞加砂46.00m³,携砂液530.00m³,加砂128.00m³,最高砂比44.00%,平均砂比30.00%,顶替液165.00m³,共加纤维170Kg,施工最高压力为99.00MPa,破裂压力98.00MPa,停泵油压65.50MPa，最大排量15.00m³/min，施工用时250分钟。16:40施工结束,压裂车组于16:55整修设备,18:25离开井场。",
        {
            "entities": [
                ("11:00", "ArrivalTime"), ("11:15", "VehicleCheckTime"), ("11:30", "StartTime"),
                ("120.00 m³", "FluidForEmptying"), ("76.00MPa", "Pressure"), ("190.00 m³", "FluidForWellFlushing"), ("63.00MPa", "Pressure"), 
                ("52.00m³", "FluidForBridgePlug"), ("65.00MPa", "Pressure"), ("72.00 m³", "FluidForBall"), ("840.00m³", "FluidForPrepad"), 
                ("46.00m³", "SandForPlug"), ("530.00m³", "FluidForProppant"), ("128.00m³", "Sand"), ("44.00%", "MaxSandRatio"), 
                ("30.00%", "AverageSandRatio"), ("165.00m³", "DisplacementFluid"), ("170Kg", "Fiber"), ("99.00MPa", "MaxPressure"), 
                ("98.00MPa", "FracturePressure"), ("65.50MPa", "PumpStopPressure"), ("15.00m³/min", "MaxFlowRate"), ("250分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十三段施工人员于2019年2月19日10:00到达井场，10:15检查好车辆并召开安全技术交底会，10:30开始施工。第二十三段排空用液125.00 m³，打备压78.00MPa，洗井用液195.00 m³，打备压64.00MPa，泵送桥塞用液54.00m³，打备压66.00MPa，送球用液74.00 m³，泵入前置液850.00m³,段塞加砂48.00m³,携砂液535.00m³,加砂130.00m³,最高砂比45.00%,平均砂比31.00%,顶替液170.00m³,共加纤维175Kg,施工最高压力为101.00MPa,破裂压力100.00MPa,停泵油压66.50MPa，最大排量15.50m³/min，施工用时255分钟。15:30施工结束,压裂车组于15:45整修设备,17:15离开井场。",
        {
            "entities": [
                ("10:00", "ArrivalTime"), ("10:15", "VehicleCheckTime"), ("10:30", "StartTime"),
                ("125.00 m³", "FluidForEmptying"), ("78.00MPa", "Pressure"), ("195.00 m³", "FluidForWellFlushing"), ("64.00MPa", "Pressure"), 
                ("54.00m³", "FluidForBridgePlug"), ("66.00MPa", "Pressure"), ("74.00 m³", "FluidForBall"), ("850.00m³", "FluidForPrepad"), 
                ("48.00m³", "SandForPlug"), ("535.00m³", "FluidForProppant"), ("130.00m³", "Sand"), ("45.00%", "MaxSandRatio"), 
                ("31.00%", "AverageSandRatio"), ("170.00m³", "DisplacementFluid"), ("175Kg", "Fiber"), ("101.00MPa", "MaxPressure"), 
                ("100.00MPa", "FracturePressure"), ("66.50MPa", "PumpStopPressure"), ("15.50m³/min", "MaxFlowRate"), ("255分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十四段施工人员于2019年2月24日9:00到达井场，9:15检查好车辆并召开安全技术交底会，9:30开始施工。第二十四段排空用液130.00 m³，打备压80.00MPa，洗井用液200.00 m³，打备压65.00MPa，泵送桥塞用液56.00m³，打备压67.00MPa，送球用液76.00 m³，泵入前置液860.00m³,段塞加砂50.00m³,携砂液540.00m³,加砂132.00m³,最高砂比46.00%,平均砂比32.00%,顶替液175.00m³,共加纤维180Kg,施工最高压力为103.00MPa,破裂压力102.00MPa,停泵油压67.50MPa，最大排量16.00m³/min，施工用时260分钟。14:30施工结束,压裂车组于14:45整修设备,16:15离开井场。",
        {
            "entities": [
                ("9:00", "ArrivalTime"), ("9:15", "VehicleCheckTime"), ("9:30", "StartTime"),
                ("130.00 m³", "FluidForEmptying"), ("80.00MPa", "Pressure"), ("200.00 m³", "FluidForWellFlushing"), ("65.00MPa", "Pressure"), 
                ("56.00m³", "FluidForBridgePlug"), ("67.00MPa", "Pressure"), ("76.00 m³", "FluidForBall"), ("860.00m³", "FluidForPrepad"), 
                ("50.00m³", "SandForPlug"), ("540.00m³", "FluidForProppant"), ("132.00m³", "Sand"), ("46.00%", "MaxSandRatio"), 
                ("32.00%", "AverageSandRatio"), ("175.00m³", "DisplacementFluid"), ("180Kg", "Fiber"), ("103.00MPa", "MaxPressure"), 
                ("102.00MPa", "FracturePressure"), ("67.50MPa", "PumpStopPressure"), ("16.00m³/min", "MaxFlowRate"), ("260分钟", "JobTime")
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
        "第二十五段施工人员于2019年3月1日8:00到达井场，8:15检查好车辆并召开安全技术交底会，8:30开始施工。第二十五段排空用液135.00 m³，打备压82.00MPa，洗井用液205.00 m³，打备压66.00MPa，泵送桥塞用液58.00m³，打备压68.00MPa，送球用液78.00 m³，泵入前置液870.00m³,段塞加砂52.00m³,携砂液545.00m³,加砂134.00m³,最高砂比47.00%,平均砂比33.00%,顶替液180.00m³,共加纤维185Kg,施工最高压力为105.00MPa,破裂压力104.00MPa,停泵油压68.50MPa，最大排量16.50m³/min，施工用时265分钟。13:30施工结束,压裂车组于13:45整修设备,15:15离开井场。",
        {
            "entities": [
                ("8:00", "ArrivalTime"), ("8:15", "VehicleCheckTime"), ("8:30", "StartTime"),
                ("135.00 m³", "FluidForEmptying"), ("82.00MPa", "Pressure"), ("205.00 m³", "FluidForWellFlushing"), ("66.00MPa", "Pressure"), 
                ("58.00m³", "FluidForBridgePlug"), ("68.00MPa", "Pressure"), ("78.00 m³", "FluidForBall"), ("870.00m³", "FluidForPrepad"), 
                ("52.00m³", "SandForPlug"), ("545.00m³", "FluidForProppant"), ("134.00m³", "Sand"), ("47.00%", "MaxSandRatio"), 
                ("33.00%", "AverageSandRatio"), ("180.00m³", "DisplacementFluid"), ("185Kg", "Fiber"), ("105.00MPa", "MaxPressure"), 
                ("104.00MPa", "FracturePressure"), ("68.50MPa", "PumpStopPressure"), ("16.50m³/min", "MaxFlowRate"), ("265分钟", "JobTime")
            ]
        }
    ),
        (
        "第二十六段施工人员于2019年3月6日7:00到达井场，7:15检查好车辆并召开安全技术交底会，7:30开始施工。第二十六段排空用液140.00 m³，打备压84.00MPa，洗井用液210.00 m³，打备压67.00MPa，泵送桥塞用液60.00m³，打备压69.00MPa，送球用液80.00 m³，泵入前置液880.00m³,段塞加砂54.00m³,携砂液550.00m³,加砂136.00m³,最高砂比48.00%,平均砂比34.00%,顶替液185.00m³,共加纤维190Kg,施工最高压力为107.00MPa,破裂压力106.00MPa,停泵油压69.50MPa，最大排量17.00m³/min，施工用时270分钟。12:30施工结束,压裂车组于12:45整修设备,14:15离开井场。",
        {
            "entities": [
                ("7:00", "ArrivalTime"), ("7:15", "VehicleCheckTime"), ("7:30", "StartTime"),
                ("140.00 m³", "FluidForEmptying"), ("84.00MPa", "Pressure"), ("210.00 m³", "FluidForWellFlushing"), ("67.00MPa", "Pressure"), 
                ("60.00m³", "FluidForBridgePlug"), ("69.00MPa", "Pressure"), ("80.00 m³", "FluidForBall"), ("880.00m³", "FluidForPrepad"), 
                ("54.00m³", "SandForPlug"), ("550.00m³", "FluidForProppant"), ("136.00m³", "Sand"), ("48.00%", "MaxSandRatio"), 
                ("34.00%", "AverageSandRatio"), ("185.00m³", "DisplacementFluid"), ("190Kg", "Fiber"), ("107.00MPa", "MaxPressure"), 
                ("106.00MPa", "FracturePressure"), ("69.50MPa", "PumpStopPressure"), ("17.00m³/min", "MaxFlowRate"), ("270分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十七段施工人员于2019年3月11日6:00到达井场，6:15检查好车辆并召开安全技术交底会，6:30开始施工。第二十七段排空用液145.00 m³，打备压86.00MPa，洗井用液215.00 m³，打备压68.00MPa，泵送桥塞用液62.00m³，打备压70.00MPa，送球用液82.00 m³，泵入前置液890.00m³,段塞加砂56.00m³,携砂液555.00m³,加砂138.00m³,最高砂比49.00%,平均砂比35.00%,顶替液190.00m³,共加纤维195Kg,施工最高压力为109.00MPa,破裂压力108.00MPa,停泵油压70.50MPa，最大排量17.50m³/min，施工用时275分钟。11:30施工结束,压裂车组于11:45整修设备,13:15离开井场。",
        {
            "entities": [
                ("6:00", "ArrivalTime"), ("6:15", "VehicleCheckTime"), ("6:30", "StartTime"),
                ("145.00 m³", "FluidForEmptying"), ("86.00MPa", "Pressure"), ("215.00 m³", "FluidForWellFlushing"), ("68.00MPa", "Pressure"), 
                ("62.00m³", "FluidForBridgePlug"), ("70.00MPa", "Pressure"), ("82.00 m³", "FluidForBall"), ("890.00m³", "FluidForPrepad"), 
                ("56.00m³", "SandForPlug"), ("555.00m³", "FluidForProppant"), ("138.00m³", "Sand"), ("49.00%", "MaxSandRatio"), 
                ("35.00%", "AverageSandRatio"), ("190.00m³", "DisplacementFluid"), ("195Kg", "Fiber"), ("109.00MPa", "MaxPressure"), 
                ("108.00MPa", "FracturePressure"), ("70.50MPa", "PumpStopPressure"), ("17.50m³/min", "MaxFlowRate"), ("275分钟", "JobTime")
            ]
        }
    ),
    (
        "第二十八段施工人员于2019年3月16日5:00到达井场，5:15检查好车辆并召开安全技术交底会，5:30开始施工。第二十八段排空用液150.00 m³，打备压88.00MPa，洗井用液220.00 m³，打备压69.00MPa，泵送桥塞用液64.00m³，打备压71.00MPa，送球用液84.00 m³，泵入前置液900.00m³,段塞加砂58.00m³,携砂液560.00m³,加砂140.00m³,最高砂比50.00%,平均砂比36.00%,顶替液195.00m³,共加纤维200Kg,施工最高压力为111.00MPa,破裂压力110.00MPa,停泵油压71.50MPa，最大排量18.00m³/min，施工用时280分钟。10:30施工结束,压裂车组于10:45整修设备,12:15离开井场。",
        {
            "entities": [
                ("5:00", "ArrivalTime"), ("5:15", "VehicleCheckTime"), ("5:30", "StartTime"),
                ("150.00 m³", "FluidForEmptying"), ("88.00MPa", "Pressure"), ("220.00 m³", "FluidForWellFlushing"), ("69.00MPa", "Pressure"), 
                ("64.00m³", "FluidForBridgePlug"), ("71.00MPa", "Pressure"), ("84.00 m³", "FluidForBall"), ("900.00m³", "FluidForPrepad"), 
                ("58.00m³", "SandForPlug"), ("560.00m³", "FluidForProppant"), ("140.00m³", "Sand"), ("50.00%", "MaxSandRatio"), 
                ("36.00%", "AverageSandRatio"), ("195.00m³", "DisplacementFluid"), ("200Kg", "Fiber"), ("111.00MPa", "MaxPressure"), 
                ("110.00MPa", "FracturePressure"), ("71.50MPa", "PumpStopPressure"), ("18.00m³/min", "MaxFlowRate"), ("280分钟", "JobTime")
            ]
        }
    ),
        (
        "第二十九段施工人员于2019年3月21日4:22到达井场，4:45检查好车辆并召开安全技术交底会，5:05开始施工。第二十九段排空用液112.34 m³，打备压91.78MPa，洗井用液201.56 m³，打备压91.78MPa，泵送桥塞用液50.12m³，打备压91.78MPa，送球用液42.33 m³，泵入前置液803.56m³,段塞加砂61.23m³,携砂液453.67m³,加砂132.45m³,最高砂比42.34%,平均砂比27.89%,顶替液178.12m³,共加纤维145Kg,施工最高压力为102.45MPa,破裂压力102.45MPa,停泵油压78.23MPa，最大排量14.67m³/min，施工用时210分钟。10:30施工结束,压裂车组于10:45整修设备,12:15离开井场。",
        {
            "entities": [
                ("4:22", "ArrivalTime"), ("4:45", "VehicleCheckTime"), ("5:05", "StartTime"),
                ("112.34 m³", "FluidForEmptying"), ("91.78MPa", "Pressure"), ("201.56 m³", "FluidForWellFlushing"), ("91.78MPa", "Pressure"), 
                ("50.12m³", "FluidForBridgePlug"), ("91.78MPa", "Pressure"), ("42.33 m³", "FluidForBall"), ("803.56m³", "FluidForPrepad"), 
                ("61.23m³", "SandForPlug"), ("453.67m³", "FluidForProppant"), ("132.45m³", "Sand"), ("42.34%", "MaxSandRatio"), 
                ("27.89%", "AverageSandRatio"), ("178.12m³", "DisplacementFluid"), ("145Kg", "Fiber"), ("102.45MPa", "MaxPressure"), 
                ("102.45MPa", "FracturePressure"), ("78.23MPa", "PumpStopPressure"), ("14.67m³/min", "MaxFlowRate"), ("210分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十段施工人员于2019年3月26日5:50到达井场，6:15检查好车辆并召开安全技术交底会，6:35开始施工。第三十段排空用液125.56 m³，打备压85.67MPa，洗井用液215.78 m³，打备压85.67MPa，泵送桥塞用液55.89m³，打备压85.67MPa，送球用液48.12 m³，泵入前置液856.78m³,段塞加砂64.34m³,携砂液467.45m³,加砂145.67m³,最高砂比44.56%,平均砂比29.34%,顶替液185.45m³,共加纤维155Kg,施工最高压力为105.56MPa,破裂压力105.56MPa,停泵油压80.12MPa，最大排量15.34m³/min，施工用时220分钟。11:00施工结束,压裂车组于11:15整修设备,12:45离开井场。",
        {
            "entities": [
                ("5:50", "ArrivalTime"), ("6:15", "VehicleCheckTime"), ("6:35", "StartTime"),
                ("125.56 m³", "FluidForEmptying"), ("85.67MPa", "Pressure"), ("215.78 m³", "FluidForWellFlushing"), ("85.67MPa", "Pressure"), 
                ("55.89m³", "FluidForBridgePlug"), ("85.67MPa", "Pressure"), ("48.12 m³", "FluidForBall"), ("856.78m³", "FluidForPrepad"), 
                ("64.34m³", "SandForPlug"), ("467.45m³", "FluidForProppant"), ("145.67m³", "Sand"), ("44.56%", "MaxSandRatio"), 
                ("29.34%", "AverageSandRatio"), ("185.45m³", "DisplacementFluid"), ("155Kg", "Fiber"), ("105.56MPa", "MaxPressure"), 
                ("105.56MPa", "FracturePressure"), ("80.12MPa", "PumpStopPressure"), ("15.34m³/min", "MaxFlowRate"), ("220分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十一段施工人员于2019年3月31日4:30到达井场，4:55检查好车辆并召开安全技术交底会，5:10开始施工。第三十一段排空用液130.45 m³，打备压88.56MPa，洗井用液230.67 m³，打备压88.56MPa，泵送桥塞用液60.78m³，打备压88.56MPa，送球用液52.34 m³，泵入前置液870.12m³,段塞加砂67.45m³,携砂液478.56m³,加砂148.78m³,最高砂比46.12%,平均砂比30.45%,顶替液190.34m³,共加纤维160Kg,施工最高压力为108.45MPa,破裂压力108.45MPa,停泵油压82.34MPa，最大排量16.12m³/min，施工用时230分钟。9:00施工结束,压裂车组于9:15整修设备,10:45离开井场。",
        {
            "entities": [
                ("4:30", "ArrivalTime"), ("4:55", "VehicleCheckTime"), ("5:10", "StartTime"),
                ("130.45 m³", "FluidForEmptying"), ("88.56MPa", "Pressure"), ("230.67 m³", "FluidForWellFlushing"), ("88.56MPa", "Pressure"), 
                ("60.78m³", "FluidForBridgePlug"), ("88.56MPa", "Pressure"), ("52.34 m³", "FluidForBall"), ("870.12m³", "FluidForPrepad"), 
                ("67.45m³", "SandForPlug"), ("478.56m³", "FluidForProppant"), ("148.78m³", "Sand"), ("46.12%", "MaxSandRatio"), 
                ("30.45%", "AverageSandRatio"), ("190.34m³", "DisplacementFluid"), ("160Kg", "Fiber"), ("108.45MPa", "MaxPressure"), 
                ("108.45MPa", "FracturePressure"), ("82.34MPa", "PumpStopPressure"), ("16.12m³/min", "MaxFlowRate"), ("230分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十二段施工人员于2019年4月5日3:15到达井场，3:40检查好车辆并召开安全技术交底会，4:00开始施工。第三十二段排空用液135.78 m³，打备压90.34MPa，洗井用液245.12 m³，打备压90.34MPa，泵送桥塞用液65.34m³，打备压90.34MPa，送球用液56.78 m³，泵入前置液885.34m³,段塞加砂70.45m³,携砂液489.12m³,加砂152.34m³,最高砂比48.34%,平均砂比31.56%,顶替液195.78m³,共加纤维165Kg,施工最高压力为110.78MPa,破裂压力110.78MPa,停泵油压84.12MPa，最大排量16.78m³/min，施工用时240分钟。8:30施工结束,压裂车组于8:45整修设备,10:15离开井场。",
        {
            "entities": [
                ("3:15", "ArrivalTime"), ("3:40", "VehicleCheckTime"), ("4:00", "StartTime"),
                ("135.78 m³", "FluidForEmptying"), ("90.34MPa", "Pressure"), ("245.12 m³", "FluidForWellFlushing"), ("90.34MPa", "Pressure"), 
                ("65.34m³", "FluidForBridgePlug"), ("90.34MPa", "Pressure"), ("56.78 m³", "FluidForBall"), ("885.34m³", "FluidForPrepad"), 
                ("70.45m³", "SandForPlug"), ("489.12m³", "FluidForProppant"), ("152.34m³", "Sand"), ("48.34%", "MaxSandRatio"), 
                ("31.56%", "AverageSandRatio"), ("195.78m³", "DisplacementFluid"), ("165Kg", "Fiber"), ("110.78MPa", "MaxPressure"), 
                ("110.78MPa", "FracturePressure"), ("84.12MPa", "PumpStopPressure"), ("16.78m³/min", "MaxFlowRate"), ("240分钟", "JobTime")
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
        "第三十三段施工人员于2019年4月10日2:10到达井场，2:35检查好车辆并召开安全技术交底会，2:50开始施工。第三十三段排空用液140.12 m³，打备压92.78MPa，洗井用液256.34 m³，打备压92.78MPa，泵送桥塞用液70.45m³，打备压92.78MPa，送球用液61.23 m³，泵入前置液896.78m³,段塞加砂73.67m³,携砂液500.45m³,加砂155.78m³,最高砂比50.45%,平均砂比32.78%,顶替液200.34m³,共加纤维170Kg,施工最高压力为112.34MPa,破裂压力112.34MPa,停泵油压86.12MPa，最大排量17.34m³/min，施工用时250分钟。7:30施工结束,压裂车组于7:45整修设备,9:15离开井场。",
        {
            "entities": [
                ("2:10", "ArrivalTime"), ("2:35", "VehicleCheckTime"), ("2:50", "StartTime"),
                ("140.12 m³", "FluidForEmptying"), ("92.78MPa", "Pressure"), ("256.34 m³", "FluidForWellFlushing"), ("92.78MPa", "Pressure"), 
                ("70.45m³", "FluidForBridgePlug"), ("92.78MPa", "Pressure"), ("61.23 m³", "FluidForBall"), ("896.78m³", "FluidForPrepad"), 
                ("73.67m³", "SandForPlug"), ("500.45m³", "FluidForProppant"), ("155.78m³", "Sand"), ("50.45%", "MaxSandRatio"), 
                ("32.78%", "AverageSandRatio"), ("200.34m³", "DisplacementFluid"), ("170Kg", "Fiber"), ("112.34MPa", "MaxPressure"), 
                ("112.34MPa", "FracturePressure"), ("86.12MPa", "PumpStopPressure"), ("17.34m³/min", "MaxFlowRate"), ("250分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十四段施工人员于2019年4月15日1:05到达井场，1:30检查好车辆并召开安全技术交底会，1:45开始施工。第三十四段排空用液145.56 m³，打备压95.12MPa，洗井用液267.56 m³，打备压95.12MPa，泵送桥塞用液75.78m³，打备压95.12MPa，送球用液65.34 m³，泵入前置液908.12m³,段塞加砂76.89m³,携砂液511.78m³,加砂159.12m³,最高砂比52.56%,平均砂比34.12%,顶替液205.12m³,共加纤维175Kg,施工最高压力为114.78MPa,破裂压力114.78MPa,停泵油压88.34MPa，最大排量18.12m³/min，施工用时260分钟。6:00施工结束,压裂车组于6:15整修设备,7:45离开井场。",
        {
            "entities": [
                ("1:05", "ArrivalTime"), ("1:30", "VehicleCheckTime"), ("1:45", "StartTime"),
                ("145.56 m³", "FluidForEmptying"), ("95.12MPa", "Pressure"), ("267.56 m³", "FluidForWellFlushing"), ("95.12MPa", "Pressure"), 
                ("75.78m³", "FluidForBridgePlug"), ("95.12MPa", "Pressure"), ("65.34 m³", "FluidForBall"), ("908.12m³", "FluidForPrepad"), 
                ("76.89m³", "SandForPlug"), ("511.78m³", "FluidForProppant"), ("159.12m³", "Sand"), ("52.56%", "MaxSandRatio"), 
                ("34.12%", "AverageSandRatio"), ("205.12m³", "DisplacementFluid"), ("175Kg", "Fiber"), ("114.78MPa", "MaxPressure"), 
                ("114.78MPa", "FracturePressure"), ("88.34MPa", "PumpStopPressure"), ("18.12m³/min", "MaxFlowRate"), ("260分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十五段施工人员于2019年4月20日0:00到达井场，0:25检查好车辆并召开安全技术交底会，0:40开始施工。第三十五段排空用液150.34 m³，打备压97.78MPa，洗井用液278.45 m³，打备压97.78MPa，泵送桥塞用液80.45m³，打备压97.78MPa，送球用液70.12 m³，泵入前置液919.45m³,段塞加砂80.45m³,携砂液523.34m³,加砂162.78m³,最高砂比54.34%,平均砂比35.56%,顶替液210.45m³,共加纤维180Kg,施工最高压力为117.34MPa,破裂压力117.34MPa,停泵油压90.12MPa，最大排量18.78m³/min，施工用时270分钟。5:30施工结束,压裂车组于5:45整修设备,7:15离开井场。",
        {
            "entities": [
                ("0:00", "ArrivalTime"), ("0:25", "VehicleCheckTime"), ("0:40", "StartTime"),
                ("150.34 m³", "FluidForEmptying"), ("97.78MPa", "Pressure"), ("278.45 m³", "FluidForWellFlushing"), ("97.78MPa", "Pressure"), 
                ("80.45m³", "FluidForBridgePlug"), ("97.78MPa", "Pressure"), ("70.12 m³", "FluidForBall"), ("919.45m³", "FluidForPrepad"), 
                ("80.45m³", "SandForPlug"), ("523.34m³", "FluidForProppant"), ("162.78m³", "Sand"), ("54.34%", "MaxSandRatio"), 
                ("35.56%", "AverageSandRatio"), ("210.45m³", "DisplacementFluid"), ("180Kg", "Fiber"), ("117.34MPa", "MaxPressure"), 
                ("117.34MPa", "FracturePressure"), ("90.12MPa", "PumpStopPressure"), ("18.78m³/min", "MaxFlowRate"), ("270分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十六段施工人员于2019年4月25日23:50到达井场，0:15检查好车辆并召开安全技术交底会，0:30开始施工。第三十六段排空用液155.12 m³，打备压100.34MPa，洗井用液289.12 m³，打备压100.34MPa，泵送桥塞用液85.78m³，打备压100.34MPa，送球用液75.34 m³，泵入前置液930.78m³,段塞加砂83.67m³,携砂液534.12m³,加砂165.45m³,最高砂比56.12%,平均砂比36.78%,顶替液215.78m³,共加纤维185Kg,施工最高压力为119.78MPa,破裂压力119.78MPa,停泵油压92.34MPa，最大排量19.12m³/min，施工用时280分钟。5:30施工结束,压裂车组于5:45整修设备,7:15离开井场。",
        {
            "entities": [
                ("23:50", "ArrivalTime"), ("0:15", "VehicleCheckTime"), ("0:30", "StartTime"),
                ("155.12 m³", "FluidForEmptying"), ("100.34MPa", "Pressure"), ("289.12 m³", "FluidForWellFlushing"), ("100.34MPa", "Pressure"), 
                ("85.78m³", "FluidForBridgePlug"), ("100.34MPa", "Pressure"), ("75.34 m³", "FluidForBall"), ("930.78m³", "FluidForPrepad"), 
                ("83.67m³", "SandForPlug"), ("534.12m³", "FluidForProppant"), ("165.45m³", "Sand"), ("56.12%", "MaxSandRatio"), 
                ("36.78%", "AverageSandRatio"), ("215.78m³", "DisplacementFluid"), ("185Kg", "Fiber"), ("119.78MPa", "MaxPressure"), 
                ("119.78MPa", "FracturePressure"), ("92.34MPa", "PumpStopPressure"), ("19.12m³/min", "MaxFlowRate"), ("280分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十七段施工人员于2019年4月30日22:40到达井场，23:05检查好车辆并召开安全技术交底会，23:20开始施工。第三十七段排空用液160.45 m³，打备压102.78MPa，洗井用液300.45 m³，打备压102.78MPa，泵送桥塞用液90.45m³，打备压102.78MPa，送球用液80.12 m³，泵入前置液941.45m³,段塞加砂87.45m³,携砂液545.78m³,加砂168.12m³,最高砂比57.78%,平均砂比37.34%,顶替液221.45m³,共加纤维190Kg,施工最高压力为122.78MPa,破裂压力122.78MPa,停泵油压94.12MPa，最大排量19.78m³/min，施工用时290分钟。4:30施工结束,压裂车组于4:45整修设备,6:15离开井场。",
        {
            "entities": [
                ("22:40", "ArrivalTime"), ("23:05", "VehicleCheckTime"), ("23:20", "StartTime"),
                ("160.45 m³", "FluidForEmptying"), ("102.78MPa", "Pressure"), ("300.45 m³", "FluidForWellFlushing"), ("102.78MPa", "Pressure"), 
                ("90.45m³", "FluidForBridgePlug"), ("102.78MPa", "Pressure"), ("80.12 m³", "FluidForBall"), ("941.45m³", "FluidForPrepad"), 
                ("87.45m³", "SandForPlug"), ("545.78m³", "FluidForProppant"), ("168.12m³", "Sand"), ("57.78%", "MaxSandRatio"), 
                ("37.34%", "AverageSandRatio"), ("221.45m³", "DisplacementFluid"), ("190Kg", "Fiber"), ("122.78MPa", "MaxPressure"), 
                ("122.78MPa", "FracturePressure"), ("94.12MPa", "PumpStopPressure"), ("19.78m³/min", "MaxFlowRate"), ("290分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十八段施工人员于2019年5月5日21:30到达井场，21:55检查好车辆并召开安全技术交底会，22:10开始施工。第三十八段排空用液165.12 m³，打备压105.34MPa，洗井用液311.12 m³，打备压105.34MPa，泵送桥塞用液95.78m³，打备压105.34MPa，送球用液85.34 m³，泵入前置液952.78m³,段塞加砂90.67m³,携砂液556.12m³,加砂170.78m³,最高砂比59.12%,平均砂比38.56%,顶替液225.78m³,共加纤维195Kg,施工最高压力为125.34MPa,破裂压力125.34MPa,停泵油压96.34MPa，最大排量20.12m³/min，施工用时300分钟。3:30施工结束,压裂车组于3:45整修设备,5:15离开井场。",
        {
            "entities": [
                ("21:30", "ArrivalTime"), ("21:55", "VehicleCheckTime"), ("22:10", "StartTime"),
                ("165.12 m³", "FluidForEmptying"), ("105.34MPa", "Pressure"), ("311.12 m³", "FluidForWellFlushing"), ("105.34MPa", "Pressure"), 
                ("95.78m³", "FluidForBridgePlug"), ("105.34MPa", "Pressure"), ("85.34 m³", "FluidForBall"), ("952.78m³", "FluidForPrepad"), 
                ("90.67m³", "SandForPlug"), ("556.12m³", "FluidForProppant"), ("170.78m³", "Sand"), ("59.12%", "MaxSandRatio"), 
                ("38.56%", "AverageSandRatio"), ("225.78m³", "DisplacementFluid"), ("195Kg", "Fiber"), ("125.34MPa", "MaxPressure"), 
                ("125.34MPa", "FracturePressure"), ("96.34MPa", "PumpStopPressure"), ("20.12m³/min", "MaxFlowRate"), ("300分钟", "JobTime")
            ]
        }
    ),
    (
        "第三十九段施工人员于2019年5月10日20:20到达井场，20:45检查好车辆并召开安全技术交底会，21:00开始施工。第三十九段排空用液170.34 m³，打备压107.78MPa，洗井用液322.45 m³，打备压107.78MPa，泵送桥塞用液100.45m³，打备压107.78MPa，送球用液90.12 m³，泵入前置液964.45m³,段塞加砂94.34m³,携砂液567.78m³,加砂173.12m³,最高砂比60.78%,平均砂比39.34%,顶替液230.45m³,共加纤维200Kg,施工最高压力为127.78MPa,破裂压力127.78MPa,停泵油压98.12MPa，最大排量20.78m³/min，施工用时310分钟。2:30施工结束,压裂车组于2:45整修设备,4:15离开井场。",
        {
            "entities": [
                ("20:20", "ArrivalTime"), ("20:45", "VehicleCheckTime"), ("21:00", "StartTime"),
                ("170.34 m³", "FluidForEmptying"), ("107.78MPa", "Pressure"), ("322.45 m³", "FluidForWellFlushing"), ("107.78MPa", "Pressure"), 
                ("100.45m³", "FluidForBridgePlug"), ("107.78MPa", "Pressure"), ("90.12 m³", "FluidForBall"), ("964.45m³", "FluidForPrepad"), 
                ("94.34m³", "SandForPlug"), ("567.78m³", "FluidForProppant"), ("173.12m³", "Sand"), ("60.78%", "MaxSandRatio"), 
                ("39.34%", "AverageSandRatio"), ("230.45m³", "DisplacementFluid"), ("200Kg", "Fiber"), ("127.78MPa", "MaxPressure"), 
                ("127.78MPa", "FracturePressure"), ("98.12MPa", "PumpStopPressure"), ("20.78m³/min", "MaxFlowRate"), ("310分钟", "JobTime")
            ]
        }
    ),
    (
        "第四十段施工人员于2019年5月15日19:10到达井场，19:35检查好车辆并召开安全技术交底会，19:50开始施工。第四十段排空用液175.12 m³，打备压110.34MPa，洗井用液333.12 m³，打备压110.34MPa，泵送桥塞用液105.78m³，打备压110.34MPa，送球用液95.34 m³，泵入前置液975.78m³,段塞加砂97.67m³,携砂液578.12m³,加砂175.78m³,最高砂比62.34%,平均砂比40.56%,顶替液235.78m³,共加纤维205Kg,施工最高压力为130.34MPa,破裂压力130.34MPa,停泵油压100.34MPa，最大排量21.12m³/min，施工用时320分钟。1:30施工结束,压裂车组于1:45整修设备,3:15离开井场。",
        {
            "entities": [
                ("19:10", "ArrivalTime"), ("19:35", "VehicleCheckTime"), ("19:50", "StartTime"),
                ("175.12 m³", "FluidForEmptying"), ("110.34MPa", "Pressure"), ("333.12 m³", "FluidForWellFlushing"), ("110.34MPa", "Pressure"), 
                ("105.78m³", "FluidForBridgePlug"), ("110.34MPa", "Pressure"), ("95.34 m³", "FluidForBall"), ("975.78m³", "FluidForPrepad"), 
                ("97.67m³", "SandForPlug"), ("578.12m³", "FluidForProppant"), ("175.78m³", "Sand"), ("62.34%", "MaxSandRatio"), 
                ("40.56%", "AverageSandRatio"), ("235.78m³", "DisplacementFluid"), ("205Kg", "Fiber"), ("130.34MPa", "MaxPressure"), 
                ("130.34MPa", "FracturePressure"), ("100.34MPa", "PumpStopPressure"), ("21.12m³/min", "MaxFlowRate"), ("320分钟", "JobTime")
            ]
        }
    ),
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

import time
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers.trainer_callback import TrainerCallback

# 样本数据准备
texts = df["text"].tolist()
entities = df["entities"].tolist()

# 定义tokenizer和model
tokenizer = BertTokenizer.from_pretrained("nghuyong/ernie-3.0-base-zh")
model = BertForTokenClassification.from_pretrained("nghuyong/ernie-3.0-base-zh", num_labels=len(label_map))

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

# 创建DataLoader，并设置shuffle=True以确保数据在每个epoch开始时被打乱
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

training_args = TrainingArguments(output_dir="./baidu", num_train_epochs=100, per_device_train_batch_size=2, logging_dir='./logs', logging_steps=10)

class CustomCallback(TrainerCallback):
    def __init__(self, threshold):
        self.threshold = threshold
        self.losses = []
        self.learning_rates = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.losses.append(logs['loss'])
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])

    def on_step_end(self, args, state, control, **kwargs):
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'loss' in latest_log and latest_log['loss'] <= self.threshold:
                print(f"Stopping early as loss reached {self.threshold}")
                control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        # 绘制loss和learning rate曲线
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(self.losses, 'g-')
        ax2.plot(self.learning_rates, 'b-')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss', color='g')
        ax2.set_ylabel('Learning Rate', color='b')

        plt.title('Loss and Learning Rate over Iterations')
        plt.savefig("Baidu regular loss_learning_rate 0.02.png")
        plt.show()
        

callback = CustomCallback(threshold=0.02)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=lambda data: {"input_ids": torch.stack([f["input_ids"] for f in data]),
                                "attention_mask": torch.stack([f["attention_mask"] for f in data]),
                                "labels": torch.stack([f["labels"] for f in data])},
    callbacks=[callback]
)

start = time.time()
trainer.train()
print(f"Total training time: {time.time() - start} seconds")
print(f"Training time per epoch: {(time.time() - start) / 4} seconds")

# Save the trained model
trainer.save_model(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
