import pandas as pd

# Save results to an Excel file with customizable keyword names
def save_results_to_excel(results, well_name, base_filename, keyword_mapping):
    # Units associated with the keyword names
    units = {
        "施工用时": "min",
        "试压": "MPa",
        "打备压": "MPa",
        "试挤用液": "m³",
        "洗井用液": "m³",
        "泵送桥塞用液": "m³",
        "送球用液": "m³",
        "排空用液": "m³",
        "预处理酸": "m³",
        "顶替液": "m³",
        "停泵反应": "min",
        "后打备压": "MPa",
        "泵入前置液": "m³",
        "段塞加砂": "m³",
        "携砂液": "m³",
        "加砂": "m³",
        "最高砂比": "%",
        "平均砂比": "%",
        "共加纤维": "Kg",
        "暂堵颗粒": "Kg",
        "施工最高压力": "MPa",
        "破裂压力": "MPa",
        "停泵油压": "MPa",
        "最大排量": "m³/min",
        "West": "m",
        "East": "m",
        "North": "m",
        "South": "m",
        "表面管处理压力": "MPa",
        "表面管压力": "MPa",
        "表面管初始激发压力": "MPa",
        "最大回流速率": "m³/min",
        "预搅拌浆体积": "m³",
        "宽度": "m",
        "高度": "m",
        "压裂方向": "",
        "vol 1": "m³",
        "vol 2": "m³",
        "vol 3": "m³",
        "vol 4": "m³",
        "vol 5": "m³",
        "vol 6": "m³",
        "vol 7": "m³",
        "vol 8": "m³",
        "vol 9": "m³",
        "阶段顶部测深": "ft",
        "阶段底部测深": "ft",
        "真垂直深度": "ft",
        "阶间分钟": "min",
        "地层": "",
        "流体系统": "",
        "压裂系统": "",
        "平均处理压力": "psi",
        "最大处理压力": "psi",
        "平均底压": "psi",
        "最大底压": "psi",
        "压裂梯度": "psi/ft",
        "浆体积": "bbl",
        "注入体积": "bbl",
        "冲洗体积": "bbl"
    }

    filename = f"{base_filename}.xlsx"
    df = pd.DataFrame(results)
    
    # Generate the unit row dynamically based on the keyword_mapping names
    unit_row = {}
    for col in df.columns:
        base_name = col.split('1')[0]  # Extract base name before numeric suffix
        # Find the corresponding key in keyword_mapping
        unit = units.get(base_name, '')
        unit_row[col] = unit

    df = pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)
    df = df.dropna(subset=[col for col in df.columns if col != '段数'], how='all')

    df.insert(0, "井名", well_name)

    main_columns = ["井名", "段数"] + list(keyword_mapping.keys())

    organized_cols = []
    for col in main_columns:
        related_cols = [c for c in df.columns if c.startswith(col)]
        organized_cols.extend(related_cols)

    df = df[organized_cols]
    df.to_excel(filename, index=False)





    


