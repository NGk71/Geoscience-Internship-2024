import docx
import re
import pandas as pd
import time
from datetime import datetime

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def preprocess_text(text):
    # Modify this if specific preprocessing is required for docx text.
    return text

def search_keywords(text, keywords):
    results = []
    segments = re.split(r'(第[\d一二三四五六七八九十]+段)', text)
    segment_name = ""

    for i in range(1, len(segments), 2):
        segment_name = segments[i].strip()
        piece = segments[i+1]
        found_keywords = {"段数": segment_name}
        for keyword in keywords:
            pattern = rf"{keyword}\s*[^\d]*\s*(\d+\.?\d*)"
            matches = re.findall(pattern, piece, re.DOTALL)
            if matches:
                for idx, match in enumerate(matches):
                    col_name = f"{keyword}{idx+1}" if idx > 0 else keyword
                    found_keywords[col_name] = match.strip()
            else:
                found_keywords[keyword] = None
        results.append(found_keywords)
    return results

def save_results_to_excel(results, base_filename):
    units = {
        "试压": "MPa", "打备压": "MPa", "试挤用液": "m³", "洗井用液": "m³", "泵送桥塞用液": "m³", "送球用液": "m³",
        "排空用液": "m³", "预处理酸": "m³", "顶替液": "m³", "停泵反应": "min", "后打备压": "MPa",
        "泵入前置液": "m³", "段塞加砂": "m³", "携砂液": "m³", "加砂": "m³", "最高砂比": "%", "平均砂比": "%",
        "共加纤维": "Kg", "暂堵颗粒": "Kg", "施工最高压力": "MPa", "破裂压力": "MPa", "停泵油压": "MPa",
        "最大排量": "m³/min", "施工用时": "min"
    }
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{base_filename}_{current_time}.xlsx"
    df = pd.DataFrame(results)

    # Insert a row for units
    unit_row = {col: units.get(col.split('1')[0], '') for col in df.columns}
    df = pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)

    # Drop rows where all elements except '段数' are None
    df = df.dropna(subset=[col for col in df.columns if col != '段数'], how='all')

    # Define the custom column order and sort additional columns dynamically
    main_columns = ["段数", "施工用时", "试压", "打备压", "试挤用液", "洗井用液", "泵送桥塞用液", "送球用液",
                    "排空用液", "预处理酸", "顶替液", "停泵反应", "后打备压", "泵入前置液", "段塞加砂",
                    "携砂液", "加砂", "最高砂比", "平均砂比", "共加纤维", "暂堵颗粒", "施工最高压力", "破裂压力",
                    "停泵油压", "最大排量"]
    organized_cols = [c for col in main_columns for c in df.columns if c.startswith(col)]
    df = df[organized_cols]
    df.to_excel(filename, index=False)

start_time = time.time()
docx_path = 'testforall.docx'
keywords = ["试压", "打备压", "试挤用液", "洗井用液", "泵送桥塞用液", "送球用液", "排空用液", "预处理酸",
            "顶替液", "停泵反应", "后打备压", "泵入前置液", "段塞加砂", "携砂液", "加砂", "最高砂比", 
            "平均砂比", "共加纤维", "暂堵颗粒", "施工最高压力", "破裂压力", "停泵油压", "最大排量", "施工用时"]

text = extract_text_from_docx(docx_path)
text = preprocess_text(text)
results = search_keywords(text, keywords)
save_results_to_excel(results, 'Extracted_Data')

print('Time:', time.time() - start_time)


