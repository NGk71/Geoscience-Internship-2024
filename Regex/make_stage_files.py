import re
import pandas as pd
from datetime import datetime

digit_to_chinese = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
    11: "十一",
    12: "十二",
    13: "十三",
    14: "十四",
    15: "十五",
    16: "十六",
    17: "十七",
    18: "十八",
    19: "十九",
    20: "二十",
    21: "二十一",
    22: "二十二",
    23: "二十三",
    24: "二十四",
    25: "二十五",
}

# Find tables for each segment
def find_segment_tables(tables):
    segment_tables = {}
    pattern = re.compile(r'第(\d+|[一二三四五六七八九十十一十二十三十四十五十六十七十八十九二十]+)段')

    current_segment = None
    for table in tables:
        for row in table:
            row_text = ''.join(row)
            match = pattern.search(row_text)
            if match:
                segment_str = match.group(1)
                if segment_str.isdigit():
                    segment_num = int(segment_str)
                else:
                    segment_num = next(key for key, value in digit_to_chinese.items() if value == segment_str)
                current_segment = segment_num
                if current_segment not in segment_tables:
                    segment_tables[current_segment] = []
            if current_segment:
                segment_tables[current_segment].append(row)
    
    return segment_tables

# Process the values to handle ranges by taking the average
def process_values(value):
    try:
        if '-' in str(value):
            start_val, end_val = map(float, value.split('-'))
            return (start_val + end_val) / 2
    except ValueError:
        pass
    return value

# Add new column based on conditions
def only_table_needed(operations_df):
    if operations_df.empty:
        return operations_df
    
    # Identify the header row where "工序" appears in column A
    header_row_indices = operations_df[operations_df.iloc[:, 0].str.contains("工    序", na=False)].index
    
    if len(header_row_indices) == 0:
        return operations_df
    
    header_row_index = header_row_indices[0]
    
    # Use this row as the header row for the dataframe
    operations_df.columns = operations_df.iloc[header_row_index]
    operations_df = operations_df.drop(index=header_row_index).reset_index(drop=True)

    # Remove "/" in all cells and evrything after it, and add on more column named "NO USE"
    #主要为了处理特定情况下的数据
    operations_df = operations_df.applymap(lambda x: str(x).split("/")[0])
    operations_df["NO USE"] = ""

    if "备注" in operations_df.columns:
        operations_df.drop(columns=["备注"], inplace=True)

    return operations_df

# Save each segment's data to an Excel file
def save_segment_to_excel(segment_tables, segment_number, base_filename):
    if segment_number not in segment_tables:
        print(f"No operations data found for 第{digit_to_chinese[segment_number]}段. Skipping file creation.")
        return

    operations_data = segment_tables[segment_number]
    max_columns = max(len(row) for row in operations_data)
    formatted_data = [row + [""] * (max_columns - len(row)) for row in operations_data]

    # Apply processing to the data to handle ranges, except for "时间" column
    header = formatted_data[0]
    for i, row in enumerate(formatted_data[1:], start=1):
        for j, cell in enumerate(row):
            if header[j] != "时间":
                formatted_data[i][j] = process_values(cell)

    # Create a DataFrame to fill empty values
    operations_df = pd.DataFrame(formatted_data[1:], columns=formatted_data[0])

    # Add the new column based on specified conditions
    updated_df = only_table_needed(operations_df)

    # Fill empty values using forward fill method to connect lines
    updated_df.ffill(inplace=True)
    updated_df.bfill(inplace=True)

    filename = f"{base_filename}_process.xlsx"
    with pd.ExcelWriter(filename) as writer:
        updated_df.to_excel(writer, sheet_name=f'第{digit_to_chinese[segment_number]}段施工操作', index=False)
















