import pandas as pd

# Load the Excel file
file_path = 'Table from design document.xlsx'
df = pd.read_excel(file_path, header=1)

# Print the column names to diagnose the issue
print("Column names in the DataFrame:", df.columns)

# Strip any leading/trailing whitespace from column names
df.columns = df.columns.str.strip()
print("Cleaned column names in the DataFrame:", df.columns)

# Convert the relevant columns to numeric values, setting errors='coerce' to turn non-numeric values into NaN
df['阶段混砂液'] = pd.to_numeric(df['阶段混砂液'], errors='coerce')
df['阶段砂量'] = pd.to_numeric(df['阶段砂量'], errors='coerce')
df['时间'] = pd.to_numeric(df['时间'], errors='coerce')

# Define the columns of interest and keywords
columns_of_interest = ['排量', '时间', '基液', '阶段砂量', '阶段混砂液']
keywords = ["前置液", "顶替", "投球", "段塞", "高挤携砂液"]

# Initialize dictionaries to store the results
results_by_stage = {}
results_by_keyword = {}
segment_number = 1
chinese_numbers = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '十一', '十二', '十三', '十四', '十五', '十六', '十七', '十八', '十九', '二十']
segment_key = f"第{chinese_numbers[segment_number-1]}段"

# Initialize a DataFrame and totals for the first segment
results_by_stage[segment_key] = pd.DataFrame(columns=columns_of_interest)
totals = {keyword: 0 for keyword in keywords}
sand_totals = {keyword: 0 for keyword in ['段塞', '高挤携砂液']}
results_by_keyword[segment_key] = {'时间': 0, '阶段混砂液': totals.copy(), '阶段砂量': sand_totals.copy()}

# Process the dataframe row by row
for index, row in df.iterrows():
    stage = row['施工']
    mixed_sand_liquid = row['阶段混砂液']
    sand_volume = row['阶段砂量']
    time_value = row['时间']

    # Check if we reached a new segment
    if stage == "水力压裂合计":
        # Increment segment number and reset totals
        segment_number += 1
        if segment_number <= len(chinese_numbers):
            segment_key = f"第{chinese_numbers[segment_number-1]}段"
        else:
            segment_key = f"第{segment_number}段"
        results_by_stage[segment_key] = pd.DataFrame(columns=columns_of_interest)
        results_by_keyword[segment_key] = {'时间': 0, '阶段混砂液': {keyword: 0 for keyword in keywords},
                                           '阶段砂量': {keyword: 0 for keyword in ['段塞', '高挤携砂液']}}
        continue

    # Append the row to the current segment DataFrame
    row_data = row[columns_of_interest].to_frame().T
    results_by_stage[segment_key] = pd.concat([results_by_stage[segment_key], row_data], ignore_index=True)

    # Sum the values for the relevant stages in the current segment
    if not pd.isna(time_value):
        results_by_keyword[segment_key]['时间'] += time_value

    if stage in keywords and not pd.isna(mixed_sand_liquid):
        results_by_keyword[segment_key]['阶段混砂液'][stage] += mixed_sand_liquid

    if stage in ['段塞', '高挤携砂液'] and not pd.isna(sand_volume):
        results_by_keyword[segment_key]['阶段砂量'][stage] += sand_volume

# Save each stage's data into separate Excel files
for segment_number, (segment_key, segment_df) in enumerate(results_by_stage.items(), start=1):
    output_file_path = f'{segment_number}_design.xlsx'
    segment_df.to_excel(output_file_path, index=False)
    print(f'Saved {segment_key} data to {output_file_path}')

# Prepare the data for output
output_data = []

for segment, data in results_by_keyword.items():
    row = [segment, data['时间']]
    for stage in keywords:
        row.append(data['阶段混砂液'].get(stage, 0))
    row.append(data['阶段砂量'].get('段塞', 0))
    row.append(data['阶段砂量'].get('高挤携砂液', 0))
    output_data.append(row)

# Define the column names
columns = ['段数', '时间', '前置液', '顶替', '投球', '段塞', '高挤携砂液', '段塞加砂', '高挤携砂液加砂']

# Create a DataFrame for the output
output_df = pd.DataFrame(output_data, columns=columns)

# Add a row for units
units = ['单位', 'min', 'm3', 'm3', 'm3', 'm3', 'm3', 'm3', 'm3']
output_df.loc[-1] = units  # Adding the units row
output_df.index = output_df.index + 1  # Shifting the index
output_df = output_df.sort_index()  # Sorting the index

# Save the output DataFrame to an Excel file
output_file_path = 'Well Design Overview.xlsx'
output_df.to_excel(output_file_path, index=False)







