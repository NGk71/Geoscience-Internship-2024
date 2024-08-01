import docx
import pandas as pd

# Mapping from digits to Chinese numerals
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
    13: "十三"
}

def extract_tables_from_docx(docx_path):
    doc = docx.Document(docx_path)
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)
    return tables

def find_segment_tables(tables, segment_number):
    operations_data = []
    liquid_data = []
    current_segment_keyword = f"第{digit_to_chinese[segment_number]}段"
    next_segment_keywords = [f"第{digit_to_chinese[i]}段" for i in range(segment_number + 1, 14)]
    
    capturing_operations = False
    capturing_liquid = False

    for table in tables:
        temp_operations = []
        temp_liquid = []

        # Determine the type of data in the table for capturing
        for row in table:
            joined_row = ''.join(row)  # Joining row cells to make searching more inclusive
            
            # Start capturing operations data
            if current_segment_keyword in joined_row:
                capturing_operations = True
            
            # Check if this row indicates the start of the next segment to stop capturing
            if any(seg in joined_row for seg in next_segment_keywords):
                capturing_operations = False
                capturing_liquid = False
                continue
            
            # Capture operations data if the flag is set
            if capturing_operations:
                temp_operations.append(row)
            
            # Check if the table is for liquid data and start capturing
            if "类 型" in joined_row and f"{current_segment_keyword}\n施工用液" in joined_row:
                capturing_liquid = True
                capturing_operations = False  # Ensure no overlap in data capture
            
            # Capture liquid data if the flag is set
            if capturing_liquid:
                temp_liquid.append(row)
        
        # Assign the captured data to the appropriate variable if data was found
        if temp_operations:
            operations_data = temp_operations
        if temp_liquid:
            liquid_data = temp_liquid

    return operations_data, liquid_data





def save_segment_to_excel(tables, segment_number, filename):
    operations_data, liquid_data = find_segment_tables(tables, segment_number)

    if not operations_data and not liquid_data:
        print(f"No data found for 第{digit_to_chinese[segment_number]}段. Skipping file creation.")
        return

    with pd.ExcelWriter(filename) as writer:
        if operations_data:
            operations_df = pd.DataFrame(operations_data[1:], columns=operations_data[0])
            operations_df.to_excel(writer, sheet_name=f'第{digit_to_chinese[segment_number]}段施工操作', index=False)
        
        if liquid_data:
            liquid_df = pd.DataFrame(liquid_data[1:], columns=liquid_data[0])
            liquid_df.to_excel(writer, sheet_name=f'第{digit_to_chinese[segment_number]}段施工用液', index=False)

# Main script
docx_path = 'testforall.docx'
for i in range(1, 14):
    output_filename = f'stage_{i}_extracted.xlsx'
    
    # Extract tables from the document
    tables = extract_tables_from_docx(docx_path)
    
    # Save the extracted data to a specific Excel file for each stage
    save_segment_to_excel(tables, i, output_filename)



