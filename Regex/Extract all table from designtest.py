#现在仅用于提取工艺设计文件中的表格，提取的表格数据将保存到一个Excel文件中。（extracted_table_data）

from docx import Document
import pandas as pd

def read_docx_table(file_path):
    # Read the DOCX file
    document = Document(file_path)
    table_data = []

    # Extract data from tables
    for table in document.tables:
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)

    return table_data

def main():
    file_path = 'designtest.docx'

    # Read the file twice
    table_data_first_read = read_docx_table(file_path)
    table_data_second_read = read_docx_table(file_path)

    # Ensure the data is the same in both reads
    if table_data_first_read == table_data_second_read:
        print("Data is consistent across both reads.")
    else:
        print("Data inconsistency detected between the reads.")

    # Convert to DataFrame for better visualization and manipulation
    table_df = pd.DataFrame(table_data_first_read)

    # Display the DataFrame
    print(table_df)

    # Save the DataFrame to an Excel file
    output_file = 'extracted_table_data.xlsx'
    table_df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
