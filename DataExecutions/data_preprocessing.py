import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from DataFunctions.text_functions import process_row_reading
from main import EXCEL_FOLDER


files = os.listdir(EXCEL_FOLDER+"\\CombinedData")
combined_data = pd.DataFrame()
for file in files:
    df = pd.read_excel(EXCEL_FOLDER+"\\CombinedData\\" + file)
    combined_data = pd.concat([combined_data, df], ignore_index=True)

pdf_files = list(Path("D:\\MasterThesisTijs\\PdfFiles").rglob('*.pdf'))
id_to_path = {Path(pdf_file).stem: pdf_file for pdf_file in pdf_files}
combined_data['path'] = combined_data['ID'].astype(str).map(id_to_path)

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = []
    for index, row in combined_data.iterrows():
        future = executor.submit(process_row_reading, row)
        futures.append((index, future))

    # Iterate over futures and update DataFrame
    for index, future in tqdm(futures, total=len(combined_data), desc='Reading pdf files', unit='row'):
        clean_text, full_text = future.result()
        combined_data.loc[index, 'clean_text'] = clean_text
        combined_data.loc[index, 'full_text'] = full_text


combined_data.to_excel(EXCEL_FOLDER + "\\combined_data.xlsx", index=False)
