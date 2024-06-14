import pandas as pd
from main import EXCEL_FOLDER
from transformers import MarianMTModel, MarianTokenizer
from DataFunctions.text_functions import process_row_parallel_bt, translate_long_text
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import numpy as np
from tqdm import tqdm


data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
positive_negative_df = data[data['sentiment'] != 1]

model_name_ge_en = 'Helsinki-NLP/opus-mt-de-en'
model_name_en_ge = 'Helsinki-NLP/opus-mt-en-de'

tokenizer_ge_en = MarianTokenizer.from_pretrained(model_name_ge_en)
model_ge_en = MarianMTModel.from_pretrained(model_name_ge_en)

tokenizer_en_ge = MarianTokenizer.from_pretrained(model_name_en_ge)
model_en_ge = MarianMTModel.from_pretrained(model_name_en_ge)

model_name_ge_cz = 'Helsinki-NLP/opus-mt-de-cs'
model_name_cz_ge = 'Helsinki-NLP/opus-mt-cs-de'

tokenizer_ge_cz = MarianTokenizer.from_pretrained(model_name_ge_cz)
model_ge_cz = MarianMTModel.from_pretrained(model_name_ge_cz)

tokenizer_cz_ge = MarianTokenizer.from_pretrained(model_name_cz_ge)
model_cz_ge = MarianMTModel.from_pretrained(model_name_cz_ge)


def start_parallelization(df, model, tokenizer, model_bt, tokenizer_bt, excel_file):
    results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_row = {
            executor.submit(process_row_parallel_bt, row, model, tokenizer, model_bt,
                            tokenizer_bt): (index, row)
            for index, row in df.iterrows()
        }
        for future in tqdm(as_completed(future_to_row), total=len(future_to_row), desc="Processing"):
            result = future.result()
            results.append(result)

    result_df = pd.DataFrame(results)
    result_df.to_excel(EXCEL_FOLDER + excel_file)


NUM_CHUNKS = 5
df_chunks = np.array_split(positive_negative_df, NUM_CHUNKS)

for i, chunk in enumerate(df_chunks):
    start_parallelization(chunk, model_ge_cz, tokenizer_ge_cz, model_cz_ge, tokenizer_cz_ge,
                          '\\BT\\train_data_cz_ge_' + str(i) + '.xlsx')
    print(f'Dataframe Czech number {i} created')

for i, chunk in enumerate(df_chunks):
    start_parallelization(chunk, model_ge_en, tokenizer_ge_en, model_en_ge, tokenizer_en_ge,
                          '\\BT\\train_data_en_ge_' + str(i) + '.xlsx')
    print(f'Dataframe English number {i} created')

a = 1
