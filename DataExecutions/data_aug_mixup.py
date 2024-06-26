import ast

import pandas as pd
from main import EXCEL_FOLDER
from transformers import BertTokenizer, BertModel
from concurrent.futures import ThreadPoolExecutor
from DataFunctions.text_functions import process_row_parallel_mixup
import os


train_data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
positive_negative_df = train_data[train_data['sentiment'] != 2].reset_index()

unique_aspects = positive_negative_df['aspect'].unique()
list_df_per_aspect = [(positive_negative_df[positive_negative_df['aspect'] == aspect]).head(2) for aspect in unique_aspects]

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')


def start_parallel_mixup(token, model, list_df, alpa_value, interpolate):
    with ThreadPoolExecutor(os.cpu_count()) as executor:
        futures = []
        for df_aspect in list_df:
            for idx, row in df_aspect.iterrows():
                future = executor.submit(process_row_parallel_mixup, token, model, df_aspect, row, alpa_value, interpolate)
                futures.append(future)

        result_list = []
        for future in futures:
            result = future.result()
            result_list.append(result)

        mixup_df = pd.DataFrame(result_list, columns=["aspect", "sentiment", "embedding"])
    mixup_df.to_csv(EXCEL_FOLDER + '\\Mixup\\mixup_data' + str(alpa_value) + '.csv', index=False)


alphas_mixup = [0.1, 0.2, 0.3, 0.4]

for alpha in alphas_mixup:
    start_parallel_mixup(tokenizer_bert, model_bert, list_df_per_aspect, alpha, True)

val_data = pd.read_excel(EXCEL_FOLDER + '\\validation_data.xlsx')
test_data = pd.read_excel(EXCEL_FOLDER + '\\test_data.xlsx')

data_dict = {'_train': train_data, '_validation': val_data, '_test': test_data}

for key, value in data_dict.items():
    list_df_per_aspect = [value[value['aspect'] == aspect] for aspect in unique_aspects]
    start_parallel_mixup(tokenizer_bert, model_bert, list_df_per_aspect, key, False)

