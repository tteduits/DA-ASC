import pandas as pd
from main import EXCEL_FOLDER
import nltk
from nltk.tokenize import sent_tokenize
import random
from fuzzywuzzy import fuzz
import time
import numpy as np
from tqdm import tqdm
from DataFunctions.text_functions import perform_rd, perform_ri, perfrom_rs, perform_sr
nltk.download('punkt')


PERCENTAGE_TO_CHANGE = 25

data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
positive_negative_df = data[data['sentiment'] != 2].reset_index()

NUM_CHUNKS = 20
df_chunks = np.array_split(positive_negative_df, NUM_CHUNKS)
df_chunks = df_chunks[:-4]
calls = 0

for idx_chunk, chunk in enumerate(reversed(df_chunks)):
    reverse_idx = len(df_chunks) - 1 - idx_chunk
    eda_df = pd.DataFrame()
    for idx, row in tqdm(chunk.iterrows(), total=chunk.shape[0]):
        sentences = sent_tokenize(row['clean_text'], language='german')
        filtered_sentences = [sentence for sentence in sentences if len(sentence.split()) >= 3]

        eda_sentences_doc = ''
        random_value = random.randint(1, 4)
        start_time = time.time()

        for sentence in filtered_sentences:
            if random_value == 1:
                sentence_eda = perform_rd(sentence, PERCENTAGE_TO_CHANGE)
            elif random_value == 2:
                sentence_eda = perfrom_rs(sentence, PERCENTAGE_TO_CHANGE)
            elif random_value == 3:
                sentence_eda, calls, start_time = perform_ri(sentence, PERCENTAGE_TO_CHANGE-10, calls, start_time)
            else:
                sentence_eda, calls, start_time = perform_sr(sentence, PERCENTAGE_TO_CHANGE-10, calls, start_time)

            sentence_eda = sentence_eda + '.'
            eda_sentences_doc += sentence_eda + ' '
        ratio = fuzz.ratio(row['clean_text'], eda_sentences_doc)
        eda_df = eda_df._append({'sentiment': row['sentiment'], 'aspect': row['aspect'], 'clean_text': eda_sentences_doc}, ignore_index=True)

    eda_df.to_excel(EXCEL_FOLDER + '\\EDA\\eda_train_data' + str(reverse_idx) + '.xlsx', index=False)
    print(f'xlsx file number {reverse_idx} created')
