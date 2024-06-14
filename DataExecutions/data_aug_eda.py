import pandas as pd
from main import EXCEL_FOLDER
import nltk
from nltk.tokenize import sent_tokenize
import random
from fuzzywuzzy import fuzz
import time
from DataFunctions.text_functions import perform_rd, perform_ri, perfrom_rs, perform_sr
nltk.download('punkt')


PERCENTAGE_TO_CHANGE = 20

data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
positive_negative_df = data[data['sentiment'] != 1].reset_index()
calls = 0

eda_df = pd.DataFrame()
for idx, row in positive_negative_df.iterrows():
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
            sentence_eda, calls, start_time = perform_ri(sentence, PERCENTAGE_TO_CHANGE, calls, start_time)
        else:
            sentence_eda, calls, start_time = perform_sr(sentence, PERCENTAGE_TO_CHANGE, calls, start_time)

        sentence_eda = sentence_eda + '.'
        eda_sentences_doc += sentence_eda + ' '
    ratio = fuzz.ratio(row['clean_text'], eda_sentences_doc)
    eda_df = eda_df._append({'sentiment': row['sentiment'], 'aspect': row['aspect'], 'clean_text': eda_sentences_doc}, ignore_index=True)
    print(f"we are at file : {idx+1}")

eda_df.to_excel(EXCEL_FOLDER + '\\eda_train_data.xlsx')
