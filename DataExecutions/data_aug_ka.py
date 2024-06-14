from main import EXCEL_FOLDER
import pandas as pd
import nlpaug.augmenter.char as nac
import nltk
import concurrent.futures
import os
nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
positive_negative_df = data[data['sentiment'] != 1].head(10)


german_stopwords = stopwords.words('german')
aug = nac.KeyboardAug(lang='de', aug_char_max=1, aug_word_p=0.1, stopwords=german_stopwords, include_upper_case=False, include_numeric=True, include_special_char=False)

ka_list = []


def process_row(row):
    text = row['clean_text']
    text_ka = aug.augment(text)[0]
    information_dict = {'sentiment': row['sentiment'], 'aspect': row['aspect'], 'clean_text': text_ka}
    return information_dict


with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
    future_to_row = {executor.submit(process_row, row): row for _, row in positive_negative_df.iterrows()}

    for future in concurrent.futures.as_completed(future_to_row):
        ka_list.append(future.result())

ka_df = pd.DataFrame(ka_list)

ka_df.to_excel(EXCEL_FOLDER + '\\ka_train_data.xlsx')
a = 1
