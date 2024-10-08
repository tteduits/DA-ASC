import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

model_name_ge_en = 'Helsinki-NLP/opus-mt-de-en'
model_name_en_ge = 'Helsinki-NLP/opus-mt-en-de'

tokenizer_ge_en = MarianTokenizer.from_pretrained(model_name_ge_en)
model_ge_en = MarianMTModel.from_pretrained(model_name_ge_en)

tokenizer_en_ge = MarianTokenizer.from_pretrained(model_name_en_ge)
model_en_ge = MarianMTModel.from_pretrained(model_name_en_ge)

stop_words = set(stopwords.words('english'))
PERCENTAGE_TO_CHANGE = 0.10


def bt_translate_text(df, beam1, beam2):
    bt_data = []
    for idx, row in df.iterrows():
        inputs = tokenizer_en_ge(row['Review_Text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model_en_ge.generate(**inputs, num_beams=beam1, num_return_sequences=1, do_sample=True)
        translated_text = tokenizer_en_ge.decode(translated[0], skip_special_tokens=True)

        inputs_bt = tokenizer_ge_en(translated_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated_bt = model_ge_en.generate(**inputs_bt, num_beams=beam2, num_return_sequences=1, do_sample=True)
        translated_text_bt = tokenizer_ge_en.decode(translated_bt[0], skip_special_tokens=True)

        bleu_score = sentence_bleu([translated_text_bt.lower()], row['Review_Text'].lower())
        bt_data.append({
            'Review_ID': row['Review_ID'],
            'Review_Text': translated_text_bt,  # Remove trailing whitespace
            'Categories': row['Categories'],
            'Polarities': row['Polarities'],
            'bleu_score': bleu_score
        })

    df_bt_reviews = pd.DataFrame(bt_data)
    print(f"The average of column 'A' is: {df_bt_reviews['bleu_score'].mean():.2f}")
    return df_bt_reviews


def ka_perform(df):
    aug = nac.KeyboardAug(lang='en', aug_char_max=1, stopwords=stopwords.words('english'), aug_word_p=0.1,
                          include_upper_case=False, include_numeric=True,
                          include_special_char=False)

    ka_data = []
    for idx, row in df.iterrows():
        text_ka = aug.augment(row['Review_Text'])[0]
        ka_data.append({
            'Review_ID': row['Review_ID'],
            'Review_Text': text_ka,
            'Categories': row['Categories'],
            'Polarities': row['Polarities']
        })

    df_ka_reviews = pd.DataFrame(ka_data)

    return df_ka_reviews


def eda_perform(df):
    eda_data = []
    for idx, row in df.iterrows():
        eda_text = ''
        for sentence in sent_tokenize(row['Review_Text']):
            random_value = random.randint(0, 3)
            if random_value == 0:
                aug = naw.SynonymAug(aug_p=PERCENTAGE_TO_CHANGE, stopwords=stop_words)
                new_sentence_text = aug.augment(sentence)

            elif random_value == 1:
                aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert',
                                                aug_p=PERCENTAGE_TO_CHANGE, stopwords=stop_words)
                new_sentence_text = aug.augment(sentence)

            elif random_value == 2:
                aug = naw.RandomWordAug(action='swap', aug_p=PERCENTAGE_TO_CHANGE, stopwords=stop_words)
                new_sentence_text = aug.augment(sentence)

            elif random_value == 3:
                aug = naw.RandomWordAug(action='delete', aug_p=PERCENTAGE_TO_CHANGE, stopwords=stop_words)
                new_sentence_text = aug.augment(sentence)

            eda_text = eda_text + new_sentence_text[0]

        eda_data.append({
            'Review_ID': row['Review_ID'],
            'Review_Text': eda_text,
            'Categories': row['Categories'],
            'Polarities': row['Polarities']
        })

    df_eda_reviews = pd.DataFrame(eda_data)

    return df_eda_reviews
