import fitz
import difflib
from nltk.translate.bleu_score import sentence_bleu
import torch
import numpy as np
from scipy.stats import beta
import json
import random
import string
import nltk
from nltk.corpus import stopwords
from py_openthesaurus import OpenThesaurusWeb
import time
import spacy
import re

nlp = spacy.load(r'C:\Users\tijst\AppData\Roaming\Python\Python38\site-packages\de_core_news_sm\de_core_news_sm-3.7.0')
open_thesaurus = OpenThesaurusWeb()
nltk.download('punkt')
nltk.download('stopwords')


def clean_full_text(text, title):
    text = text.lower()
    text = text.replace('-\n', ' ')
    text = text.replace('\n', ' ')
    text = text.replace("nur zum internen gebrauch", "")
    text = text.replace("  ", " ")
    text = add_spaces_after_period(text)
    title_found = False

    first_index = text.find(title)
    if first_index != -1:
        second_index = text.find(title, first_index + len(title))
        if second_index != -1:
            text = text[second_index:]
        else:
            text = text[first_index:]

    if title_found:
        return text

    for i in range(len(text) - len(title)):
        similarity = difflib.SequenceMatcher(None, text[i:i + len(title)], title).ratio()
        if similarity >= 0.75:
            text = text[i:]
            break

    return text


def read_pdf(data: bytes) -> str:
    text = ""
    pdf_document = fitz.open(data)
    for page in pdf_document:
        text += page.get_text()

    pdf_document.close()

    return text


def add_spaces_after_period(text):
    """
    Add spaces after periods where needed in the text.
    """
    corrected_text = ""
    for i, char in enumerate(text):
        if char == ".":
            if i + 1 < len(text) and (i == len(text) - 1 or text[i + 1] != " ") and (
                    i + 1 == len(text) or not text[i + 1].isdigit()):
                corrected_text += char + " "
            else:
                corrected_text += char
        else:
            corrected_text += char
    return corrected_text


def process_row_reading(row):
    full_text = read_pdf(row['path'])
    clean_text = clean_full_text(full_text, row['Titel'])
    return clean_text, full_text


def translate_long_text(text, max_length, model, tokenizer):
    segments = make_parts_text(text, max_length)

    translated_segments = []
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_segments.append(translated_text)

    translated_text = " ".join(translated_segments)

    return translated_text.lower()


def make_parts_text(text, max_length):
    segments = []
    while text:
        if len(text) <= max_length:
            segments.append(text)
            break
        else:
            last_space_idx = text.rfind(" ", 0, max_length)  # Find the last space before the max_length
            if last_space_idx == -1:  # If no space found within max_length, just cut
                segments.append(text[:max_length])
                text = text[max_length:]
            else:
                segments.append(text[:last_space_idx])
                text = text[last_space_idx + 1:]

    return segments


def process_row_parallel_bt(row, model, tokenizer, model_bt, tokenizer_bt):
    text = row['clean_text']
    translated_text = translate_long_text(text, 512, model, tokenizer)
    bt_text = translate_long_text(translated_text, 512, model_bt, tokenizer_bt)
    bt_text = re.sub(r'\.{2,}', '.', bt_text)
    bleu_score = sentence_bleu([text.split()], bt_text.split())

    return {
        'aspect': row['aspect'],
        'sentiment': row['sentiment'],
        'clean_text': bt_text,
        'bleu_score': bleu_score
    }


def process_row_parallel_mixup(tokenizer, model, df, row, alpha, interpolate):
    if not interpolate:
        tokens = tokenizer(row['clean_text'], return_tensors="pt", padding='max_length', truncation=True,
                           add_special_tokens=False, max_length=512)
        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state
        interpolated_embeddings_json = embeddings.numpy().tolist()
        return {
            'sentiment': row['sentiment'],
            'aspect': row['aspect'],
            'embedding': json.dumps(interpolated_embeddings_json)
        }

    random_idx1, random_idx2 = np.random.choice(df.index, size=2, replace=False)

    tokens1 = tokenizer(df.loc[random_idx1, 'clean_text'], return_tensors="pt", padding='max_length', truncation=True,
                        add_special_tokens=False, max_length=512)
    tokens2 = tokenizer(df.loc[random_idx2, 'clean_text'], return_tensors="pt", padding='max_length', truncation=True,
                        add_special_tokens=False, max_length=512)
    with torch.no_grad():
        embeddings1 = model(**tokens1).last_hidden_state
        embeddings2 = model(**tokens2).last_hidden_state

    lambda_value = beta.rvs(alpha, alpha, size=1)[0]
    interpolated_embeddings = lambda_value * embeddings1 + (1 - lambda_value) * embeddings2
    interpolated_sentiment = lambda_value * df.loc[random_idx1, 'sentiment'] + (1 - lambda_value) * \
                             df.loc[random_idx2, 'sentiment']

    interpolated_embeddings_json = interpolated_embeddings.numpy().tolist()
    return {
        'sentiment': interpolated_sentiment,
        'aspect': row['aspect'],
        'embedding': json.dumps(interpolated_embeddings_json)
    }


def perform_rd(sentence, percentage):
    words, non_stopwords, num_words_to_change = perform_eda_preparation(sentence, percentage)
    words_to_remove = random.sample(non_stopwords, num_words_to_change)

    result_words = [word for word in words if word not in words_to_remove]
    result_sentence = ' '.join(result_words)

    return result_sentence


def perform_ri(sentence, percentage, calls, start_time):
    words, non_stopwords, num_synonyms_to_find = perform_eda_preparation(sentence, percentage)

    synonyms_to_find = random.sample(non_stopwords, num_synonyms_to_find)
    for original_word in synonyms_to_find:
        synonyms, lemma, calls, start_time = find_synonyms(original_word, calls, start_time)

        if len(synonyms) == 0:
            continue

        synonyms = [item.lower() for item in synonyms]
        if lemma in synonyms:
            synonyms.remove(lemma)

        if len(synonyms) == 0:
            continue

        random_index = random.randint(0, len(words))
        random_synonym = (random.choice(synonyms)).lower()

        words.insert(random_index, random_synonym)

    sentence_changed = ' '.join(words)

    return sentence_changed, calls, start_time


def perfrom_rs(sentence, percentage):
    words, non_stopwords, num_words_to_swap = perform_eda_preparation(sentence, percentage)

    for _ in range(num_words_to_swap):
        index1, index2 = random.sample(range(len(non_stopwords)), 2)
        # Get the indices of non-stopwords in the original sentence
        index1 = words.index(non_stopwords[index1])
        index2 = words.index(non_stopwords[index2])
        words[index1], words[index2] = words[index2], words[index1]

    sentence_changed = ' '.join(words)

    return sentence_changed


def perform_sr(sentence, percentage, calls, start_time):
    words, non_stopwords, num_synonyms_to_find = perform_eda_preparation(sentence, percentage)

    synonyms_to_find = random.sample(non_stopwords, num_synonyms_to_find)
    for original_word in synonyms_to_find:
        synonyms, lemma, calls, start_time = find_synonyms(original_word, calls, start_time)

        if len(synonyms) == 0:
            continue

        synonyms = [item.lower() for item in synonyms]
        if lemma in synonyms:
            synonyms.remove(lemma)

        if len(synonyms) == 0:
            continue

        random_synonym = (random.choice(synonyms)).lower()

        words = [random_synonym if x == original_word else x for x in words]

    sentence_changed = ' '.join(words)

    return sentence_changed, calls, start_time


def perform_eda_preparation(sentence, percentage):
    stop_words = set(stopwords.words('german'))
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))

    words = sentence.split()

    non_stopwords = [word for word in words if word.lower() not in stop_words and word.isalnum()]
    num_words_to_swap = int(len(non_stopwords) * percentage / 100)

    return words, non_stopwords, num_words_to_swap


def find_synonyms(find_word, calls, start_time):
    end_time = time.time()

    if calls == 60 and end_time - start_time < 60:
        print('while started')
        while end_time - start_time < 64:
            end_time = time.time()
        print('we start again')
        start_time = time.time()
        calls = 0

    lemma = nlp(find_word)[0].lemma_

    synonyms = open_thesaurus.get_synonyms(word=lemma, form='long')

    calls += 1
    return synonyms, lemma.lower(), calls, start_time
