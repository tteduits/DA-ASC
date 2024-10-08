import pandas as pd
import nltk
import torch
from nltk.corpus import stopwords
from transformers import XLMRobertaTokenizer
from torch.utils.data import TensorDataset
from nltk.tokenize import sent_tokenize
from datasets import Dataset

nltk.download('stopwords')
german_stop_words = set(stopwords.words('german'))
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
nltk.download('punkt')


def make_cls_sep(dataframe):

    new_rows = []
    for index, row in dataframe.reset_index().iterrows():
        aspect = row['aspect']
        text = row['clean_text']
        sentences = sent_tokenize(text)
        formatted_sentences = ' [SEP] '.join(sentences)  # Combine sentences into a single text
        formatted_text = f"[CLS]{formatted_sentences}"
        new_rows.append({'text': formatted_text, 'sentiment': row['sentiment']})

        df = pd.DataFrame(new_rows)

    return df


def preprocess_data(df):
    sentences = df['text'].tolist()
    labels = df['sentiment'].tolist()

    encoded_data = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,  # Adjust max_length as needed
        return_tensors='pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']

    labels = torch.tensor(labels)

    data_dict = {
    'input_ids': input_ids,
    'attention_mask': attention_masks,
    'labels': labels
    }

    dataset = Dataset.from_dict(data_dict)

    return dataset
