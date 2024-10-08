from torch import nn
from transformers import LongformerForSequenceClassification, LongformerTokenizer, XLMRobertaTokenizer, XLMRobertaForSequenceClassification, XLMRobertaModel, XLMRobertaConfig, AutoTokenizer, AdamW, EarlyStoppingCallback, BertModel, get_linear_schedule_with_warmup, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from datasets import Dataset
import torch
from main import EXCEL_FOLDER

TRAINING_MODE = '-'
MAX_POSITION_EMBEDDING = 1048
MODEL_NAME = 'xlm-roberta-base'

train_data = pd.read_excel(EXCEL_FOLDER + '/train_data.xlsx')
validation_data = pd.read_excel(EXCEL_FOLDER + '/validation_data.xlsx')

unique_values_counts = train_data['sentiment'].value_counts()

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

nltk.download('stopwords')
german_stop_words = set(stopwords.words('german'))
nltk.download('punkt')


def make_mapping(df):
    category_map = {
    0: 'negative',

    1: 'snegative',
    2: 'neutral',
    3: 'spositive',
    4: 'positive',
    }

    df['sentiment'] = df['sentiment'].map(category_map)

    category_map_new = {
    'negative':0,
    'snegative':0,
    'neutral':1,
    'spositive':2,
    'positive':2,
    }

    df['sentiment'] = df['sentiment'].map(category_map_new)

    return df


def sample_group(group, number):
    n = min(number, len(group))
    return group.sample(n=n, replace=True)


def make_cls_sep(dataframe):
    german_stopwords = set(stopwords.words('german'))

    new_rows = []
    for index, row in dataframe.reset_index().iterrows():
        aspect = row['aspect']
        text = row['clean_text']

        sentences = sent_tokenize(text)
        formatted_sentences = '[SEP]'.join(sentences)
        formatted_text = f"[CLS]{aspect}[SEP]{formatted_sentences}"
        new_rows.append({'text': formatted_text, 'sentiment': row['sentiment']})

        df = pd.DataFrame(new_rows)

    return df


def calculate_document_embeddings(input_ids_list, chunk_size=512):

    mean_parts_tensors_list = []
    print('start')
    for input_ids in input_ids_list:
        chunk_parts = [input_ids[i:i+chunk_size] for i in range(0, len(input_ids), chunk_size)]
        parts_tensors = [torch.tensor(part, dtype=torch.float32) for part in chunk_parts]
        parts_tensor = torch.stack(parts_tensors)
        mean_part_tensor = torch.mean(parts_tensor, dim=0)
        mean_parts_tensors_list.append(mean_part_tensor.long())

    document_embeddings = torch.stack(mean_parts_tensors_list)
    print('done')
    return document_embeddings


def preprocess_data(df, max_position_embedding):
    sentences = df['text'].tolist()
    labels = df['sentiment'].tolist()

    # Tokenize the sentences
    encoded_data = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_position_embedding,
        return_tensors='pt'
    )
    lengths = [len(tokenizer.tokenize(sentence)) for sentence in sentences]

    # Count how many sentences are longer than the max length
    num_long_sentences = sum(length > 1048 for length in lengths)

    print(f"Number of sentences longer than {1048}: {num_long_sentences}")
    document_embeddings = calculate_document_embeddings(encoded_data['input_ids'])

    data_dict = {
        'labels': torch.tensor(labels),
        'input_ids': document_embeddings,
    }

    dataset = Dataset.from_dict(data_dict)

    return dataset


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    accuracy = accuracy_score(labels, preds)
    precision_micro = precision_score(labels, preds, average='micro')
    recall_micro = recall_score(labels, preds, average='micro')
    f1_micro = f1_score(labels, preds, average='micro')

    precision_macro = precision_score(labels, preds, average='macro')
    recall_macro = recall_score(labels, preds, average='macro')
    f1_macro = f1_score(labels, preds, average='macro')

    conf_mat = confusion_matrix(labels, preds)

    return {
        'accuracy': accuracy,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'confusion_matrix': conf_mat.tolist()
    }


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


train_data = make_mapping(train_data)
validation_data = make_mapping(validation_data)
#
# train_data = train_data.groupby('sentiment', group_keys=False).apply(lambda x: sample_group(x, 10))
# validation_data = validation_data.groupby('sentiment', group_keys=False).apply(lambda x: sample_group(x, 1))
unique_values_counts = train_data['sentiment'].value_counts()

df_sampled = train_data[train_data['sentiment'] == 1].sample(frac=0.25, random_state=42)
train_data = pd.concat([df_sampled, train_data[train_data['sentiment'] != 1]])

unique_values_counts = train_data['sentiment'].value_counts()


df_train = make_cls_sep(train_data)
df_validation = make_cls_sep(validation_data)

train_dataset = preprocess_data(df_train, MAX_POSITION_EMBEDDING)
val_dataset = preprocess_data(df_validation, MAX_POSITION_EMBEDDING)

class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_data['sentiment']), y=train_data['sentiment']), dtype=torch.float32)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class_weights = class_weights.to(device)

learning_rates = [5e-5, 4e-5, 3e-5, 2e-5]
for lr in reversed(learning_rates):

    model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(np.unique(train_data['sentiment'])))

    model.to(device)
    training_args = TrainingArguments(
        per_device_train_batch_size=16,
        num_train_epochs=5,
        logging_dir='./logs_' + str(lr),
        logging_steps=100,
        save_steps=1000,
        evaluation_strategy="epoch",
        output_dir='./results_' + str(lr),
        learning_rate=lr,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        save_total_limit=1,

    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],

    )

    trainer.train()





