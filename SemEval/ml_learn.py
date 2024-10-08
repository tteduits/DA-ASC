import pandas as pd
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, BertTokenizer
from main import EXCEL_FOLDER
from ML.ml_functions import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import torch
from torch import nn
import numpy as np
import random

random.seed(42)
MAX_POSITION_EMBEDDING = 512
MODEL_NAME = 'google-bert/bert-base-uncased'
MIX_UP = True

# neutral_sentiment_data = train_data[train_data['sentiment'] == 1]
# duplicated_data = pd.concat([neutral_sentiment_data] * 2, ignore_index=True)
# train_data = pd.concat([train_data, duplicated_data], ignore_index=True)

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def sample_group(group, number):
    n = min(number, len(group))
    return group.sample(n=n, replace=True)


def make_cls_sep(dataframe):
    new_rows = []
    for index, row in dataframe.reset_index().iterrows():
        aspect = row['aspect']
        text = row['Review_Text']
        sentences = sent_tokenize(text)
        formatted_sentences = '[SEP]'.join(sentences)
        formatted_text = f"[CLS]{aspect}[SEP]{formatted_sentences}"
        new_rows.append({'text': formatted_text, 'sentiment': row['sentiment']})

        df = pd.DataFrame(new_rows)

    return df


def preprocess_data(df, max_position_embedding):
    sentences = df['text'].tolist()
    labels = df['sentiment'].tolist()

    encoded_data = tokenizer(
        sentences,
        padding='max_length',
        truncation=True,
        max_length=max_position_embedding,
        return_tensors='pt'
    )

    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']

    data_dict = {
        'labels': torch.tensor(labels),
        'input_ids': input_ids,
        'attention_mask': attention_masks,

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
        labels = inputs.get('labels')
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def map_label(label_value):
    if label_value < 0.67:
        return 0
    elif 0.67 <= label_value < 1.33:
        return 1
    else:
        return 2


column_labels = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'f1_macro']
models = ['mixup']

alpha_values = [0.1, 0.2, 0.3, 0.4]

df_result = pd.DataFrame(index=alpha_values, columns=column_labels)
ratio = 1
for model_type in reversed(models):

    train_data = pd.read_csv(EXCEL_FOLDER + '\\SemEval_train' + model_type + '.csv')
    test_data = pd.read_csv(EXCEL_FOLDER + '\\SemEval_test.csv')

    df_test = make_cls_sep(test_data)
    test_dataset = preprocess_data(df_test, MAX_POSITION_EMBEDDING)

    if model_type == 'mixup':

        for alpha in alpha_values:
            train_dataset = pd.DataFrame()
            for value in train_data['aspect'].unique():
                df_per_aspect = train_data[train_data['aspect'] == value]

                df_train_aspect = make_cls_sep(df_per_aspect)
                train_dataset_aspect = preprocess_data(df_train_aspect, MAX_POSITION_EMBEDDING)

                df_encoded = pd.DataFrame(
                    {'labels': train_dataset_aspect['labels'], 'input_ids': train_dataset_aspect['input_ids'], 'attention_mask': train_dataset_aspect['attention_mask']})
                mixup_df = pd.DataFrame(columns=['labels', 'input_ids', 'attention_mask'])
                for _ in range(len(df_per_aspect) * ratio):
                    lamda = np.random.beta(alpha, alpha)
                    random_rows = df_encoded.sample(n=2, replace=False).reset_index(drop=True)

                    label_mixup = lamda * random_rows['labels'].iloc[0] + (1 - lamda) * random_rows['labels'].iloc[1]
                    input_ids = torch.tensor((lamda * np.array(random_rows['input_ids'].iloc[0]) +
                                              (1 - lamda) * np.array(random_rows['input_ids'].iloc[1])).tolist(),
                                             dtype=torch.long)

                    attention_mask = (input_ids != 0).long()

                    mixup_row = {
                        'labels': label_mixup,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }

                    mixup_df = mixup_df._append(mixup_row, ignore_index=True)

                df_combined = pd.concat([mixup_df, df_encoded], axis=0, ignore_index=True)
                train_dataset = pd.concat(([df_combined, train_dataset]))

            train_dataset['labels'] = train_dataset['labels'].apply(map_label)
            train_dataset = train_dataset.to_dict(orient='list')
            train_dataset = Dataset.from_dict(train_dataset)

            lr = 3e-6
            weight = 0.01
            best_accuracy = 0.0
            best_lr = None
            best_weight = None

            # model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
            # training_args = TrainingArguments(
            #     per_device_train_batch_size=10,
            #     num_train_epochs=1,  # Train for 3 epochs
            #     logging_dir='./logs_' + str(lr),
            #     logging_steps=100,
            #     save_steps=1000,
            #     evaluation_strategy="no",  # Disable evaluation
            #     output_dir='./results_' + str(lr),
            #     learning_rate=lr,
            #     weight_decay=weight,
            #     load_best_model_at_end=False,  # No evaluation means no best model saving
            #     save_strategy="epoch",
            #     save_total_limit=1,
            # )
            #
            # trainer = CustomTrainer(
            #     model=model,
            #     args=training_args,
            #     train_dataset=train_dataset,
            #     eval_dataset=test_dataset,
            #     compute_metrics=compute_metrics,
            #     # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            #
            # )
            #
            # trainer.train()
            #
            # final_model_dir = EXCEL_FOLDER + '/final_model_' + str(model_type)
            # trainer.save_model(final_model_dir)
            # eval_results = trainer.evaluate(eval_dataset=test_dataset)

            df_result.at[alpha, 'accuracy'] = 1
            df_result.at[alpha, 'precision_micro'] = 1
            df_result.at[alpha, 'recall_micro'] = 1
            df_result.at[alpha, 'f1_micro'] = 1
            df_result.at[alpha, 'f1_macro'] = 1

            print(f'ratio {ratio}')
            print(f'alpha {alpha}')
            print(df_result)

    else:
        df_train = make_cls_sep(train_data)
        train_dataset = preprocess_data(df_train, MAX_POSITION_EMBEDDING)

        # Best Learning Rate: 3e-06 | Best weight: 0.01 | Best Validation Accuracy: 0.8608058608058609
        lr = 3e-6
        weight = 0.01
        best_accuracy = 0.0
        best_lr = None
        best_weight = None

        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
        training_args = TrainingArguments(
            per_device_train_batch_size=10,
            num_train_epochs=1,  # Train for 3 epochs
            logging_dir='./logs_' + str(lr),
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="no",  # Disable evaluation
            output_dir='./results_' + str(lr),
            learning_rate=lr,
            weight_decay=weight,
            load_best_model_at_end=False,  # No evaluation means no best model saving
            save_strategy="epoch",
            save_total_limit=1,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

        )

        trainer.train()

        final_model_dir = EXCEL_FOLDER + '/final_model_' + str(model_type)
        trainer.save_model(final_model_dir)
        eval_results = trainer.evaluate(eval_dataset=test_dataset)

        df_result.at[alpha, 'accuracy'] = eval_results['eval_accuracy']
        df_result.at[alpha, 'precision_micro'] = eval_results['eval_precision_micro']
        df_result.at[alpha, 'recall_micro'] = eval_results['eval_recall_micro']
        df_result.at[alpha, 'f1_micro'] = eval_results['eval_f1_micro']
        df_result.at[alpha, 'f1_macro'] = eval_results['eval_f1_macro']

