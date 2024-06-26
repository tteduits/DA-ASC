from main import EXCEL_FOLDER
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from nltk.tokenize import sent_tokenize
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


test_data = pd.read_excel(EXCEL_FOLDER + '\\test_data.xlsx')
test_data = test_data[(test_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation') | (test_data['aspect'] == 'Zentralabteilung')]

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')

new_rows = []
for index, row in test_data.reset_index().iterrows():
    aspect = row['aspect']
    text = row['clean_text']
    sentences = sent_tokenize(text)
    formatted_sentences = ' [SEP] '.join(sentences)  # Combine sentences into a single text
    formatted_text = f"[CLS] '{aspect}' [SEP] {formatted_sentences}"
    # formatted_text = f"[CLS] '{formatted_sentences}"
    new_rows.append({'text': formatted_text, 'sentiment': row['sentiment']})

df_test = pd.DataFrame(new_rows)


def preprocess_test_data(df):
    sentences = df['text'].tolist()
    encoded_data = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = df['sentiment'].tolist()

    return TensorDataset(input_ids, attention_masks), labels


inference_input, labels = preprocess_test_data(df_test)

model = BertForSequenceClassification.from_pretrained('bert-base-german-cased', num_labels=5)
state_dict = torch.load('C:\\Users\\tijst\\Downloads\\best_model (1).pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

batch_size = 16
dataloader = DataLoader(inference_input, sampler=SequentialSampler(inference_input), batch_size=batch_size)

all_probs = []

with torch.no_grad():
    for batch in dataloader:
        input_ids, attention_masks = batch
        outputs = model(input_ids, attention_mask=attention_masks)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        all_probs.extend(probs.tolist())

# Convert labels to tensor for comparison
labels_tensor = torch.tensor(labels)
all_probs = torch.tensor(all_probs)


def evaluate_confidence_thresholds(all_probs, labels, thresholds):
    results = []
    confident_predictions = torch.max(all_probs, dim=1)
    confidences, predictions = confident_predictions.values.tolist(), confident_predictions.indices.tolist()
    inference_df = pd.DataFrame({'label': labels.tolist(), 'probability': confidences, 'prediction': predictions})

    for threshold in thresholds:
        threshold_df = inference_df[inference_df['probability'] >= threshold]
        correct_predictions = (threshold_df['label'] == threshold_df['prediction']).sum()

        precision, recall, fscore, support = precision_recall_fscore_support(threshold_df['label'], threshold_df['prediction'], average='macro')

        results.append({'Threshold': threshold, 'Accuracy': correct_predictions/len(threshold_df), 'Percentage of predictions': len(threshold_df)/len(inference_df), 'F1-score': fscore, 'Precision': precision, 'Recall': recall})

    return pd.DataFrame(results)


thresholds = np.arange(0.0, 1.1, 0.1)

# Evaluate the accuracy at different confidence thresholds
# results_df = evaluate_confidence_thresholds(all_probs, labels_tensor, thresholds)

predicted_labels = torch.argmax(all_probs, dim=1)

labels_np = labels_tensor.numpy()
predicted_labels_np = predicted_labels.numpy()

# Compute confusion matrix
cm = confusion_matrix(labels_np, predicted_labels_np, labels=[0, 1, 2, 3, 4])

# Print confusion matrix
print("Confusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negativ', 'leicht negativ', 'ausgeglichen', 'positiv', 'leicht positiv'],
            yticklabels=['negativ', 'leicht negativ', 'ausgeglichen', 'positiv', 'leicht positiv'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
a = 1
