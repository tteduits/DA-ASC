import pandas as pd
from main import EXCEL_FOLDER
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split

train_data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
train_data = train_data[(train_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]

validation_data = pd.read_excel(EXCEL_FOLDER + '\\validation_data.xlsx')
validation_data = validation_data[(validation_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]

test_data = pd.read_excel(EXCEL_FOLDER + '\\test_data.xlsx')
test_data = test_data[(test_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the text data
train_encodings = tokenizer(train_data['clean_text'].tolist(), train_data['aspect'].tolist(), truncation=True, padding=True, return_tensors='tf')
val_encodings = tokenizer(validation_data['clean_text'].tolist(), validation_data['aspect'].tolist(), truncation=True, padding=True, return_tensors='tf')
test_encodings = tokenizer(test_data['clean_text'].tolist(), test_data['aspect'].tolist(), truncation=True, padding=True, return_tensors='tf')

# Convert labels to TensorFlow tensors
y_train = tf.convert_to_tensor(train_data['sentiment'])
y_val = tf.convert_to_tensor(validation_data['sentiment'])
y_test = tf.convert_to_tensor(test_data['sentiment'])

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])


history = model.fit(
    train_encodings.data,  # Accessing the underlying numpy array
    y_train,
    validation_data=(val_encodings.data, y_val),  # Accessing the underlying numpy array
    epochs=5,
    batch_size=16,
    verbose=1,
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_encodings, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
