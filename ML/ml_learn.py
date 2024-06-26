from transformers import AdamW, XLMRobertaForSequenceClassification
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import time
from main import EXCEL_FOLDER
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from ML.ml_functions import *


TRAINING_MODE = 'bt'
train_data = pd.read_excel(EXCEL_FOLDER + '\\train_data.xlsx')
train_data = train_data.loc[(train_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]

if TRAINING_MODE == 'bt':
    bt_data_en = pd.read_excel(EXCEL_FOLDER + '\\BT\\train_data_bt_en_all.xlsx')
    bt_data_en = bt_data_en.loc[(bt_data_en['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]
    train_data = pd.concat([train_data, bt_data_en], ignore_index=True)

    # bt_data_cz = pd.read_excel(EXCEL_FOLDER + '\\BT\\train_data_bt_cz_all.xlsx')
    # bt_data_cz = bt_data_cz.loc[(bt_data_cz['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]
    # train_data = pd.concat([train_data, bt_data_cz], ignore_index=True)

validation_data = pd.read_excel(EXCEL_FOLDER + '\\validation_data.xlsx')
validation_data = validation_data[(validation_data['aspect'] == 'Ländliche Entwicklung, Digitale Innovation')]

df_train = make_cls_sep(train_data)
df_validation = make_cls_sep(validation_data)

train_dataset = preprocess_data(df_train)
valid_dataset = preprocess_data(df_validation)

classes_tensor = torch.tensor(df_train['sentiment'].values, dtype=torch.long)
unique_classes = torch.unique(classes_tensor)
class_weights = compute_class_weight('balanced', classes=[0,1,2,3,4], y=classes_tensor.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=5)
batch_size = 8
epochs = 5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

learning_rates = [3e-4, 1e-4, 5e-5, 3e-5]
for learning_rate in reversed(learning_rates):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    class_weights = class_weights.to(device)
    torch.cuda.empty_cache()

    best_val_loss = 1_000_000
    loss_fn = CrossEntropyLoss(weight=class_weights)

    print('We start training')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        start_time_train = time.time()

        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, inputs['labels'])
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        end_time_train = time.time()
        print(f'Training epoch: {epoch} took {end_time_train-start_time_train} seconds')

        # Calculate average training loss for this epoch
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        start_time_val = time.time()

        for batch in tqdm(valid_loader, desc="Validating"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).flatten()
            labels = inputs['labels'].flatten()

            for pred, label in zip(predictions, labels):
                if pred == label:
                    print(f"Predicted: {pred}, Actual: {label}")
                else:
                    print(pred)

            total_eval_accuracy += (predictions == labels).sum().item()

            # Calculate validation loss
            loss = loss_fn(outputs.logits, inputs['labels'])
            total_eval_loss += loss.item()

        end_time_val = time.time()
        print(f'Validating epoch: {epoch} took {end_time_val-start_time_val} seconds')

        # Calculate average validation loss and accuracy
        avg_val_loss = total_eval_loss / len(valid_loader)
        avg_val_accuracy = total_eval_accuracy / len(valid_dataset)

        print(f'Epoch {epoch + 1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  Validation Accuracy: {avg_val_accuracy:.2f}')

        # Check if current validation loss is lower than best validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model checkpoint with the lowest validation loss
            torch.save(model.state_dict(), f'C:\\Users\\tijst\\Downloads\\best_{learning_rate}_model.pth')
        print(f'Best val loss: {best_val_loss} best learning rate: {learning_rate}')
        model.train()  # Set back to train mode

    print(f'Best Validation Loss: {best_val_loss:.4f}')

