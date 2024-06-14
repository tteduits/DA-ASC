import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset

# Custom Dataset class to handle your data structure
class CustomDataset(Dataset):
    def __init__(self, texts, aspects, labels, tokenizer, max_length):
        self.texts = texts
        self.aspects = aspects
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect = self.aspects[idx]
        label = self.labels[idx]

        # Combine text and aspect into a single input
        input_text = f"{text} [SEP] Aspect: {aspect}"

        # Tokenize input
        encoding = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

train_texts = [
    "The food was absolutely delicious, I couldn't get enough!",
    "The drinks were refreshing and perfectly mixed.",
    "The service was impeccable, the staff were attentive and friendly.",
    "The food was disappointing, lacked flavor and presentation.",
    "The drinks were overpriced for their quality.",
    "The service was slow and inattentive, had to wait too long for everything.",
    "The food was exquisite, every dish was a masterpiece.",
    "The drinks menu was extensive with a great variety to choose from.",
    "The service was outstanding, the staff went above and beyond to accommodate us.",
    "The food was mediocre, nothing special.",
    "The drinks were fantastic, especially the cocktails.",
    "The service was terrible, rude staff and slow service.",
    "The food was divine, flavors danced on my palate.",
    "The drinks were lackluster, not worth the price.",
    "The service was prompt and efficient, no complaints.",
    "The food was delectable, I savored every bite.",
    "The drinks selection was limited, could use more options.",
    "The service was friendly and attentive, made us feel welcome.",
    "The food was alright, but nothing memorable.",
    "The drinks were overpriced and underwhelming.",
    "The service was slow and forgetful, had to keep reminding them of our orders.",
    "The food was heavenly, exceeded my expectations.",
    "The drinks were superb, crafted with care and skill.",
    "The service was top-notch, the staff anticipated our needs.",
    "The food was bland and uninspiring, lacked seasoning.",
    "The drinks were overpriced, better options elsewhere.",
    "The service was excellent, attentive without being intrusive.",
    "The food was scrumptious, left me wanting more.",
    "The drinks were delightful, perfect for a relaxing evening.",
    "The service was subpar, had to constantly ask for refills and attention.",
    "The food was a bit underwhelming, expected more based on reviews.",
    "The drinks were refreshing, a great way to start the meal.",
    "The service was slow, seemed understaffed for the evening rush.",
    "The food was phenomenal, each dish was a work of art.",
    "The drinks were overpriced for what they offered.",
    "The service was friendly and efficient, no complaints here.",
    "The food was tasty, but portions were small for the price.",
    "The drinks were average, nothing stood out.",
    "The service was attentive, but seemed a bit rushed.",
    "The food was exquisite, a culinary delight.",
    "The drinks were mediocre, nothing special.",
    "The service was impeccable, the staff made us feel like VIPs.",
    "The food was disappointing, lacked freshness and creativity.",
    "The drinks were overpriced, better options nearby.",
    "The service was slow and inattentive, had to flag down waiters multiple times.",
    "The food was delicious, left me wanting to try more from the menu.",
    "The drinks were excellent, especially the wine selection.",
    "The service was top-notch, the staff were friendly and efficient.",
    "The food was decent, but didn't blow me away.",
    "The drinks were refreshing, perfect for a hot day."
]

train_aspects = [
    "food", "drinks", "service", "food", "drinks",
    "service", "food", "drinks", "service", "food",
    "drinks", "service", "food", "drinks", "service",
    "food", "drinks", "service", "food", "drinks",
    "service", "food", "drinks", "service", "food",
    "drinks", "service", "food", "drinks", "service",
    "food", "drinks", "service", "food", "drinks",
    "service", "food", "drinks", "service", "food",
    "drinks", "service", "food", "drinks", "service",
    "food", "drinks", "service", "food", "drinks"
]

train_labels = [
    1, 1, 1, 0, 0,
    0, 1, 1, 1, 0,
    1, 0, 1, 0, 1,
    1, 0, 1, 0, 0,
    0, 1, 1, 1, 0,
    0, 1, 1, 1, 0,
    1, 1, 0, 0, 1,
    0, 1, 1, 1, 1,
    0, 1, 0, 1, 1,
    1, 0, 0, 1, 1
]


# Define your test data
test_texts = [
    "The food was outstanding, with bold flavors and inventive combinations.",
    "The drinks were expertly crafted, with attention to detail.",
    "The service was attentive and friendly, making us feel like valued guests.",
    "The food was bad",
    "The drinks were mediocre, not worth the price."
]

test_aspects = [
    "food",
    "drinks",
    "service",
    "food",
    "drinks"
]

test_labels = [
    1, 1, 1, 0, 0
]


# Create CustomDataset instance
train_dataset = CustomDataset(train_texts, train_aspects, train_labels, tokenizer, max_length=512)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Training loop
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.train()

for epoch in range(3):  # 3 epochs as an example
    for batch in train_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    scheduler.step()

# Testing loop
test_dataset = CustomDataset(test_texts, test_aspects, test_labels, tokenizer, max_length=128)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model.eval()
total_accuracy = 0
model.eval()
predictions = []
true_labels = []
from sklearn.metrics import precision_recall_fscore_support
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=1)

        predictions.extend(batch_predictions.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
