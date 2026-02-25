import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ----------------------------
# 1️⃣ Load Dataset
# ----------------------------
df = pd.read_csv("data/final_dataset.csv")

# Check required columns
if "query" not in df.columns or "intent_label" not in df.columns:
    raise ValueError("CSV must contain 'query' and 'intent_label' columns")

# Remove missing values
df = df.dropna()

# Convert float labels (0.0, 1.0) → int (0, 1)
df["intent_label"] = df["intent_label"].astype(int)

# Count number of unique labels
num_labels = len(df["intent_label"].unique())

print("Number of intents:", num_labels)

# ----------------------------
# 2️⃣ Train-Test Split
# ----------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["query"].tolist(),
    df["intent_label"].tolist(),
    test_size=0.2,
    random_state=42
)

# ----------------------------
# 3️⃣ Load Tokenizer
# ----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# ----------------------------
# 4️⃣ Custom Dataset Class
# ----------------------------
class IntentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# ----------------------------
# 5️⃣ Load Model
# ----------------------------
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels
)

# ----------------------------
# 6️⃣ Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10
)

# ----------------------------
# 7️⃣ Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# ----------------------------
# 8️⃣ Train Model
# ----------------------------
trainer.train()

# ----------------------------
# 9️⃣ Save Model
# ----------------------------
model.save_pretrained("intent_model")
tokenizer.save_pretrained("intent_model")

print("✅ Training Completed Successfully!")
print("✅ Model saved inside 'intent_model' folder")