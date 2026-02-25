import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from multi_task_model import MultiTaskModel
import torch.nn as nn

df = pd.read_csv("data/encoded_voice_dataset.csv")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class VoiceDataset(Dataset):
    def __init__(self, df):
        self.texts = df["query"].tolist()
        self.intent = df["intent_label"].tolist()
        self.voice = df["voice_label"].tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "intent": torch.tensor(self.intent[idx]),
            "voice": torch.tensor(self.voice[idx])
        }

dataset = VoiceDataset(df)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

num_intents = len(df["intent_label"].unique())
model = MultiTaskModel(num_intents)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

model.train()

for epoch in range(3):
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        intent_logits, voice_logits = model(
            batch["input_ids"],
            batch["attention_mask"]
        )

        loss1 = loss_fn(intent_logits, batch["intent"])
        loss2 = loss_fn(voice_logits, batch["voice"])

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

torch.save(model.state_dict(), "voice_multitask_model.pt")
print("Training complete.")