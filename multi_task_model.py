import torch
import torch.nn as nn
from transformers import DistilBertModel

class MultiTaskModel(nn.Module):
    def __init__(self, num_intents):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        hidden = self.bert.config.hidden_size

        self.intent_classifier = nn.Linear(hidden, num_intents)
        self.voice_classifier = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]

        intent_logits = self.intent_classifier(pooled)
        voice_logits = self.voice_classifier(pooled)

        return intent_logits, voice_logits