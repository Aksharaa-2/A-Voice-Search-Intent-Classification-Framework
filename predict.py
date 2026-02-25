import torch
import joblib
from transformers import DistilBertTokenizer
from multi_task_model import MultiTaskModel

intent_encoder = joblib.load("intent_encoder.pkl")
voice_encoder = joblib.load("voice_encoder.pkl")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

num_intents = len(intent_encoder.classes_)
model = MultiTaskModel(num_intents)
model.load_state_dict(torch.load("voice_multitask_model.pt"))
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        intent_logits, voice_logits = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )

    intent_pred = torch.argmax(intent_logits, dim=1).item()
    voice_pred = torch.argmax(voice_logits, dim=1).item()

    intent = intent_encoder.inverse_transform([intent_pred])[0]
    voice = voice_encoder.inverse_transform([voice_pred])[0]

    return intent, voice

while True:
    text = input("You: ")
    intent, voice = predict(text)
    print("Intent:", intent)
    print("Voice Style:", voice)