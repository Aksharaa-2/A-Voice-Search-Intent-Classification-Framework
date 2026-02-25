import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("data/clean_voice_dataset.csv")

# Encode intent
intent_encoder = LabelEncoder()
df["intent_label"] = intent_encoder.fit_transform(df["intent"])

# Encode voice_style
voice_encoder = LabelEncoder()
df["voice_label"] = voice_encoder.fit_transform(df["voice_style"])

joblib.dump(intent_encoder, "intent_encoder.pkl")
joblib.dump(voice_encoder, "voice_encoder.pkl")

df.to_csv("data/encoded_voice_dataset.csv", index=False)

print("Encoding completed successfully.")
print("Intent classes:", intent_encoder.classes_)
print("Voice style classes:", voice_encoder.classes_)