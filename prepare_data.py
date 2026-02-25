import pandas as pd

voice_df = pd.read_csv("data/clean_voice_dataset.csv")
typed_df = pd.read_csv("data/typed_queries.csv")

final_df = pd.concat([voice_df, typed_df])

intent2id = {
    "information": 0,
    "shopping": 1,
    "navigation": 2,
    "command": 3
}

# CREATE LABEL COLUMN
final_df["intent_label"] = final_df["intent"].map(intent2id)

# SAVE AFTER creating label
final_df.to_csv("data/final_dataset.csv", index=False)

print("Final dataset created!")
print(final_df.head())