import pandas as pd

df = pd.read_csv("../data/train_dataset.csv")

val_df = df.sample(
    n=64,
    random_state=42
)

train_df = df.drop(
    val_df.index
)

train_df.to_csv(
    "../data/train_split.csv",
    index=False
)

val_df.to_csv(
    "../data/validation_split.csv",
    index=False
)

print("Training Samples :", len(train_df))
print("Validation Samples:", len(val_df))