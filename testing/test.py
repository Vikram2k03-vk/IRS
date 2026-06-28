import pandas as pd

df = pd.read_csv("../data/train_split.csv")
print(df.iloc[0:50,181:])

'''test.py'''