import pandas as pd
data = pd.read_csv("train.csv")
data["Sex"] = data['Sex'].transform(lambda x: (x == "male").astype(int))
print(data.head())