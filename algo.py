import pandas as pd

df = pd.read_csv("train.csv")
df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))
org_ages = df['Age'].values
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
predicted_ages = df['Age'].values

for i, (x, y) in enumerate(zip(org_ages, predicted_ages)):
    color = '\033[92m\033[1m' if x != y else '\033[30m'
    enda = "\n" if i % 10 == 0 else "    "
    print(f"{color}{y:05.2f}\033[0m", end=enda)
