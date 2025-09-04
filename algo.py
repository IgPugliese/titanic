import pandas as pd
df = pd.read_csv("train.csv")
df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))

print('Switching male/female for binary values')
print(df.head())

print('First class passengers')
first_class_passagers = df.loc[df['Pclass'] == 1]
second_class_passagers = df.loc[df['Pclass'] == 2]
tird_class_passagers = df.loc[df['Pclass'] == 3]


print('counting the amount of unique fares in first class')
first_class_fares = pd.Series(data=first_class_passagers['Fare'])

ages_by_fare = (
    first_class_passagers
    .groupby("Fare")
    .agg(
        count=("Fare", "size"),        # number of occurrences
        ages=("Age", lambda x: list(x.dropna().astype(int)))  # collect ages as list
    )
    .sort_values("count", ascending=False)
)

print(ages_by_fare)
## no evident pattern age -fare??

first_class_age_mean = first_class_passagers['Age'].mean()
second_class_age_mean = second_class_passagers['Age'].mean()
trird_class_age_mean = tird_class_passagers['Age'].mean()

print(first_class_age_mean, second_class_age_mean, trird_class_age_mean)
