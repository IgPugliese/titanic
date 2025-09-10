import pandas as pd
df = pd.read_csv("train.csv")

# going crazy
pd.set_option('display.max_rows', None)      

# setting sex as binary
df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))

# filling ages
org_ages = df.loc[df['Age'].isna()].index
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median')).round()
predicted_ages = df.loc[org_ages]


df['FamilyNumber'] = df['SibSp'] + df['Parch']

df["Cabin"] = df['Cabin'].transform(lambda x: (~x.isna()).astype(bool))
df['Embarked'] = df['Embarked'].fillna('S')
df['TicketCount'] = df['Ticket'].transform(lambda x: len(df[df['Ticket'] == x]))
df['FarePerPerson'] = df['Fare'] / df['TicketCount']


print(df[df['Pclass' ]==2]['Fare'].values)
print(df[df['Pclass' ]==2]['FarePerPerson'].values)





# deleting passager id to avoid noise
del df['PassengerId']






