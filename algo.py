import pandas as pd
import numpy as np
import re
import sklearn as skt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("train.csv")
 #{}
# going crazy
pd.set_option('display.max_rows', None)      

# setting sex as binary
df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))

# filling ages
org_ages = df.loc[df['Age'].isna()].index
df['Age'] = df['Age'].fillna(df.groupby(['Pclass', 'Sex'])['Age'].transform('median')).round()
predicted_ages = df.loc[org_ages]

# total family number
df['FamilyNumber'] = df['SibSp'] + df['Parch']

# cast cabin to binary
df["Cabin"] = df['Cabin'].transform(lambda x: (~x.isna()).astype(bool))

#first approac to filling embarked
df['Embarked'] = df['Embarked'].fillna('S')

#creation of ticketCount, farePerPerson and their transform to log scale
df['TicketCount'] = df['Ticket'].transform(lambda x: len(df[df['Ticket'] == x]))
df['FarePerPerson'] = df['Fare'] / df['TicketCount']
df['FarePerPerson'] = np.where(df['FarePerPerson'] == 0, 0, np.log(df['FarePerPerson']))
df['Fare'] = np.where(df['Fare'] == 0, 0, np.log(df['Fare']))

#model titles in names
def modelate_nobiliarie_titles(nombre):
    nobiliary_diccionary={'Mrs.':'Mrs','Miss.':'Miss','Mr.':'Mr','Master.':'Master'}
    match= re.search(r'\b\w+\.', nombre)
    nobiliary_title=match.group().lstrip()
    return nobiliary_diccionary.get(nobiliary_title,'Rare')

#creating columns with titles
df['Title']=df['Name'].map(modelate_nobiliarie_titles)



#drop unnecesarie and noise columns
del df['Ticket']
del df['PassengerId']
del df['Name']
del df['Fare']
y=df.pop('Survived')

df = pd.get_dummies(df, columns=['Embarked',"Title"], prefix=['OheEmbarked','OheTitle'])

X_train, X_test, y_train, y_test = train_test_split(df, y, random_state=0,test_size=0.1)

model= LogisticRegression(max_iter=10000)
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
print(model.score(X_test,y_test))








