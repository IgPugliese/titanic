import re

import numpy as np
import pandas as pd


class TicketPreprocessor():
    def __init__(self):
        self.age_medians_per_class_and_sex = None
        
    def fit(self, X, y=None):
        self.age_medians_per_class_and_sex = X.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        return self

    def transform(self, X):
        df = X.copy()

        # Age
        mask = df["Age"].isna()
        df.loc[mask, 'Age'] = df[mask].apply(self.lookup_age, axis=1)

        # Sex
        df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))

        # FamilyNumber
        df['FamilyNumber'] = df['SibSp'] + df['Parch']
        del df['SibSp']
        del df['Parch']

        # Cabin
        df["Cabin"] = df['Cabin'].transform(lambda x: (~x.isna()).astype(bool))

        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        df["Embarked"] = df["Embarked"]

        # Fare
        # Rien

        # Title
        df['Title']=df['Name'].map(self.modelate_nobiliarie_titles)

        df = pd.get_dummies(df, columns=['Embarked',"Title"], prefix=['OheEmbarked','OheTitle'])
        del df['Name']
        del df['PassengerId']
        del df['Ticket']

        return df

    def modelate_nobiliarie_titles(self, nombre):
        nobiliary_diccionary={'Mrs.':'Mrs','Miss.':'Miss','Mr.':'Mr','Master.':'Master'}
        match= re.search(r'\b\w+\.', nombre)
        nobiliary_title=match.group().lstrip()
        return nobiliary_diccionary.get(nobiliary_title,'Rare')
    
    def lookup_age(self, row):
        return self.age_medians_per_class_and_sex.get((row["Pclass"], row["Sex"]))

  