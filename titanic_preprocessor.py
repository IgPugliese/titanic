import re

import numpy as np


class TitanicPreprocessor():
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

        # Cabin
        df["Cabin"] = df['Cabin'].transform(lambda x: (~x.isna()).astype(bool))

        # Embarked
        df['Embarked'] = df['Embarked'].fillna('S')
        df["Embarked"] = df["Embarked"].astype("category")

        # Fare
        mask = df['Fare'] > 0
        df.loc[mask, 'Fare'] = np.log1p(df.loc[mask, 'Fare'])

        # Title
        df['Title']=df['Name'].map(self.modelate_nobiliarie_titles)
        df["Title"] = df["Title"].astype("category")


        #df = pd.get_dummies(df, columns=['Embarked',"Title"], prefix=['OheEmbarked','OheTitle'])
        return df

    def modelate_nobiliarie_titles(self, nombre):
        nobiliary_diccionary={'Mrs.':'Mrs','Miss.':'Miss','Mr.':'Mr','Master.':'Master'}
        match= re.search(r'\b\w+\.', nombre)
        nobiliary_title=match.group().lstrip()
        return nobiliary_diccionary.get(nobiliary_title,'Rare')
    
    def lookup_age(self, row):
        return self.age_medians_per_class_and_sex.get((row["Pclass"], row["Sex"]))

  