import pandas as pd
import numpy as np
import re
import sklearn as skt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

df = pd.read_csv("train.csv")
pd.set_option('display.max_rows', None)      

class XPreprosessior():
    def __init__(self):
        self.age_medians_per_class_and_sex = None
        
    def fit(self, X, y=None):
        self.age_medians_per_class_and_sex = X.groupby(['Pclass', 'Sex'])['Age'].median().to_dict()
        return self

    def transform(self, X):
        df = X.copy()

        # filling ages
        mask = df["Age"].isna()
        df.loc[mask, 'Age'] = df[mask].apply(self.lookup_age, axis=1)

        df["Sex"] = df['Sex'].transform(lambda x: (x == "male").astype(int))

        # total family number
        df['FamilyNumber'] = df['SibSp'] + df['Parch']

        # cast cabin to binary
        df["Cabin"] = df['Cabin'].transform(lambda x: (~x.isna()).astype(bool))

        #first approac to filling embarked
        df['Embarked'] = df['Embarked'].fillna('S')

        mask = df['Fare'] > 0
        df.loc[mask, 'Fare'] = np.log1p(df.loc[mask, 'Fare'])

        df['Title']=df['Name'].map(self.modelate_nobiliarie_titles)
        del df['Ticket']
        del df['PassengerId']
        del df['Name']

        df = pd.get_dummies(df, columns=['Embarked',"Title"], prefix=['OheEmbarked','OheTitle'])
        return df

    def modelate_nobiliarie_titles(self, nombre):
        nobiliary_diccionary={'Mrs.':'Mrs','Miss.':'Miss','Mr.':'Mr','Master.':'Master'}
        match= re.search(r'\b\w+\.', nombre)
        nobiliary_title=match.group().lstrip()
        return nobiliary_diccionary.get(nobiliary_title,'Rare')
    
    def lookup_age(self, row):
        return self.age_medians_per_class_and_sex.get((row["Pclass"], row["Sex"]))
    
y=df.pop('Survived')

data_preprocessor = XPreprosessior()
model= LogisticRegression(max_iter=10000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipeline = Pipeline([('pre-processor', data_preprocessor), ('logisticRegression', model)])
cv_scores = cross_val_score(pipeline, df, y, cv=cv, scoring='accuracy')


print(f"Mean CV Precision: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")






