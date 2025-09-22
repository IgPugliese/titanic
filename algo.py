import pandas as pd
import numpy as np
import re
import sklearn as skt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from noise_remover import NoiseRemover

df = pd.read_csv("train.csv")
dt = pd.read_csv("test.csv")
pd.set_option('display.max_rows', None)


model= XGBClassifier(
    objective="binary:logistic", # modela conteos
    tree_method="hist",         # rápido y eficiente en CPU
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    enable_categorical=True,
)

param_grid = {
    'rfb__n_estimators': [100, 200, 300, 500],
    'rfb__max_depth': [7, 9 ,12, 13, 14, 15],
    'rfb__learning_rate': [0.0001, 0.009 ,0.01, 0.1, 0.2, 0.8, 5],
    'rfb__subsample': [0.5, 0.8, 0.9, 1.0],
    'rfb__colsample_bytree': [0.3, 0.8, 0.9, 1.0]
}

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

    

y=df.pop('Survived')

data_preprocessor = TitanicPreprocessor()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


pipeline = Pipeline([('pre-processor', data_preprocessor), ('nosie remover', NoiseRemover()),('rfb', model)])
#cv_scores = cross_val_score(pipeline, df, y, cv=cv, scoring='accuracy')

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=1  # Print progress
)


print("Starting grid search...")
grid_search.fit(df, y)

# Best params found
print("Best parameters:", grid_search.best_params_)

# Best cross-validation score
print("Best CV score:", grid_search.best_score_)


cv_results = pd.DataFrame(grid_search.cv_results_)
#print(cv_results[["params", "mean_test_score", "std_test_score"]])

# pipeline.fit(df,y)

#predictions_array=pipeline.predict(dt)
#ids_to_predict=dt.pop('PassengerId')
#results = pd.DataFrame({'PassengerId' : ids_to_predict, 'Survived' : pd.Series(predictions_array)})
#results.to_csv("submission.csv", index=False)
#print(f"Mean CV Precision: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")









