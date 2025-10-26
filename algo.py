
import sys
from pathlib import Path
from dataclasses import dataclass, field

from ticket_preprocessor import TicketPreprocessor
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from data_filler import DataFiller
from utils import load_config, load_datasets
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
from titanic_preprocessor import TitanicPreprocessor
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from xgboost.callback import EarlyStopping

load_config()
dataFrameTrain,dataFrameTest = load_datasets()




X=dataFrameTrain.drop(columns=["Survived"])
Y=dataFrameTrain["Survived"]

#print(dataFrameTest)


ids_to_predict=dataFrameTest["PassengerId"]


model= XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",       # modela conteos
    tree_method="hist",  
    colsample_bytree = 0.9,
    n_estimators=52,
    learning_rate=0.4,
    subsample=0.9,
    max_depth=3,
    reg_lambda=1,            # L2 ayuda a estabilizar
    reg_alpha=20,
    enable_categorical=True,
    seed=42
)

data_preprocessor = TitanicPreprocessor()

pipeline = Pipeline([('pre-processor', TicketPreprocessor()),('fill_ticket', DataFiller()), ('nosie remover', NoiseRemover()),('rfb', model)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, Y, scoring="accuracy", cv=cv, n_jobs=-1)
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f},, {scores.mean()+scores.std():.4f}")

pipeline.fit(X,Y)
predictions_array=pipeline.predict(dataFrameTest)
results = pd.DataFrame({'PassengerId' : ids_to_predict, 'Survived' : pd.Series(predictions_array)})
results.to_csv("nueva_feature2.csv", index=False)