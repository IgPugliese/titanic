
import sys
from pathlib import Path
from dataclasses import dataclass, field
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from ticket_preprocessor import TicketPreprocessor
from utils import load_config, load_datasets
import pandas as pd
import numpy as np
import re
import sklearn as skt
from sklearn.model_selection import KFold, train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor
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
dataFrameTrain, _ = load_datasets()
dataFrameTrain['TicketCount'] = dataFrameTrain['Ticket'].transform(lambda x: len(dataFrameTrain[dataFrameTrain['Ticket'] == x]))

X=dataFrameTrain.drop(columns=["TicketCount", "Survived"])
Y=dataFrameTrain["TicketCount"]


model = RandomForestRegressor(    
  )

data_preprocessor = TicketPreprocessor()
pipeline = Pipeline([('pre-processor', data_preprocessor), ('features', model)])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, Y, scoring="r2", cv=cv, n_jobs=-1)

pipeline.fit(X, Y)
trained_model = pipeline.named_steps['features']
joblib.dump(trained_model, "trained_ticket_model.joblib")

print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f},, {scores.mean()+scores.std():.4f}")

