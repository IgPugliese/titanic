import sys
from pathlib import Path
project_root = Path().resolve().parent
sys.path.append(str(project_root))

from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
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

from utils import load_config, load_datasets

load_config()
dataset, dt = load_datasets()

model= XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",       # modela conteos
    tree_method="hist",         # rápido y eficiente en CPU
    n_estimators=5000,
    learning_rate=0.1,
    callbacks= [EarlyStopping(rounds=50, save_best=True)],
    max_depth=3,
    enable_categorical=True,
    verbosity=0
)

def generate_predefined_split(y, prep):
    X_train, X_eval, y_train, y_eval = train_test_split(df, y, test_size=0.2, stratify=y)
    X_train_preprocessed = prep.fit_transform(X_train, y_train)  
    X_eval_preprocessed  = prep.transform(X_eval)   
    X_preprocessed = pd.concat([X_train_preprocessed, X_eval_preprocessed], axis=0)
    y_all   = pd.concat([y_train, y_eval], axis=0)
    test_fold = np.r_[[-1]*len(X_train_preprocessed), [0]*len(X_eval_preprocessed)]
    return X_preprocessed, X_eval_preprocessed, y_all, y_eval, test_fold


param_grid = {
    'max_depth': [2, 3, 4, 5, 7, 9 ,12, 13, 14, 15],
    'learning_rate': [0.008,0.009 ,0.01, 0.1, 0.2, 0.8],
    'subsample': [0.5, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.3, 0.8, 0.9, 1.0]
}

df=dataset.copy()
y = df.pop('Survived')
dfcv=df.copy()
params=[]

class XGBoostCVWrapper(BaseEstimator, TransformerMixin):
    def __init__(self,max_depth=7, learning_rate=0.01, subsample=0.9, colsample_bytree=0.9, n_estimators=5000 ):
        self.max_depth = max_depth
        self.subsample = subsample
        self.n_estimators = n_estimators
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.cv_results_ = None
        self.best_iteration_ = None
    
    def fit(self, X, y=None):
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
        self.cv_results_ = xgb.cv(
        params={'max_depth': self.max_depth, 'learning_rate': self.learning_rate, 'subsample': self.subsample, 'colsample_bytree': self.colsample_bytree,  'objective': 'binary:logistic'},  # objective needed for regression
        dtrain=dtrain,
        num_boost_round=50000,
        nfold=5,
        metrics='logloss',
        early_stopping_rounds=50,
        seed=42,
        as_pandas=True,
        verbose_eval=False
        )
        self.best_iteration_ = len(self.cv_results_)

        cv=StratifiedKFold(n_splits=5,shuffle=False,random_state=None)

        self.model_ = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            n_estimators=self.best_iteration_,
            enable_categorical=True,
            eval_metric='logloss'
        )

        
        pipeline = Pipeline([('pre-processor', data_preprocessor), ('nosie remover', NoiseRemover()),('rfb', self.model_)])


        self.score_val = cross_val_score(pipeline, dfcv,y,cv=cv,scoring="log_loss").mean()
        self.model_.fit(X, y)
        self.best_result_ = self.cv_results_['test-logloss-mean'].min()

        print(f"'max_depth': {self.max_depth}, 'learning_rate': {self.learning_rate}, 'subsample': {self.subsample}, 'colsample_bytree': {self.colsample_bytree} 'n_estimators': {self.best_iteration_} , result: {self.score_val}")
        return self
    
    
    def score(self, X, y):
        return self.score_val
    
    
data_preprocessor = TitanicPreprocessor()

modelo_falopa = XGBoostCVWrapper()
prep = Pipeline([('pre-processor', TitanicPreprocessor()),('noise_remover', NoiseRemover())])

prep.fit(df)
X_preprocessed = prep.transform(df)
dummy_cv = [(np.arange(len(X_preprocessed)), np.arange(len(X_preprocessed)))]


grid_search = GridSearchCV(
    estimator=modelo_falopa,
    param_grid=param_grid,
    scoring=None,
    cv=dummy_cv,
    n_jobs=-1,  # Use all available cores
    verbose=0  # Print progress
)

print("Starting grid search...")
grid_search.fit(X_preprocessed, y)
print("Best score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)
print()







#[{'colsample_bytree': 0.8, 'learning_rate': 0.8, 'max_depth': 9, 'subsample': 0.5}, {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 5, 'subsample': 0.5}]

