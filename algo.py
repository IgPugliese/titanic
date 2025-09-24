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
dt, df = load_datasets()


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

param_grid = {
    'max_depth': [7, 9 ,12, 13, 14, 15],
    'learning_rate': [0.009 ,0.01, 0.1, 0.2, 0.8],
    'subsample': [0.5, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.3, 0.8, 0.9, 1.0]
}
  

y=df.pop('Survived')
data_preprocessor = TitanicPreprocessor()


pipeline = Pipeline([('pre-processor', data_preprocessor), ('nosie remover', NoiseRemover()),('rfb', model)])
prep = Pipeline([('pre-processor', TitanicPreprocessor()),('noise_remover', NoiseRemover())])



X_tr, X_va, y_tr, y_va = train_test_split(df, y, test_size=0.2, stratify=y)
X_tr_p = prep.fit_transform(X_tr, y_tr)  
X_va_p  = prep.transform(X_va)   
X_all_p = pd.concat([X_tr_p, X_va_p], axis=0)
y_all   = pd.concat([y_tr, y_va], axis=0)
test_fold = np.r_[[-1]*len(X_tr_p), [0]*len(X_va_p)]

cv = PredefinedSplit(test_fold)   

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,  # Use all available cores
    verbose=0  # Print progress
)


print("Starting grid search...")



grid_search.fit(X_all_p, y_all,eval_set=[(X_va_p, y_va)],verbose=False)

best_xgb = grid_search.best_estimator_
best_iter = getattr(best_xgb, "best_iteration", None)
print("Best params:", grid_search.best_params_)
print("Best (hold-out) score:", grid_search.best_score_)
print("Best n_estimators (ES):", (best_iter + 1) if best_iter is not None else "N/A")

# Best params found
#print("Best parameters:", grid_search.best_params_)

# Best cross-validation score
#print("Best CV score:", grid_search.best_score_)


#cv_results = pd.DataFrame(grid_search.cv_results_)
#print(cv_results[["params", "mean_test_score", "std_test_score"]])



