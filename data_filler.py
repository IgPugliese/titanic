import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DataFiller(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = joblib.load("trained_ticket_model.joblib")
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        x = X.copy()
        x['TicketCount'] = self.model.predict(x).round()
        x['FarePerPerson'] = x['Fare'] / x['TicketCount']
        return x
