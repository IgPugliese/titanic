class NoiseRemover():
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        del df['Cabin']
        del df['Sex']
        del df['Fare']
    
        return df
