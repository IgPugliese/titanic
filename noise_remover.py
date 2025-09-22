class NoiseRemover():
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        del df['Ticket']
        del df['PassengerId']
        del df['Name']
        del df['Cabin']
        del df['Sex']
        del df["SibSp"]
        del df["Parch"]
        return df
