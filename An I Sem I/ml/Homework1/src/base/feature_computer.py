from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class FeatureComputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 select_data=None,
                 fill_data=None,
                 poly_feats=None,
                 one_hot_feats=None,
                 scale_data=None,
                 clf=None):
        self.select_data = select_data
        self.fill_data = fill_data
        self.poly_feats = poly_feats
        self.one_hot_feats = one_hot_feats
        self.scale_data = scale_data
        self.clf = clf
        self._pipeline = Pipeline([('select_data', select_data),
                                   ('fill_data', fill_data),
                                   ('poly_feats', poly_feats),
                                   ('one_hot_feats', one_hot_feats),
                                   ('scale_data', scale_data),
                                   ('clf', clf)])

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self._pipeline.transform(X)

    def predict(self, X):
        return self._pipeline.predict(X)
