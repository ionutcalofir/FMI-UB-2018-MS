from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

class FeatureComputer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 tf_idf=None,
                 count_vectorizer=None,
                 dim_reduction=None,
                 clf=None):
        self.tf_idf = tf_idf
        self.count_vectorizer = count_vectorizer
        self.dim_reduction = dim_reduction
        self.clf = clf
        self._pipeline = Pipeline([('tf_idf', tf_idf),
                                   ('count_vectorizer', count_vectorizer),
                                   ('dim_reduction', dim_reduction),
                                   ('clf', clf)])

    def fit(self, X, y=None):
        self._pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self._pipeline.transform(X)

    def predict(self, X):
        return self._pipeline.predict(X)
