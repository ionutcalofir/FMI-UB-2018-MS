import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class SelectData(BaseEstimator, TransformerMixin):
    def __init__(self, fields_dict, fields_type, fields):
        self.fields_dict = fields_dict
        self.fields_type = fields_type
        self.fields = fields

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = np.empty((X.shape[0], 0))
        fields_dict_new = {}
        fields_type_new = {}
        for idx, field in enumerate(self.fields):
            if field in self.fields_dict:
                fields_dict_new[field] = idx
                fields_type_new[field] = self.fields_type[field]
                X_new = np.append(X_new, X[:, self.fields_dict[field]].reshape(-1, 1), axis=1)
        fit_params = {'fields_dict': self.fields_dict, 'fields_type': self.fields_type}

        return X_new, fields_dict_new, fields_type_new
