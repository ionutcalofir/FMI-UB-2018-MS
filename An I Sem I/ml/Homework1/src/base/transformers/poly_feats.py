import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class PolyFeats(BaseEstimator, TransformerMixin):
    def __init__(self, grade):
        self.grade = grade

    def fit(self, X, y=None):
        X_new = X[0]
        self.fields_dict = X[1]
        self.fields_type = X[2]

        X_cat = np.empty((X_new.shape[0], 0))
        X_real = np.empty((X_new.shape[0], 0))
        for field, tp in self.fields_type.items():
            if tp == 'r':
                X_real = np.append(X_real, X_new[:, self.fields_dict[field]].reshape(-1, 1), axis=1)
            elif tp == 'c':
                X_cat = np.append(X_cat, X_new[:, self.fields_dict[field]].reshape(-1, 1), axis=1)

        self.poly = PolynomialFeatures(self.grade)
        if X_real.shape[1] != 0:
            self.poly.fit(X_real)

        return self

    def transform(self, X):
        X_new = X[0]
        fields_dict_new = {}
        fields_type_new = {}

        X_cat = np.empty((X_new.shape[0], 0))
        X_real = np.empty((X_new.shape[0], 0))
        for field, tp in self.fields_type.items():
            if tp == 'r':
                X_real = np.append(X_real, X_new[:, self.fields_dict[field]].reshape(-1, 1), axis=1)
            elif tp == 'c':
                X_cat = np.append(X_cat, X_new[:, self.fields_dict[field]].reshape(-1, 1), axis=1)

        X = np.empty((X_new.shape[0], 0))
        # remove the first column of 1s
        X_real_t = self.poly.transform(X_real)[:, 1:] if X_real.shape[1] != 0 else X_real
        for idx in range(X_real_t.shape[1]):
            field = 'field_poly_' + str(idx)
            fields_dict_new[field] = len(fields_dict_new)
            fields_type_new[field] = 'r'
        X = np.append(X, X_real_t, axis=1)
        for idx, (field, tp) in enumerate(self.fields_type.items()):
            if tp == 'c':
                fields_dict_new[field] = len(fields_dict_new)
                fields_type_new[field] = self.fields_type[field]
        X = np.append(X, X_cat, axis=1)

        return X, fields_dict_new, fields_type_new
