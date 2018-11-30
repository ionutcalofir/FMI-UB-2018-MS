import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class ScaleData(BaseEstimator, TransformerMixin):
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

        self.ss = StandardScaler()
        if X_real.shape[1] != 0:
            self.ss.fit(X_real)

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
        for idx, (field, tp) in enumerate(self.fields_type.items()):
            if tp == 'r':
                fields_dict_new[field] = len(fields_dict_new)
                fields_type_new[field] = self.fields_type[field]
        X_real_t = self.ss.transform(X_real) if X_real.shape[1] != 0 else X_real
        X = np.append(X, X_real_t, axis=1)
        for idx, (field, tp) in enumerate(self.fields_type.items()):
            if tp == 'c':
                fields_dict_new[field] = len(fields_dict_new)
                fields_type_new[field] = self.fields_type[field]
        X = np.append(X, X_cat, axis=1)

        return X
