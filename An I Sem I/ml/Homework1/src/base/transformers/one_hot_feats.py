import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

class OneHotFeats(BaseEstimator, TransformerMixin):
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

        self.label_encoders = {}
        self.one_hot_encoders = {}
        for field, tp in self.fields_type.items():
            if tp == 'c':
                le = LabelEncoder()
                field_encoded = le.fit_transform(X_new[:, self.fields_dict[field]])
                ohe = OneHotEncoder(sparse=False)
                field_encoded = field_encoded.reshape(len(field_encoded), 1)
                ohe.fit(field_encoded)

                self.label_encoders[field] = le
                self.one_hot_encoders[field] = ohe

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
        X = np.append(X, X_real, axis=1)
        for field, tp in self.fields_type.items():
            if tp == 'c':
                le = self.label_encoders[field]
                field_encoded = le.transform(X_new[:, self.fields_dict[field]])
                ohe = self.one_hot_encoders[field]
                field_encoded = field_encoded.reshape(len(field_encoded), 1)
                field_encoded = ohe.transform(field_encoded)
                for idx in range(field_encoded.shape[1]):
                    field_name = field + '_' + str(idx)
                    fields_dict_new[field_name] = len(fields_dict_new)
                    fields_type_new[field_name] = 'c'
                X = np.append(X, field_encoded, axis=1)

        return X, fields_dict_new, fields_type_new
