import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self,
                 data_path='data/Calofir_A_Petrișor_Ionuț_train.csv'):
        self._data_path = data_path
        self._data_df = None
        self._read_dataset()
        self._classification_field = 'Breed Name'
        self._regression_field = 'Longevity(yrs)'
        self._unnecessary_fields = ['Owner Name']

        self._fields = self._data_df.keys().values
        self._fields_dict = {k: v for (v, k) in enumerate(self._data_df.keys())}

    def _read_dataset(self):
        self._data_df = pd.read_csv(self._data_path)

    def get_data_classification(self, data, test_size=0.2):
        classes = np.unique(data[:, self._fields_dict[self._classification_field]])
        self._name_to_classes = {k: v for (v, k) in enumerate(classes)}

        X = np.empty((data.shape[0], 0))
        for field in self._fields:
            if field != self._classification_field \
                and field != self._regression_field \
                and field not in self._unnecessary_fields:
                X = np.append(X, data[:, self._fields_dict[field]].reshape(-1, 1), axis=1)

        y = np.array([self._name_to_classes[cls]
                      for cls in data[:, self._fields_dict[self._classification_field]]])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        X_fields = np.delete(self._fields, [self._fields_dict[self._classification_field],
                                            self._fields_dict[self._regression_field]])

        X_fields_dict = {k: v for (v, k) in enumerate(X_fields)}
        X_fields = np.delete(X_fields, [X_fields_dict[field]
                                            for field in self._unnecessary_fields])

        X_fields_dict = {k: v for (v, k) in enumerate(X_fields)}
        X_fields_type = {'Weight(g)': 'r',
                         'Height(cm)': 'r',
                         'Energy level': 'c',
                         'Attention Needs': 'c',
                         'Coat Lenght': 'c',
                         'Sex': 'c'}

        return X_train, X_test, y_train, y_test, X_fields_dict, X_fields_type

    def get_data_predict_file(self, data):
        X = np.empty((data.shape[0], 0))
        for field in self._fields:
            if field != self._classification_field \
                and field != self._regression_field \
                and field not in self._unnecessary_fields:
                X = np.append(X, data[:, self._fields_dict[field]].reshape(-1, 1), axis=1)

        return X

    def get_data_regression(self, data, test_size=0.2):
        X = np.empty((data.shape[0], 0))
        for field in self._fields:
            if field != self._classification_field \
                and field != self._regression_field \
                and field not in self._unnecessary_fields:
                X = np.append(X, data[:, self._fields_dict[field]].reshape(-1, 1), axis=1)

        y = np.empty((X.shape[0], 0))
        y_reg = np.array([val
                          for val in data[:, self._fields_dict[self._regression_field]]])
        classes = np.unique(data[:, self._fields_dict[self._classification_field]])
        self._name_to_classes = {k: v for (v, k) in enumerate(classes)}
        y_cls = np.array([self._name_to_classes[cls]
                          for cls in data[:, self._fields_dict[self._classification_field]]])

        y = np.append(y, y_reg.reshape(-1, 1), axis=1)
        y = np.append(y, y_cls.reshape(-1, 1), axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        y_train_reg = y_train[:, 0]
        y_train_cls = y_train[:, 1]
        y_test_reg = y_test[:, 0]
        y_test_cls = y_test[:, 1]

        X_fields = np.delete(self._fields, [self._fields_dict[self._classification_field],
                                            self._fields_dict[self._regression_field]])

        X_fields_dict = {k: v for (v, k) in enumerate(X_fields)}
        X_fields = np.delete(X_fields, [X_fields_dict[field]
                                            for field in self._unnecessary_fields])

        X_fields_dict = {k: v for (v, k) in enumerate(X_fields)}
        X_fields_type = {'Weight(g)': 'r',
                         'Height(cm)': 'r',
                         'Energy level': 'c',
                         'Attention Needs': 'c',
                         'Coat Lenght': 'c',
                         'Sex': 'c'}

        return X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls, X_fields_dict, X_fields_type

    @property
    def get_dataset(self):
        return self._data_df
