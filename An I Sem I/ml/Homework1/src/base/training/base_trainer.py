class BaseTrainer():
    def __init__(self,
                 data_path):
        self._data_path = data_path

        self._build_train(data_path)

    def fit(self):
        return self._model.fit(self._X_train, self._y_train)
