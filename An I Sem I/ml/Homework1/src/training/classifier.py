import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from base.feature_computer import FeatureComputer
from base.training.base_trainer import BaseTrainer
from base.transformers.select_data import SelectData
from base.transformers.fill_data import FillData
from base.transformers.poly_feats import PolyFeats
from base.transformers.one_hot_feats import OneHotFeats
from base.transformers.scale_data import ScaleData
from dataset import Dataset

class Classifier(BaseTrainer):
    def __init__(self,
                 data_path,
                 clf=None):
        self._clf = clf
        super().__init__(data_path)

    def _build_train(self, data_path):
        self._data = Dataset(data_path)
        self._X_train, self._X_test, self._y_train, self._y_test, self._X_fields_dict, self._X_fields_type = \
            self._data.get_data_classification(self._data.get_dataset.values)

        self._model = FeatureComputer(
                        select_data=SelectData(
                            fields_dict = self._X_fields_dict,
                            fields_type = self._X_fields_type,
                            fields=['Weight(g)', 'Height(cm)', 'Energy level', 'Attention Needs', 'Coat Lenght', 'Sex']),
                        fill_data=FillData(),
                        poly_feats=PolyFeats(grade=2),
                        one_hot_feats=OneHotFeats(),
                        scale_data=ScaleData(),
                        clf=self._clf
                    )

    def grid_search(self, name='default', param_grid=None, n_splits=10):
        if param_grid is None:
            param_grid = {
                    'poly_feats__grade': [1, 2]
            }
        gsCV = GridSearchCV(
            self._model,
            param_grid,
            scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
            refit=False,
            return_train_score=True,
            cv=n_splits,
            n_jobs=-2,
            verbose=2
        )
        gsCV.fit(self._X_train, self._y_train)
        df = pd.DataFrame(data=gsCV.cv_results_)
        df.to_csv(path_or_buf=name + '.csv')

    def save_model(self, name='default'):
        joblib.dump(self, 'models/' + name + '.joblib')

    def predict_test(self, verbose=False):
        y_pred = self._model.predict(self._X_test)

        if verbose:
            print('Acc:', accuracy_score(self._y_test, y_pred))
            print('Precision Weighted:', precision_score(self._y_test, y_pred, average='weighted'))
            print('Recall Weighted:', recall_score(self._y_test, y_pred, average='weighted'))
            print('F1 Weighted:', f1_score(self._y_test, y_pred, average='weighted'))

        return y_pred

    def predict_train(self, verbose=False):
        y_pred = self._model.predict(self._X_train)

        if verbose:
            print('Acc:', accuracy_score(self._y_train, y_pred))
            print('Precision Weighted:', precision_score(self._y_train, y_pred, average='weighted'))
            print('Recall Weighted:', recall_score(self._y_train, y_pred, average='weighted'))
            print('F1 Weighted:', f1_score(self._y_train, y_pred, average='weighted'))

        return y_pred

    def predict_from_file(self, data_path):
        X = self._build_predict(data_path)
        y_pred = self._model.predict(X)

        return y_pred

    def _build_predict(self, data_path):
        data = Dataset(data_path)
        X = data.get_data_predict_file(data.get_dataset.values)

        return X
