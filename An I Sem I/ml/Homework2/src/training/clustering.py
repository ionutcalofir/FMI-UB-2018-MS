import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD

from base.feature_computer import FeatureComputer
from base.training.base_trainer import BaseTrainer
from dataset import Dataset

class Clustering(BaseTrainer):
    def __init__(self,
                 data_path='data/Calofir A. Petrișor-Ionuț.csv',
                 tf_idf=None,
                 count_vectorizer=None,
                 dim_reduction=None,
                 clf=None):
        self._tf_idf = tf_idf
        self._count_vectorizer = count_vectorizer
        self._dim_reduction = dim_reduction
        self._clf = clf
        super().__init__(data_path)

    def _build_train(self, data_path):
        self._data = Dataset(data_path)
        self._X_train, self._y_train = \
            self._data.get_data_clustering()

        self._model = FeatureComputer(
                        tf_idf=TfidfVectorizer() if self._tf_idf else None,
                        count_vectorizer=CountVectorizer() if self._count_vectorizer else None,
                        dim_reduction=self._dim_reduction,
                        clf=self._clf
                    )

    def save_model(self, name='default'):
        joblib.dump(self, 'models/' + name + '.joblib')
