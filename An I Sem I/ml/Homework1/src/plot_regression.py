import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

from plot import Plot
from predicting.regressor_pred import RegressorPred

class PlotRegression(Plot):
    def __init__(self):
        super().__init__()
        self._class_field = 'Breed Name'
        self._longevity = 'Longevity(yrs)'

    def plot_dataset(self):
        classes_name = np.unique(self._X[:, self._fields_dict[self._class_field]])
        classes_dict = {k: v for (v, k) in enumerate(classes_name)}

        dogs = self._X[:, self._fields_dict[self._class_field]]
        dogs_int = np.array([classes_dict[dog] for dog in dogs])
        lgv = self._X[:, self._fields_dict[self._longevity]]

        fig = plt.figure('Regression')
        ax = fig.subplots()

        ax_leg = ax.scatter(np.arange(lgv.shape[0]), lgv, marker='o', c=dogs_int)
        ax.set_xlabel('Dogs')
        ax.set_ylabel('Longevity')

        plt.show()

    def plot_3d_points(self, model_path='models/reg_knn.joblib'):
        model = RegressorPred().load_model(model_path)
        X = model._X_train
        X = model._model._pipeline.named_steps['select_data'].transform(X)
        X = model._model._pipeline.named_steps['fill_data'].transform(X)
        X = model._model._pipeline.named_steps['poly_feats'].transform(X)
        X = model._model._pipeline.named_steps['one_hot_feats'].transform(X)
        X = model._model._pipeline.named_steps['scale_data'].transform(X)

        y_pred = model.predict_train()
        y_true = model._y_train
        cls = model._y_train_cls

        # tsne = TSNE()
        # X_t = tsne.fit_transform(X)
        pca = PCA(n_components=2)
        X_t = pca.fit_transform(X)

        fig = plt.figure('Regression')
        fig.suptitle('Longevity with PCA')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax1.scatter(X_t[:, 0], X_t[:, 1], y_pred, marker='o', c=cls)
        ax2.scatter(X_t[:, 0], X_t[:, 1], y_true, marker='o', c=cls)
        ax1.set_xlabel('First Feature')
        ax1.set_ylabel('Second Feature')
        ax1.set_zlabel('Longevity(yrs)')
        ax1.set_title('Predictions')
        ax2.set_xlabel('First Feature')
        ax2.set_ylabel('Second Feature')
        ax2.set_title('True Classes')
        ax2.set_zlabel('Longevity(yrs)')

        plt.show()
