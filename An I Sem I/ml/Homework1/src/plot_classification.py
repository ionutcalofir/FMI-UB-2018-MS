import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

from plot import Plot
from predicting.classifier_pred import ClassifierPred

class PlotClassification(Plot):
    def __init__(self):
        super().__init__()
        self._class_field = 'Breed Name'
        self._fields_type = {'Weight(g)': 'r',
                             'Height(cm)': 'r',
                             'Energy level': 'c',
                             'Attention Needs': 'c',
                             'Coat Lenght': 'c',
                             'Sex': 'c'} # r - real, c - categorical

    def plot_no_classes(self):
        classes_name = np.unique(self._X[:, self._fields_dict[self._class_field]])

        fig = plt.figure('Classification Classes')
        fig.suptitle('Classes')
        ax = fig.subplots()

        values = []
        ticks = []
        for class_name in classes_name:
            pos_class = self._fields_dict[self._class_field]
            class_data = self._X[self._X[:, pos_class] == class_name]

            values.append(class_data.shape[0])
            ticks.append(class_name)

        ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(ticks)))
        ax.set_xticklabels(ticks)
        ax.set_xlabel('Classes Name')
        ax.set_ylabel('Counts')

        plt.show()

    def plot_2d_points(self, model_path='models/clf_logistic_regression.joblib'):
        model = ClassifierPred().load_model(model_path)
        X = model._X_train
        X = model._model._pipeline.named_steps['select_data'].transform(X)
        X = model._model._pipeline.named_steps['fill_data'].transform(X)
        X = model._model._pipeline.named_steps['poly_feats'].transform(X)
        X = model._model._pipeline.named_steps['one_hot_feats'].transform(X)
        X = model._model._pipeline.named_steps['scale_data'].transform(X)

        y_pred = model.predict_train()
        y_true = model._y_train

        # tsne = TSNE()
        # X_t = tsne.fit_transform(X)
        pca = PCA(n_components=2)
        X_t = pca.fit_transform(X)

        fig = plt.figure('Classification')
        fig.suptitle('Classes with PCA')
        ax = fig.subplots(1, 2).flatten()
        ax[0].scatter(X_t[:, 0], X_t[:, 1], marker='o', c=y_pred)
        ax[1].scatter(X_t[:, 0], X_t[:, 1], marker='o', c=y_true)
        ax[0].set_xlabel('First Feature')
        ax[0].set_ylabel('Second Feature')
        ax[0].set_title('Predictions')
        ax[1].set_xlabel('First Feature')
        ax[1].set_ylabel('Second Feature')
        ax[1].set_title('True Classes')

        plt.show()

    def plot_dataset(self):
        classes_name = np.unique(self._X[:, self._fields_dict[self._class_field]])

        fig = plt.figure('Classification')
        fig.suptitle('Classes')
        ax = fig.subplots(2, 2).flatten()
        for ax_idx, class_name in enumerate(classes_name):
            pos_class = self._fields_dict[self._class_field]
            class_data = self._X[self._X[:, pos_class] == class_name]

            values = []
            ticks = []
            for field, field_type in self._fields_type.items():
                pos_field = self._fields_dict[field]
                if field_type == 'r':
                    value = np.nanmean(class_data[:, pos_field].astype(np.float64))

                    if field == 'Weight(g)':
                        value = value * 1e-3
                        name = str(len(values)) + ': ' + 'Weight(kg) (avg)'
                    elif field == 'Height(cm)':
                        name = str(len(values)) + ': ' + field + ' (avg)'
                    else:
                        name = str(len(values)) + ': ' + field

                    values.append(value)
                    ticks.append(name)
                else:
                    names, counts = np.unique(class_data[:, pos_field], return_counts=True)
                    for name, count in zip(names, counts):
                        values.append(count)
                        ticks.append(str(len(values) - 1) + ': ' + field + ' - ' + name + ' (count)')

            ax_bars = ax[ax_idx].bar(np.arange(len(values)), values)
            ax[ax_idx].set_xticks(np.arange(len(values)))
            ax[ax_idx].set_xlabel(class_name)
            ax[ax_idx].set_ylabel('Values')
            ax[ax_idx].legend(ax_bars, ticks, loc=2, fontsize='xx-small')

        plt.show()
