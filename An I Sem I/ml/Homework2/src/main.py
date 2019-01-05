import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD

from training.classifier import Classifier
from training.clustering import Clustering

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--clf_grid',
                        action='store_true')
    parser.add_argument('--cluster_grid',
                        action='store_true')
    parser.add_argument('--plot_clf_data',
                        action='store_true')
    parser.add_argument('--plot_cluster_data',
                        action='store_true')
    parser.add_argument('--cluster_data',
                        action='store_true')

    args = parser.parse_args()

    if args.plot_clf_data:
        model = Classifier(tf_idf=True,
                        count_vectorizer=None,
                        dim_reduction=None,
                        clf=SVC(C=1, kernel='linear'))
        model.fit()

        X = model._X_train
        X = model._model._pipeline.named_steps['tf_idf'].transform(X)

        y_pred = model.predict_train()
        y_true = model._y_train

        pca = TruncatedSVD(n_components=2)
        X_pca = pca.fit_transform(X)

        fig = plt.figure('Classification')
        fig.suptitle('Classes with PCA')
        ax = fig.subplots(1, 2).flatten()
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y_pred)
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y_true)
        ax[0].set_xlabel('First Feature')
        ax[0].set_ylabel('Second Feature')
        ax[0].set_title('Predictions')
        ax[1].set_xlabel('First Feature')
        ax[1].set_ylabel('Second Feature')
        ax[1].set_title('True Classes')

        plt.show()
    elif args.plot_cluster_data:
        pca_ob = TruncatedSVD(n_components=200)
        model = Clustering(tf_idf=True,
                           count_vectorizer=False,
                           dim_reduction=pca_ob,
                           clf=KMeans(n_clusters=20, init='k-means++'))

        model.fit()
        labels = model._model._pipeline.named_steps['clf'].labels_
        y_true = model._y_train

        X = model._X_train
        X = model._model._pipeline.named_steps['tf_idf'].transform(X)

        pca = TruncatedSVD(n_components=2)
        X_pca = pca.fit_transform(X)

        fig = plt.figure('Clustering')
        fig.suptitle('Classes with PCA')
        ax = fig.subplots(1, 2).flatten()
        ax[0].scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=labels)
        ax[1].scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y_true)
        ax[0].set_xlabel('First Feature')
        ax[0].set_ylabel('Second Feature')
        ax[0].set_title('Predictions')
        ax[1].set_xlabel('First Feature')
        ax[1].set_ylabel('Second Feature')
        ax[1].set_title('True Classes')

        plt.show()
    elif args.cluster_data:
        pca_ob = TruncatedSVD(n_components=200)
        model = Clustering(tf_idf=True,
                           count_vectorizer=False,
                           dim_reduction=pca_ob,
                           clf=KMeans(n_clusters=20, init='k-means++'))

        model.fit()
        labels = model._model._pipeline.named_steps['clf'].labels_
        y_true = model._y_train

        clusters = [i for i in range(20)]
        d_auth = {}
        d_word = {}
        for c in clusters:
            d_auth[c] = {} 
            d_word[c] = {}
        for i, label in enumerate(labels):
            print(i)
            if label in clusters:
                auth = model._data._class_to_author[int(i / 20)]
                if auth not in d_auth[label]:
                    d_auth[label][auth] = 1
                else:
                    d_auth[label][auth] += 1

                for word in model._X_train[i].split():
                    if word not in d_word[label]:
                        d_word[label][word] = 1
                    else:
                        d_word[label][word] += 1
        words_v = []
        for i in clusters:
            print(i)
            words = []
            for k, v in d_word[i].items():
                words.append((k, v))
            words.sort(key=lambda k: k[1], reverse=True)
            words_v.append(words[:20])

        with open('cluster_data.csv', 'w') as csvfile:
            fieldnames = ['cluster',
                          'authors',
                          'frequent words']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for i in clusters:
                writer.writerow({
                                'cluster': i,
                                'authors': d_auth[i],
                                'frequent words': words_v[i],
                })
    elif args.clf_grid:
        C_v = [0.1, 1, 10]
        kernel_v = ['linear', 'rbf']
        gamma_v = ['auto', 'scale']
        pca_comp_v = [200]
        transformers = [(True, False, True), (True, False, None), (False, True, None)]

        with open('clf_results.csv', 'w') as csvfile:
            fieldnames = ['C',
                          'kernel',
                          'gamma',
                          'tf_idf',
                          'count_vectorizer',
                          'pca',
                          'pca_n_components',
                          'accuracy',
                          'precision',
                          'recall',
                          'f1']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            it = 0
            for tfidf_t, cv_t, pca_t in transformers:
                for C in C_v:
                    for kernel in kernel_v:
                        for gamma in gamma_v:
                            for pca_comp in pca_comp_v:
                                it += 1
                                if pca_t is None:
                                    ob = Classifier(tf_idf=tfidf_t,
                                                    count_vectorizer=cv_t,
                                                    dim_reduction=pca_t,
                                                    clf=SVC(C=C, kernel=kernel, gamma=gamma))
                                else:
                                    pca_ob = TruncatedSVD(n_components=pca_comp)
                                    ob = Classifier(tf_idf=tfidf_t,
                                                    count_vectorizer=cv_t,
                                                    dim_reduction=pca_ob,
                                                    clf=SVC(C=C, kernel=kernel, gamma=gamma))

                                ob.fit()
                                print('C: {0}, kernel: {1}, gamma: {2}, tfidf: {3}, cv: {4}, pca: {5}, pca_comp: {6}, it: {7}, model: {8}' \
                                    .format(C, kernel, gamma, tfidf_t, cv_t, pca_t, pca_comp, it, ob._model))

                                y_pred = ob._model.predict(ob._X_val)
                                acc = accuracy_score(ob._y_val, y_pred)
                                prec = precision_score(ob._y_val, y_pred, average='weighted')
                                rec = recall_score(ob._y_val, y_pred, average='weighted')
                                f1 = f1_score(ob._y_val, y_pred, average='weighted')

                                writer.writerow({
                                    'C': C,
                                    'kernel': kernel,
                                    'gamma': gamma,
                                    'tf_idf': tfidf_t,
                                    'count_vectorizer': cv_t,
                                    'pca': pca_t,
                                    'pca_n_components': pca_comp,
                                    'accuracy': acc,
                                    'precision': prec,
                                    'recall': rec,
                                    'f1': f1
                                })
    elif args.cluster_grid:
        eps_v = [0.7, 0.8, 0.9, 1, 1.1]
        n_clusters_v = [3, 4, 5, 6, 10, 20, 25]
        init_v = ['k-means++', 'random']

        with open('cluster_results.csv', 'w') as csvfile:
            fieldnames = ['name',
                          'eps',
                          'eps_val',
                          'n_clusters',
                          'init',
                          'silhouette']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            print('KMeans')
            for n_clusters in n_clusters_v:
                for init in init_v:
                    pca_ob = TruncatedSVD(n_components=200)
                    ob = Clustering(tf_idf=True,
                                    count_vectorizer=False,
                                    dim_reduction=pca_ob,
                                    clf=KMeans(n_clusters=n_clusters, init=init))

                    ob.fit()
                    labels = ob._model._pipeline.named_steps['clf'].labels_

                    X_tfidf = ob._model._pipeline.named_steps['tf_idf'].transform(ob._X_train)
                    X_trans = ob._model._pipeline.named_steps['dim_reduction'].transform(X_tfidf)

                    s = silhouette_score(X_trans, labels)

                    writer.writerow({
                        'name': 'KMeans',
                        'eps': None,
                        'eps_val': None,
                        'n_clusters': n_clusters,
                        'init': init,
                        'silhouette': s
                    })

            print('AgglomerativeClustering')
            for n_clusters in n_clusters_v:
                pca_ob = TruncatedSVD(n_components=200)
                ob = Clustering(tf_idf=True,
                                count_vectorizer=False,
                                dim_reduction=pca_ob,
                                clf=AgglomerativeClustering(n_clusters=n_clusters))

                ob.fit()
                labels = ob._model._pipeline.named_steps['clf'].labels_

                X_tfidf = ob._model._pipeline.named_steps['tf_idf'].transform(ob._X_train)
                X_trans = ob._model._pipeline.named_steps['dim_reduction'].transform(X_tfidf)

                s = silhouette_score(X_trans, labels)

                writer.writerow({
                    'name': 'AgglomerativeClustering',
                    'eps': None,
                    'eps_val': None,
                    'n_clusters': n_clusters,
                    'init': None,
                    'silhouette': s
                })

            print('DBSCAN')
            for eps in eps_v:
                pca_ob = TruncatedSVD(n_components=200)
                ob = Clustering(tf_idf=True,
                                count_vectorizer=False,
                                dim_reduction=pca_ob,
                                clf=DBSCAN(eps=eps))

                ob.fit()
                labels = ob._model._pipeline.named_steps['clf'].labels_
                unique, counts = np.unique(labels, return_counts=True)
                d = dict(zip(unique, counts))

                writer.writerow({
                    'name': 'DBSCAN',
                    'eps': eps,
                    'eps_val': d,
                    'n_clusters': None,
                    'init': None,
                    'silhouette': None
                })
