import argparse
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from dataset import Dataset
from plot import Plot
from plot_classification import PlotClassification
from plot_regression import PlotRegression
from training.classifier import Classifier
from training.regressor import Regressor
from predicting.classifier_pred import ClassifierPred
from predicting.regressor_pred import RegressorPred

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='if true, train a model',
                        action='store_true')
    parser.add_argument('--predict', help='if true, predict',
                        action='store_true')
    parser.add_argument('--logistic_regression', help='if true, train a logistic regression model',
                        action='store_true')
    parser.add_argument('--knn_regressor', help='if true, train a knn model for regression',
                        action='store_true')
    parser.add_argument('--grid_search', help='if true, perform a grid search',
                        action='store_true')
    parser.add_argument('--plot', help='plotting',
                        action='store_true')
    parser.add_argument('--plot_classification', help='plotting',
                        action='store_true')
    parser.add_argument('--plot_regression', help='plotting',
                        action='store_true')

    args = parser.parse_args()

    if args.plot:
        if args.plot_classification:
            pl = PlotClassification()
            pl.plot_dataset()
            pl.plot_no_classes()
            pl.plot_2d_points()
        elif args.plot_regression:
            pl = PlotRegression()
            pl.plot_dataset()
            pl.plot_3d_points()
        else:
            pl = Plot()
            pl.plot_missing_values()
    elif args.train:
        if args.logistic_regression:
            ob_clf = Classifier(clf=LogisticRegression(class_weight='balanced',
                                                       multi_class='multinomial',
                                                       solver='lbfgs',
                                                       C=30),
                                data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
            ob_clf.fit()
            ob_clf.save_model('clf_logistic_regression')
        elif args.knn_regressor:
            ob_reg = Regressor(clf=KNeighborsRegressor(n_neighbors=5,
                                                       leaf_size=5,
                                                       weights='uniform',
                                                       p=1),
                               data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
            ob_reg.fit()
            ob_reg.save_model('reg_knn')
    elif args.predict:
        if args.logistic_regression:
            model = ClassifierPred().load_model('models/clf_logistic_regression.joblib')
            model.predict_test(verbose=True)
        elif args.knn_regressor:
            model = ClassifierPred().load_model('models/reg_knn.joblib')
            model.predict_test(verbose=True)
    elif args.grid_search:
        ob_clf = Classifier(clf=LogisticRegression(class_weight='balanced',
                                                   multi_class='multinomial',
                                                   solver='lbfgs'),
                            data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_clf.grid_search(name='grid_search/clf_logistic_regression',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
                           })

        ob_clf = Classifier(clf=RandomForestClassifier(class_weight='balanced'),
                            data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_clf.grid_search(name='grid_search/clf_random_forest',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__n_estimators': [5, 10],
                            'clf__min_samples_split': [2, 5],
                            'clf__min_samples_leaf': [2]
                           })

        ob_clf = Classifier(clf=KNeighborsClassifier(),
                            data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_clf.grid_search(name='grid_search/clf_knn',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__n_neighbors': [5, 10],
                            'clf__weights': ['uniform', 'distance'],
                            'clf__leaf_size': [15, 30],
                            'clf__p': [1, 2]
                           })

        ob_reg = Regressor(clf=Ridge(),
                           data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_reg.grid_search(name='grid_search/reg_ridge',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__alpha': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
                           })

        ob_reg = Regressor(clf=Lasso(),
                           data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_reg.grid_search(name='grid_search/reg_lasso',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__alpha': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
                           })

        ob_reg = Regressor(clf=KNeighborsRegressor(),
                           data_path='data/Calofir_A_Petrișor_Ionuț_train.csv')
        ob_reg.grid_search(name='grid_search/reg_knn',
                           param_grid = {
                            'poly_feats__grade': [1, 2, 3],
                            'clf__n_neighbors': [5, 10],
                            'clf__weights': ['uniform', 'distance'],
                            'clf__leaf_size': [15, 30],
                            'clf__p': [1, 2]
                           })
