# class Regression:
#     def __init__(self):
#         pass
#
#     def linear_regression(self, X_train, y_train, X_test, y_test):
#         from sklearn.linear_model import LinearRegression
#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         return model
#
#     def ridge(self, X_train, y_train, X_test, y_test, alpha=0.5):
#         from sklearn import linear_model
#         model = linear_model.Ridge(alpha=alpha)
#         model.fit(X_train, y_train)
#         return model
#
#     def polynomial_regression(self, X_train, y_train, X_test, y_test, degree=3):
#         from sklearn.preprocessing import PolynomialFeatures
#         from sklearn.linear_model import LinearRegression
#         from sklearn.pipeline import Pipeline
#         model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
#                           ('linear', LinearRegression(fit_intercept=False))])
#         model.fit(X_train, y_train)
#         return model
#
#     def svr(self, X_train, y_train, X_test, y_test):
#         from sklearn import svm
#         model = svm.SVR()
#         model.fit(X_train, y_train)
#         return model
#
#     def decision_tree_regression(self, X_train, y_train, X_test, y_test, max_depth=12):
#         from sklearn.tree import DecisionTreeRegressor
#         model = DecisionTreeRegressor(max_depth=max_depth)
#         model.fit(X_train, y_train)
#         return model
#
#     def random_forest_regression(self, X_train, y_train, X_test, y_test, n_estimators=10, max_features=2,
#                                  max_leaf_nodes=5, random_state=42):
#         from sklearn.ensemble import RandomForestRegressor
#         model = RandomForestRegressor(n_estimators=10, max_features=2, max_leaf_nodes=5, random_state=42)
#         model.fit(X_train, y_train)
#         return model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectFromModel, SelectPercentile
from sklearn.metrics import f1_score, confusion_matrix

from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


class Classification:
    def __init__(self):
        pass

    def logistic_regression(self, X_train_mm, y_train, X_test_mm, y_test):
        logreg = LogisticRegression(max_iter=500).fit(X_train_mm, y_train)
        scores = cross_val_score(logreg, X_train_mm, y_train, cv=5)
        logreg_pred = logreg.predict(X_test_mm)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(logreg.score(X_test_mm, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, logreg_pred)))
        print(confusion_matrix(y_test, logreg_pred))

    def linear_svc(self, X_train_mm, y_train, X_test_mm, y_test):
        svc = LinearSVC().fit(X_train_mm, y_train)
        scores = cross_val_score(svc, X_train_mm, y_train, cv=5)
        svc_pred = svc.predict(X_test_mm)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(svc.score(X_test_mm, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, svc_pred)))
        print(confusion_matrix(y_test, svc_pred))

    def sgd(self, X_train_std, y_train, X_test_std, y_test):
        sgd = SGDClassifier(alpha=0.1).fit(X_train_std, y_train)
        scores = cross_val_score(sgd, X_train_std, y_train, cv=5)
        sgd_pred = sgd.predict(X_test_std)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(sgd.score(X_test_std, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, sgd_pred)))
        print(confusion_matrix(y_test, sgd_pred))

    def ridge(self, X_train, y_train, X_test, y_test, X_test_std, X_train_std):
        rc = RidgeClassifier(alpha=1, normalize=True)
        rc.fit(X_train, y_train)
        scores = cross_val_score(rc, X_train, y_train, cv=5)
        rc_pred = rc.predict(X_test)
        print("Normalized data:")
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(rc.score(X_test, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, rc_pred)))
        print(confusion_matrix(y_test, rc_pred))

        rc = RidgeClassifier(alpha=1)
        rc.fit(X_train_std, y_train)
        scores = cross_val_score(rc, X_train_std, y_train, cv=5)
        rc_pred = rc.predict(X_test_std)
        print("Standard scaled data:")
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(rc.score(X_test_std, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, rc_pred)))
        print(confusion_matrix(y_test, rc_pred))

    def knn(self, X_train, y_train, X_test, y_test):
        training_accuracy = []
        test_accuracy = []
        neighbors_settings = range(1, 6)
        for n_neighbors in neighbors_settings:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            training_accuracy.append(knn.score(X_train, y_train))
            test_accuracy.append(knn.score(X_test, y_test))

        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()

        knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        knn_pred = knn.predict(X_test)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(knn.score(X_test, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, knn_pred)))
        print(confusion_matrix(y_test, knn_pred))

    def decision_tree(self, X_train, y_train, X_test, y_test):
        tree = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
        scores = cross_val_score(tree, X_train, y_train, cv=5)
        tree_pred = tree.predict(X_test)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(tree.score(X_test, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, tree_pred)))
        print(confusion_matrix(y_test, tree_pred))

    def classifier(self, X_train, y_train, X_test, y_test, estimator, param_grid):
        grid_search = GridSearchCV(estimator, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        print("Best score:", grid_search.best_score_)
        print("Test score: {:.3f}".format(grid_search.score(X_test, y_test)))

    def feature_selection(self, model, X_train, y_train, X_test):
        select_features = SelectFromModel(estimator=model, threshold='median')
        select_features.fit(X_train, y_train)
        X_train_select = select_features.transform(X_train)
        X_test_select = select_features.transform(X_test)
        return X_train_select, X_test_select

    def run_model(self, model, model_feature, param_grid, X_train, y_train, X_test, y_test):
        print("Before feature selection:")
        self.classifier(X_train, y_train, X_test, y_test, model, param_grid)
        X_train_select, X_test_select = self.feature_selection(model_feature, X_train, y_train, X_test)
        print("After feature selection")
        self.classifier(X_train_select, y_train, X_test_select, y_test, model, param_grid)

    def random_forest(self, X_train, y_train, X_test, y_test):
        param_grid = {'n_estimators': [50, 75, 100], 'max_depth': [1, 2, 5]}
        self.run_model(RandomForestClassifier(), RandomForestClassifier(n_estimators=50, max_depth=2), param_grid,
                       X_train, y_train, X_test, y_test)

    def gradient_boosted_classifier(self):
        param_grid = {'max_depth': [1, 2, 5], 'learning_rate': [1, 0.1, 0.001]}
        self.run_model(GradientBoostingClassifier(), GradientBoostingClassifier(learning_rate=0.001), param_grid)

    def naive_bayes(self, X_train, y_train, X_test, y_test):
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        scores = cross_val_score(gnb, X_train, y_train, cv=5)
        gnb_pred = gnb.predict(X_test)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(gnb.score(X_test, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, gnb_pred)))
        print(confusion_matrix(y_test, gnb_pred))

    def multi_layer_perceptron(self, X_train_std, y_train, X_test_std, y_test):
        mlp = MLPClassifier(hidden_layer_sizes=[35, 20], alpha=0.001, solver='adam', activation='relu')
        mlp.fit(X_train_std, y_train)
        mlp_pred = mlp.predict(X_test_std)
        print("Train score: {:.3f}".format(mlp.score(X_train_std, y_train)))
        print("Test accuracy: {:.3f}".format(mlp.score(X_test_std, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, mlp_pred)))
        print(confusion_matrix(y_test, mlp_pred))

    def adaboost(self, X_train, y_train, X_test, y_test):
        ada = AdaBoostClassifier()
        ada.fit(X_train, y_train)
        ada_pred = ada.predict(X_test)
        scores = cross_val_score(ada, X_train, y_train, cv=5)
        print("Average cross validation score: {:.3f}".format(scores.mean()))
        print("Test accuracy: {:.3f}".format(ada.score(X_test, y_test)))
        print("F1 score: {:.3f}".format(f1_score(y_test, ada_pred)))
        print(confusion_matrix(y_test, ada_pred))

    def data_scaling(self, X_train, X_test):
        std_scaler = StandardScaler()
        std_scaler.fit(X_train)
        X_train_std = std_scaler.transform(X_train)
        X_test_std = std_scaler.transform(X_test)

        mm_scaler = MinMaxScaler()
        mm_scaler.fit(X_train)
        X_train_mm = mm_scaler.transform(X_train)
        X_test_mm = mm_scaler.transform(X_test)
        return X_train_std, X_test_std, X_train_mm, X_test_mm

    def run_all(self, X_train, y_train, X_test, y_test):
        X_train_std, X_test_std, X_train_mm, X_test_mm = self.data_scaling(X_train, X_test)
        # print("logistic_regression:")
        # self.logistic_regression(X_train_mm, y_train, X_test_mm, y_test)
        # print("linear_svc:")
        # self.linear_svc(X_train_mm, y_train, X_test_mm, y_test)
        # print("sgd:")
        # self.sgd(X_train_std, y_train, X_test_std, y_test)
        # print("ridge:")
        # self.ridge(X_train, y_train, X_test, y_test, X_test_std, X_train_std)
        # print("knn:")
        # self.knn(X_train, y_train, X_test, y_test)
        # print("decision_tree")
        # self.decision_tree( X_train, y_train, X_test, y_test)
        print("random_forest:")
        self.random_forest(X_train, y_train, X_test, y_test)
        print("gradient_boosted_classifier:")
        self.gradient_boosted_classifier(X_train, y_train, X_test, y_test)
        # print("naive_bayes:")
        # self.naive_bayes(X_train, y_train, X_test, y_test)
        # print("multi_layer_perceptron:")
        # self.multi_layer_perceptron(X_train_std, y_train, X_test_std, y_test)
        # print("adaboost:")
        # self.adaboost(X_train, y_train, X_test, y_test)
