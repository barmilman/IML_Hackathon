import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class Regression:
    def __init__(self):
        pass

    def linear_regression(self, X_train, y_train, X_test, y_test):
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    def ridge(self, X_train, y_train, X_test, y_test, alpha=0.5):
        from sklearn import linear_model

        model = linear_model.Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        print(f"Train MSE: {np.sqrt(mean_squared_error(y_train, pred_train))}")
        print(f"Train R2_Score: {r2_score(y_train, pred_train)}")

        pred_test = model.predict(X_test)
        print(f"Test MSE: {np.sqrt(mean_squared_error(y_test, pred_test))}")
        print(f"Test R2_Score: {r2_score(y_test, pred_test)}")

        return model

    def lasso(self, X_train, y_train, X_test, y_test):
        from sklearn import linear_model

        model = linear_model.Lasso(alpha=0.01)
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        print(f"Train MSE: {np.sqrt(mean_squared_error(y_train, pred_train))}")
        print(f"Train R2_Score: {r2_score(y_train, pred_train)}")

        pred_test = model.predict(X_test)
        print(f"Test MSE: {np.sqrt(mean_squared_error(y_test, pred_test))}")
        print(f"Test R2_Score: {r2_score(y_test, pred_test)}")

        return model

    def polynomial_regression(self, X_train, y_train, X_test, y_test, degree=3):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline

        model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                          ('linear', LinearRegression(fit_intercept=False))])
        model.fit(X_train, y_train)
        return model

    def svr(self, X_train, y_train, X_test, y_test):
        from sklearn import svm

        model = svm.SVR()
        model.fit(X_train, y_train)
        return model

    def decision_tree_regression(self, X_train, y_train, X_test, y_test, max_depth=12):
        from sklearn.tree import DecisionTreeRegressor

        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        return model

    def random_forest_regression(self, X_train, y_train, X_test, y_test, n_estimators=10, max_features=2,
                                 max_leaf_nodes=5, random_state=42):
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, max_features=2, max_leaf_nodes=5, random_state=42)
        model.fit(X_train, y_train)

        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)

        return pred_test

    def mlp_regressor(self, X_train, y_train, X_test, y_test):
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(random_state=1, max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Test MSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
        print(f"Test R2_Score: {r2_score(y_test, y_pred)}")

        return model

    def gbr(self, X_train, y_train, X_test, y_test=None):
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return y_pred

    def data_scalling(self, X_Train, y_Train, X_Test, y_Test):
        transformer_x = StandardScaler()
        transformer_x.fit(X_Train)
        transformer_y = StandardScaler()
        transformer_y.fit(y_Train.to_numpy().reshape(-1, 1))
        X_rtrain = transformer_x.transform(X_Train)
        y_rtrain = transformer_y.transform(y_Train.to_numpy().reshape(-1, 1))
        X_rtest = transformer_x.transform(X_Test)
        y_rtest = transformer_y.transform(y_Test.to_numpy().reshape(-1, 1))

        # y_pred_unsacled = transformer_y.inverse_transform(y_pred.reshape(-1, 1))
        # y_Test_unsacled = transformer_y.inverse_transform(y_rtest)

        # return y_pred_unsacled, y_Test_unsacled

    def run_all(self, X_Train, y_Train, X_Test, y_Test=None):
        return self.gbr(X_Train, y_Train, X_Test, y_Test)
