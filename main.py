import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from code import task_1, task_2
from code.classification import Classification
from code.data_handler import load_data, split_data
from code.regression import Regression


def temp():
    file_path = './data_files/agoda_cancellation_train.csv'

    # Order Cancelation
    # df_cancel = load_data(file_path, True)
    # df_cancel = task_1.preprocess_data(df_cancel, True)
    # train_df, test_df = split_data(df_cancel)
    # train_df = train_df.drop_duplicates()
    #
    # X_Train = train_df.loc[:, ~train_df.columns.isin(["order_cancelled"])]
    # y_Train = train_df["order_cancelled"]
    # X_Test = test_df.loc[:, ~test_df.columns.isin(["order_cancelled"])]
    # y_Test = test_df["order_cancelled"]
    #
    # Classification().run_all(X_Train, y_Train, X_Test, y_Test)

    # Price
    df_price = load_data(file_path, True)

    df_price = task_2.preprocess_data(df_price, True)
    train_df, test_df = split_data(df_price)
    train_df = train_df.drop_duplicates()

    X_Train = train_df.loc[:, ~train_df.columns.isin(["original_selling_amount"])]
    y_Train = train_df["original_selling_amount"]
    X_Test = test_df.loc[:, ~test_df.columns.isin(["original_selling_amount"])]
    y_Test = test_df["original_selling_amount"]

    y_pred = Regression().run_all(X_Train, y_Train, X_Test, y_Test)

    print(f"Test MSE: {np.sqrt(mean_squared_error(y_Test, y_pred))}")
    print(f"Test R2_Score: {r2_score(y_Test, y_pred)}")


if __name__ == "__main__":
    temp()

    train_path = './data_files/agoda_cancellation_train.csv'

    # Order Cancelation
    test_path = './data_files/Agoda_Test_1.csv'

    train_df = load_data(train_path, True)
    train_df = task_1.preprocess_data(train_df, True)
    train_df.drop("h_booking_id", axis=1, inplace=True)
    train_df = train_df.drop_duplicates()
    X_Train = train_df.loc[:, ~train_df.columns.isin(["order_cancelled"])]
    y_Train = train_df["order_cancelled"]

    test_df = load_data(test_path, False)
    test_df = task_1.preprocess_data(test_df, False)
    test_booking_ids = test_df["h_booking_id"]
    test_booking_ids.name = "id"
    test_df.drop("h_booking_id", axis=1, inplace=True)
    test_df = test_df.reindex(columns=X_Train.columns, fill_value=0)
    y_pred = pd.Series(Classification().run_all(X_Train, y_Train, test_df))
    y_pred.name = "cancellation"
    df1 = pd.concat([test_booking_ids, y_pred], axis=1)
    df1.to_csv("agoda_cancellation_prediction.csv")

    # Price
    test_path = './data_files/Agoda_Test_2.csv'

    train_df = load_data(train_path, True)
    train_df = task_2.preprocess_data(train_df, True)
    train_df.drop("h_booking_id", axis=1, inplace=True)
    train_df = train_df.drop_duplicates()
    X_Train = train_df.loc[:, ~train_df.columns.isin(["order_cancelled"])]
    y_Train = train_df["order_cancelled"]
    test_df = load_data(test_path, False)
    test_df = task_2.preprocess_data(test_df, False)
    test_df.drop("h_booking_id", axis=1, inplace=True)
    test_df = test_df.reindex(columns=X_Train.columns, fill_value=0)
    order_cancellations = Classification().run_all(X_Train, y_Train, test_df)

    X_Train = train_df.loc[:, ~train_df.columns.isin(["original_selling_amount"])]
    y_Train = train_df["original_selling_amount"]
    test_df = load_data(test_path, False)
    test_df["order_cancelled"] = order_cancellations
    test_df = task_2.preprocess_data(test_df, False)
    test_booking_ids = test_df["h_booking_id"]
    test_booking_ids.name = "id"
    test_df.drop("h_booking_id", axis=1, inplace=True)
    test_df = test_df.reindex(columns=X_Train.columns, fill_value=0)
    y_pred = pd.Series(Regression().run_all(X_Train, y_Train, test_df))
    y_pred[y_pred < 1] = 1
    y_pred.reindex(test_df.index)
    y_pred[test_df["order_cancelled"] == 0] = -1
    y_pred.name = "predicted_selling_amount"
    df2 = pd.concat([test_booking_ids, y_pred], axis=1)
    df2.to_csv("agoda_cost_cancellation.csv")
