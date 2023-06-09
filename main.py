import pandas as pd

import task_1
import task_2
from classification import Classification
from data_handler import load_data, split_data
from regression import Regression

if __name__ == "__main__":
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
