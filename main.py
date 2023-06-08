import task_1
import task_2
from classification import Classification
from data_handler import load_data, split_data
from regression import Regression

if __name__ == "__main__":
    file_path = './data_files/agoda_cancellation_train.csv'

    # Order Cancelation
    # df_cancel = load_data(file_path)
    # df_cancel = task_1.preprocess_data(df_cancel)
    # train_df, test_df, validation_df = split_data(df_cancel)
    # train_df = train_df.drop_duplicates()
    #
    # X_Train = train_df.loc[:, ~train_df.columns.isin(["order_cancelled"])]
    # y_Train = train_df["order_cancelled"]
    # X_Test = test_df.loc[:, ~test_df.columns.isin(["order_cancelled"])]
    # y_Test = test_df["order_cancelled"]
    #
    # Classification().run_all(X_Train, y_Train, X_Test, y_Test)

    # Price
    df_price = load_data(file_path)
    df_price.drop("cancellation_datetime", axis=1, inplace=True)

    df_price = task_2.preprocess_data(df_price)
    train_df, test_df, validation_df = split_data(df_price)
    train_df = train_df.drop_duplicates()

    X_Train = train_df.loc[:, ~train_df.columns.isin(["original_selling_amount"])]
    y_Train = train_df["original_selling_amount"]
    X_Test = test_df.loc[:, ~test_df.columns.isin(["original_selling_amount"])]
    y_Test = test_df["original_selling_amount"]

    Regression().run_all(X_Train, y_Train, X_Test, y_Test)
