import pandas as pd
from sklearn.model_selection import train_test_split

_dates = ["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date"]


class DataHandler:
    def __init__(self):
        pass

    def split_data_old(self, X: pd.DataFrame):
        # Splitting the DataFrame into three parts: train, validation, and test
        train_df, temp_df = train_test_split(X, test_size=0.4, random_state=42)
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Printing the sizes of the resulting DataFrames
        print("Train set size:", len(train_df))
        print("Validation set size:", len(validation_df))
        print("Test set size:", len(test_df))
        return train_df, test_df, validation_df

    def split_data(self, X: pd.DataFrame):
        # Splitting the DataFrame into three parts: train, validation, and test
        train_df, test_df = train_test_split(X)

        # Printing the sizes of the resulting DataFrames
        print("Train set size:", len(train_df))
        print("Test set size:", len(test_df))
        return train_df, test_df

    def load_data(self, filename: str, include_cancellation) -> pd.DataFrame:
        dates = _dates.copy()
        if include_cancellation:
            dates.append("cancellation_datetime")

        return pd.read_csv(filename, parse_dates=dates)
