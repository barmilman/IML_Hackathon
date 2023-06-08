import pandas as pd
from sklearn.model_selection import train_test_split

_dates = ["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date", "cancellation_datetime"]

def split_data(X: pd.DataFrame):
    # Splitting the DataFrame into three parts: train, validation, and test
    train_df, temp_df = train_test_split(X, test_size=0.4, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Printing the sizes of the resulting DataFrames
    print("Train set size:", len(train_df))
    print("Validation set size:", len(validation_df))
    print("Test set size:", len(test_df))
    return train_df, test_df, validation_df


def load_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename, parse_dates=_dates)