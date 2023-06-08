import typing
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import re

from sklearn.preprocessing import OneHotEncoder

_features = {"hotel_star_rating": (0, 5),
             "no_of_adults": (1, 19),
             "no_of_children": (0, 8),
             "no_of_extra_bed": (0, 4),
             "no_of_room": (1, 9)}

_dates = ["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date", "cancellation_datetime"]
_irrelevant_features = ["h_booking_id", "hotel_chain_code", "hotel_brand_code", "request_earlycheckin",
                        "request_airport", "request_twinbeds", "request_largebed", "request_highfloor",
                        "cancellation_policy_code", "hotel_id", "h_customer_id",
                        "request_latecheckin", "request_nonesmoke", "hotel_area_code", ]
_categorial_features = ["hotel_country_code", "accommadation_type_name", "charge_option", "language",
                        "customer_nationality", "guest_nationality_country_name", "origin_country_code",
                        "original_payment_method", "original_payment_type", "original_payment_currency",
                        "is_first_booking", "is_user_logged_in"]


def split_data(X: pd.DataFrame):
    # Splitting the DataFrame into three parts: train, validation, and test
    train_df, temp_df = train_test_split(X, test_size=0.4, random_state=42)
    validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Printing the sizes of the resulting DataFrames
    print("Train set size:", len(train_df))
    print("Validation set size:", len(validation_df))
    print("Test set size:", len(test_df))
    return train_df, test_df, validation_df


def _fill_missings_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    fills missings values by prediction Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem
    """

    return KNNImputer(n_neighbors=2).fit_transform(X)


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    df = pd.read_csv(filename, parse_dates=_dates)

    return df


def add_extra_features(X: pd.DataFrame):
    X['order_canceled'] = np.where(df['cancellation_datetime'].isna(), 0, 1)
    X['duration_days'] = (X['checkin_date'] - X['checkout_date']).dt.days
    X['booked_days_before'] = (X['booking_datetime'] - X['checkin_date']).dt.days
    X['cancel_code_day_one'] = df.apply(lambda row: parse_code_day_one(row['cancellation_policy_code']), axis=1)
    X['cancel_code_return_one'] = df.apply(lambda row: parse_code_return_one(row['cancellation_policy_code']), axis=1)
    X['cancel_code_day_two'] = df.apply(lambda row: parse_code_day_two(row['cancellation_policy_code']), axis=1)
    X['cancel_code_return_two'] = df.apply(lambda row: parse_code_return_two(row['cancellation_policy_code']), axis=1)
    X['parse_code_no_show'] = df.apply(lambda row: parse_code_no_show(row['cancellation_policy_code']), axis=1)


def preprocess_remove_columns_add_dummy(X: pd.DataFrame):
    for feat in _irrelevant_features:
        X.drop(feat, axis=1, inplace=True)
        # print(X[feat])
    for label in _dates:
        X.drop(label, axis=1, inplace=True)
    X = pd.get_dummies(df, prefix=_categorial_features, columns=_categorial_features)
    return X


def parse_code_day_one(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[0] == 'D':
            return float(numeric_values[0])
    except:
        return 0
    return 0


def parse_code_return_one(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[1] == 'P':
            return float(numeric_values[1]) / 100
        elif alphabetic_substrings[1] == 'N':
            return -1 * float(numeric_values[1])
        else:
            return 0
    except:
        return 0


def parse_code_day_two(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[2] == 'D':
            return float(numeric_values[2])
    except:
        return 0
    return 0


def parse_code_return_two(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[3] == 'P':
            return float(numeric_values[1]) / 100
        elif alphabetic_substrings[3] == 'N':
            return -1 * float(numeric_values[1])
        else:
            return 0
    except:
        return 0


def parse_code_no_show(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if len(alphabetic_substrings) % 2 != 0:
            if alphabetic_substrings[-1] == 'P':
                return float(numeric_values[-1]) / 100
            if alphabetic_substrings[-1] == 'N':
                return -1 * float(numeric_values[1])
        return 0
    except:
        return 0


def proccess_dates(df: pd.DataFrame):
    for label in _dates:
        df[f"{label}_dayofyear"] = df[label].dt.dayofyear
        df[f"{label}_year"] = df[label].dt.year


def encode_features(df: pd.DataFrame):
    enc = preprocessing.OneHotEncoder()
    enc.fit(df)
    OneHotEncoder()
    enc.transform([['female', 'from US', 'uses Safari'], ['male', 'from Europe', 'uses Safari']]).toarray()


def preprocess_data(X: pd.DataFrame, y: typing.Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    is_train = y is not None
    if is_train:
        X = X.assign(order_canceled=y)
        X = X.drop_duplicates()

    X = X.drop(_irrelevant_features, axis=1)  # Irrelevant features

    proccess_dates(X)
    X = X.drop(_dates, axis=1)  # Irrelevant features

    for label in X:  # Replaces invalid values with temporary nan value
        X[label] = X[label].mask(~X[label].between(X[label][0], X[label][1], inclusive="both"), np.nan)

    for category in _categorial_features:  # Handles categorial features
        X[category] = X[category].astype('category')
        X = pd.get_dummies(X, prefix=category, columns=[category])

    add_extra_features(X)

    _fill_missings_values(X)
    if not is_train:
        return X

    X = X.reset_index(drop=True)
    post_processed_y = X["y_train"]
    return X.drop("y_train", axis=1), post_processed_y


if __name__ == "__main__":
    file_path = './data_files/agoda_cancellation_train.csv'
    df = load_data(file_path)
    print(df.head())
    print(df.info)
    add_extra_features(df)
    df = preprocess_remove_columns_add_dummy(df)
    # df.nunique
    from Classification import Classification

    train_df, test_df, validation_df = split_data(df)
    X_Train = train_df.loc[:, ~train_df.columns.isin(['order_canceled', ])]
    X_Test = test_df.loc[:, ~test_df.columns.isin(['order_canceled', ])]

    Classification().run_all(X_Train, train_df['order_canceled'], X_Test, test_df['order_canceled'])
    print(df.columns.tolist())
