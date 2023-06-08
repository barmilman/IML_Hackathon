import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
import re

_countries_to_keep = ["KR", "MY", "TH", "TW", "HK", "ID", "PK", "RU"]
_guest_nationality_countries_to_keep = ["South Korea", "Malaysia", "Taiwan", "Thailand", "Hong Kong", "Indonesia"]
_hotel_ids_to_keep = [6452, 3098648, 3080111, 1629394]
_problematic_cust = [3403039646291800000, 989627699560000000, 6096800853640020093, 6754714678033050058,
                     8370707232058280048]
_hotel_city_code_to_keep = [220, 1403, 142, 2249, 2224, 2797]

_features = {"hotel_star_rating": (0, 5),
             "no_of_adults": (1, 19),
             "no_of_children": (0, 8),
             "no_of_extra_bed": (0, 4),
             "no_of_room": (1, 9)}

# "request_latecheckin", "request_nonesmoke", "request_earlycheckin", "request_highfloor",
_dates = ["booking_datetime", "checkin_date", "checkout_date", "hotel_live_date", "cancellation_datetime"]
_irrelevant_features = ["h_booking_id", "hotel_chain_code", "hotel_brand_code", "hotel_area_code"]
_categorial_features = ["hotel_country_code", "accommadation_type_name", "charge_option", "language",
                        "customer_nationality", "guest_nationality_country_name", "origin_country_code",
                        "original_payment_method", "original_payment_type", "original_payment_currency",
                        "is_first_booking", "is_user_logged_in", "hotel_id", "h_customer_id", "hotel_city_code"]


def fill_missings_values(df: pd.DataFrame) -> pd.DataFrame:
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    idf = pd.DataFrame(imp.fit_transform(df))
    idf.columns = df.columns
    idf.index = df.index

    return idf


def add_extra_features(X: pd.DataFrame):
    X['order_cancelled'] = np.where(X['cancellation_datetime'].isna(), 0, 1)
    X['duration_days'] = (X['checkout_date'] - X['checkin_date']).dt.days
    X['booked_days_before'] = (X['booking_datetime'] - X['checkin_date']).dt.days
    X['cancel_code_day_one'] = X.apply(lambda row: parse_code_day_one(row['cancellation_policy_code']), axis=1)
    X['cancel_code_return_one'] = X.apply(
        lambda row: parse_code_return_one(row['cancellation_policy_code'], row['duration_days']), axis=1)
    X['cancel_code_day_two'] = X.apply(lambda row: parse_code_day_two(row['cancellation_policy_code']), axis=1)
    X['cancel_code_return_two'] = X.apply(
        lambda row: parse_code_return_two(row['cancellation_policy_code'], row['duration_days']), axis=1)
    X['parse_code_no_show'] = X.apply(
        lambda row: parse_code_no_show(row['cancellation_policy_code'], row['duration_days']), axis=1)


def parse_code_day_one(row):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[0] == 'D':
            return float(numeric_values[0])
    except:
        return 0
    return 0


def parse_code_return_one(row, days):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[1] == 'P':
            return float(numeric_values[1]) / 100
        elif alphabetic_substrings[1] == 'N':
            return float(numeric_values[1]) / days
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


def parse_code_return_two(row, days):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if alphabetic_substrings[3] == 'P':
            return float(numeric_values[1]) / 100
        elif alphabetic_substrings[3] == 'N':
            return float(numeric_values[1]) / days
        else:
            return 0
    except:
        return 0


def parse_code_no_show(row, days):
    numeric_values = re.findall(r'\d+', row)
    alphabetic_substrings = re.findall(r'[a-zA-Z]+', row)
    try:
        if len(alphabetic_substrings) % 2 != 0:
            if alphabetic_substrings[-1] == 'P':
                return float(numeric_values[-1]) / 100
            if alphabetic_substrings[-1] == 'N':
                return float(numeric_values[1]) / days
        return 0
    except:
        return 0


def proccess_dates(df: pd.DataFrame):
    for label in _dates:
        if label == "cancellation_datetime":
            continue

        df[f"{label}_dayofyear"] = df[label].dt.dayofyear
        df[f"{label}_year"] = df[label].dt.year


def mika_proccess(X):
    X.loc[~X['accommadation_type_name'].isin(['Hotel', 'Apartment']), 'accommadation_type_name'] = np.nan
    X.loc[~X['guest_nationality_country_name'].isin(
        _guest_nationality_countries_to_keep), 'guest_nationality_country_name'] = np.nan
    X.loc[~X['origin_country_code'].isin(_countries_to_keep), 'origin_country_code'] = np.nan
    X.loc[~X['hotel_id'].isin(_hotel_ids_to_keep), 'hotel_id'] = np.nan
    X.loc[~X['h_customer_id'].isin(_problematic_cust), 'h_customer_id'] = 0
    X.loc[X['h_customer_id'].isin(_problematic_cust), 'h_customer_id'] = 1
    X.loc[~X['hotel_city_code'].isin(_hotel_city_code_to_keep), 'hotel_city_code'] = np.nan


def preprocess_data(X: pd.DataFrame):
    proccess_dates(X)
    add_extra_features(X)

    X.drop(_dates, axis=1, inplace=True)
    X.drop(_irrelevant_features, axis=1, inplace=True)
    X.drop("cancellation_policy_code", axis=1, inplace=True)

    X.replace(["UNKNOWN"], np.nan, inplace=True)
    # mika_proccess(X)
    # for label in X:  # Replaces invalid values with temporary nan value
    #     X[label] = X[label].mask(~X[label].between(X[label][0], X[label][1], inclusive="both"), np.nan)

    for category in _categorial_features:  # Handles categorial features
        X[category] = X[category].astype('category')
        X = pd.get_dummies(X, prefix=category, columns=[category])

    return fill_missings_values(X)
