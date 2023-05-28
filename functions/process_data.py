from datetime import datetime
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)

import csv
import numpy as np
import pandas as pd


def process_data(
    file_path,
    lags=[1, 2, 3],
    rolling_windows=[3, 7, 14],
    target_col="Price",
    datetime_col="Date",
    volume_col="Vol.",
    scaler="MinMaxScaler",
):
    # Load the data
    df = pd.read_csv(file_path)

    # Drop 'Price Movement' column if it exists
    if "Price Movement" in df.columns:
        df = df.drop(columns="Price Movement", axis=1)

    # # Drop 'Change %' column if it exists
    if "Change %" in df.columns:
        df = df.drop(columns="Change %", axis=1)

    # Rename certain columns
    df = df.rename(columns={"0": "Vader Neg", "1": "Vader Neutral", "2": "Vader Pos"})

    # Convert 'Date' column to datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Convert K, M, B into numerical representation in 'Vol.' column
    df[volume_col] = (
        df[volume_col]
        .replace({"K": "*1e3", "M": "*1e6", "B": "*1e9"}, regex=True)
        .map(pd.eval)
        .astype(int)
    )

    # Set Date as the index
    df.set_index(datetime_col, inplace=True)

    # Define numeric columns, excluding the target column
    numeric_cols = (
        df.drop(columns=[target_col]).select_dtypes(include=np.number).columns.tolist()
    )

    # Create lagged features
    for lag in lags:
        for column in numeric_cols:
            df[f"{column}_lag{lag}"] = df[column].shift(lag)

    # Create rolling window features
    for window in rolling_windows:
        for column in numeric_cols:
            df[f"{column}_rolling_mean_{window}"] = df[column].rolling(window).mean()
            df[f"{column}_rolling_min_{window}"] = df[column].rolling(window).min()
            df[f"{column}_rolling_max_{window}"] = df[column].rolling(window).max()

    # Drop rows with any NaN values (introduced when creating lagged and rolling window features)
    df = df.dropna()

    # Move the target column to the end
    df = df[[c for c in df if c not in [target_col]] + [target_col]]

    # Define features and target
    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    # Define the scaler
    if scaler == "StandardScaler":
        scaler = StandardScaler()
    elif scaler == "RobustScaler":
        scaler = RobustScaler()
    else:  # Default to MinMaxScaler
        scaler = MinMaxScaler()

    # Scale features and target
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y_scaled = pd.DataFrame(scaler.fit_transform(y), columns=y.columns, index=y.index)

    return X_scaled, y_scaled, scaler


def split_data(X_scaled, y_scaled, test_ratio):
    # Calculate the indexes for the start and end of the test set
    _len = len(X_scaled)
    test_size = int(_len * test_ratio)
    middle = _len // 2
    test_start = middle - test_size // 2
    test_end = middle + test_size // 2

    # Split the data
    X_scaled_train_val = pd.concat([X_scaled[:test_start], X_scaled[test_end:]])
    y_scaled_train_val = pd.concat([y_scaled[:test_start], y_scaled[test_end:]])
    X_scaled_test = X_scaled[test_start:test_end]
    y_scaled_test = y_scaled[test_start:test_end]

    return X_scaled_train_val, y_scaled_train_val, X_scaled_test, y_scaled_test


def doubleQuotationRemover(input_file_name_v1, delimiter=","):
    """
    Removes double quotation marks from specified columns in a CSV file.

    Args:
        input_file_name_v1 (str): Path to the input CSV file.
        delimiter (str): Delimiter used in the CSV file. Defaults to ','.

    Returns:
        None
    """

    df = pd.read_csv(input_file_name_v1, delimiter=delimiter)

    columns_to_modify = df.columns
    for col in columns_to_modify:
        df[col] = df[col].str.replace('"', "")

    df.to_csv(input_file_name_v1, index=False, sep=delimiter)


def modifyDateFormat(
    input_file_name_v1, date_column_name, date_format="%b %d, %Y", delimiter=","
):
    """
    Modifies the date format of a specific column in a CSV file.

    Args:
        input_file_name_v1 (str): Path to the input CSV file.
        date_column_name (str): Name of the column containing dates to be modified.
        date_format (str): Format of the dates in the input file. Defaults to '%b %d, %Y'.
        delimiter (str): Delimiter used in the CSV file. Defaults to ','.

    Returns:
        None

    Example:
        - Format 1: "%Y-%m-%d" (e.g., "2023-05-14")
        - Format 2: "%m-%d-%Y" (e.g., "05-14-2023")
        - Format 3: "%d-%m-%Y" (e.g., "14-05-2023")
        - Format 4: "%Y/%m/%d" (e.g., "2023/05/14")
        - Format 5: "%m/%d/%Y" (e.g., "05/14/2023")
        - Format 6: "%d/%m/%Y" (e.g., "14/05/2023")
        - Format 7: "%Y.%m.%d" (e.g., "2023.05.14")
        - Format 8: "%m.%d.%Y" (e.g., "05.14.2023")
        - Format 9: "%d.%m.%Y" (e.g., "14.05.2023")
        - Format 10: "%Y %m %d" (e.g., "2023 05 14")
        - Format 11: "%m %d %Y" (e.g., "05 14 2023")
        - Format 12: "%d %m %Y" (e.g., "14 05 2023")
        - Format 13: "%b %d, %Y" (e.g., "May 14, 2023")
    """

    df = []

    with open(input_file_name_v1, "r", encoding="ISO-8859-1") as file:
        reader = csv.reader(file, delimiter=delimiter)
        headers = next(reader)
        dateColumnIndex = headers.index(date_column_name)

        for row in reader:
            row[dateColumnIndex] = datetime.strptime(
                row[dateColumnIndex], date_format
            ).strftime("%m/%d/%y")
            df.append(dict(zip(headers, row)))

    with open(input_file_name_v1, "w", newline="", encoding="ISO-8859-1") as file:
        writer = csv.DictWriter(file, fieldnames=headers, delimiter=delimiter)
        writer.writeheader()

        for row in df:
            writer.writerow(row)


def combineData(
    output_file_name="combined_data",
    input_file_name_v1="bitcoin_price.csv",
    input_file_name_v2="headlines_sentiment.csv",
):
    """
    Combines bitcoin price data and sentiment data into a single CSV file.

    Args:
        output_file_name (str): Path to the output CSV file. Defaults to 'data\\combined_data.csv'.
        input_file_name_v1 (str): Path to the bitcoin price data CSV file. Defaults to 'data\\bitcoin_price.csv'.
        input_file_name_v2 (str): Path to the sentiment data CSV file. Defaults to 'data\\headlines_sentiment.csv'.

    Returns:
        None
    """

    output_file_name = f"data/{output_file_name}.csv"

    bitcoin_price = pd.read_csv(input_file_name_v1, delimiter=",")

    bitcoin_price["Price"] = (
        bitcoin_price["Price"].str.replace('"', "").str.replace(",", "").astype(float)
    )
    bitcoin_price["Open"] = (
        bitcoin_price["Open"].str.replace('"', "").str.replace(",", "").astype(float)
    )
    bitcoin_price["High"] = (
        bitcoin_price["High"].str.replace('"', "").str.replace(",", "").astype(float)
    )
    bitcoin_price["Low"] = (
        bitcoin_price["Low"].str.replace('"', "").str.replace(",", "").astype(float)
    )
    bitcoin_price["Change %"] = (
        bitcoin_price["Change %"].str.replace("%", "").astype(float) / 100
    )

    sentiment_data = pd.read_csv(input_file_name_v2, delimiter=",")

    bitcoin_price["Date"] = pd.to_datetime(bitcoin_price["Date"], format="%m/%d/%y")
    sentiment_data["Date"] = pd.to_datetime(sentiment_data["Date"], format="%m/%d/%y")

    merged_data = bitcoin_price.merge(sentiment_data, on="Date", how="left")
    merged_data["Price Movement"] = (merged_data["Change %"] > 0).astype(int)

    merged_data.to_csv(output_file_name, index=False)
