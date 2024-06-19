from typing import Tuple

import pandas as pd


def train_test_split_by_date(
        df: pd.DataFrame,
        date_column: str,
        split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to split a DataFrame into training and testing sets based on a
    specified date.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data to split.
        date_column (str): The name of the column in the DataFrame that
            contains the dates.
        split_date (str): The date used to split the data into training and
            testing sets.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and
            testing DataFrames.

    Example:
        train_df, test_df = train_test_split_by_date(df,
                                                     'date',
                                                     '2022-01-01')
    """
    split_date = pd.to_datetime(split_date)

    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])

    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

    grouped = df.groupby('store_id')
    for _, group in grouped:
        train_part = group[group[date_column] < split_date]
        test_part = group[group[date_column] >= split_date]

        train_df = pd.concat([train_df, train_part])
        test_df = pd.concat([test_df, test_part])

    return train_df, test_df
