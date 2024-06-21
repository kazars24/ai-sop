import os

import pandas as pd


class DataLoader:
    """
    Class representing a data loader for loading and preprocessing different
    types of data files.

    Methods:
    - _check_file_existence(file_path: str) -> None: Checks if the file exists
        at the given file path.
    - _load_data(file_path: str) -> pd.DataFrame: Loads data from a file in
        either .csv or .xlsx/.xls format.
    - _preprocess_store_sales(data: pd.DataFrame) -> pd.DataFrame: Preprocesses
        the store sales data.
    - _preprocess_dates(data: pd.DataFrame) -> pd.DataFrame: Preprocesses the
        dates data.
    - _check_store_sales_columns(data: pd.DataFrame) -> bool: Checks if the
        store sales data has the required columns and data types.
    - _check_dates_columns(data: pd.DataFrame) -> bool: Checks if the dates
        data has the required columns and data types.
    - load_store_sales(store_sales_path: str) -> pd.DataFrame: Loads and
        preprocesses the store sales data.
    - load_dates(dates_path: str) -> pd.DataFrame: Loads and preprocesses the
        dates data.
    """

    def __init__(self):
        pass

    def _check_file_existence(self, file_path: str) -> None:
        """
        Checks if the file exists at the given file path.

        Parameters:
        - file_path (str): The path to the file to be checked.

        Raises:
        - FileNotFoundError: If the file does not exist at the specified
            file path.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")

    def _load_data(self, file_path) -> pd.DataFrame:
        """
        Loads data from a file in either .csv or .xlsx/.xls format.

        Parameters:
        - file_path (str): The path to the file to be loaded.

        Returns:
        - pd.DataFrame: The loaded data from the file.

        Raises:
        - FileNotFoundError: If the file does not exist at the specified
            file path.
        - ValueError: If the file format is not supported
            (only .csv or .xlsx/.xls files are supported).
        """
        data = pd.read_csv(file_path)

        return data

    def _preprocess_store_sales(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the store sales data by converting the 'date' column
        to datetime format.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the store
            sales data.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame with the 'date' column
            converted to datetime format.
        """
        data['date'] = pd.to_datetime(data['date'])
        return data

    def _preprocess_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the dates data by filling missing values in 'event_name'
        and 'event_type' columns with 'not_event' and converting the 'date'
        column to datetime format.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the dates data.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame with filled missing values
            and 'date' column converted to datetime format.
        """
        data['event_name'] = data['event_name'].fillna('not_event')
        data['event_type'] = data['event_type'].fillna('not_event')
        data['date'] = pd.to_datetime(data['date'])
        return data

    def _check_store_sales_columns(self, data: pd.DataFrame) -> bool:
        """
        Checks if the provided DataFrame containing store sales data has
        the required columns and data types.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the
            store sales data.

        Returns:
        - bool: True if all required columns are present with expected
            data types, False otherwise.

        Raises:
        - ValueError: If any of the required columns are missing or have
            unexpected data types.
        """
        required_columns = ['store_id', 'item_id', 'date', 'cnt']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'store_id': ['int64', 'object'],
            'item_id': ['int64', 'object'],
            'date': ['object'],
            'cnt': ['int64'],
        }

        for col, expected_types in expected_dtypes.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if actual_type not in expected_types:
                    raise ValueError(
                        f"Column '{col}' should have dtype "
                        f"{' or '.join(expected_types)}, "
                        f"but found '{actual_type}'.",
                    )

        return True

    def _check_dates_columns(self, data: pd.DataFrame) -> bool:
        """
        Checks if the provided DataFrame containing dates data has the required
        columns and data types.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the dates data.

        Returns:
        - bool: True if all required columns are present with expected data
            types, False otherwise.

        Raises:
        - ValueError: If any of the required columns are missing
            or have unexpected data types.
        """
        required_columns = ['date', 'event_name', 'event_type']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'date': ['object'],
            'event_name': ['object'],
            'event_type': ['object'],
        }

        for col, expected_types in expected_dtypes.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if actual_type not in expected_types:
                    raise ValueError(
                        f"Column '{col}' should have dtype "
                        f"{' or '.join(expected_types)}, "
                        f"but found '{actual_type}'.",
                    )

        return True

    def load_store_sales(self, store_sales_path) -> pd.DataFrame:
        """
        Loads and preprocesses the store sales data.

        Parameters:
        - store_sales_path (str): The path to the store sales data file.

        Returns:
        - pd.DataFrame: The preprocessed store sales data as a DataFrame.

        Raises:
        - FileNotFoundError: If the file does not exist at the specified file
            path.
        - ValueError: If the store sales data file is missing required columns
            or has unexpected data types.
        """
        data = self._load_data(store_sales_path)

        if self._check_store_sales_columns(data):
            return self._preprocess_store_sales(data)

    def load_dates(self, dates_path: str) -> pd.DataFrame:
        """
        Loads and preprocesses the dates data.

        Parameters:
        - dates_path (str): The path to the dates data file.

        Returns:
        - pd.DataFrame: The preprocessed dates data as a DataFrame.

        Raises:
        - FileNotFoundError: If the file does not exist at the specified
            file path.
        - ValueError: If the dates data file is missing required columns
            or has unexpected data types.
        """
        data = self._load_data(dates_path)

        if self._check_dates_columns(data):
            return self._preprocess_dates(data)
