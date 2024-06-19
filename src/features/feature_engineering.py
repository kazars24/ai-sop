from typing import Optional

import pandas as pd


class FeaturesMaker:
    """
    Class to create features for regression analysis based on store sales
    and dates data.

    Methods:
    - _check_store_sales_columns: Check if the required columns and data types
        are present in the store sales data.
    - _check_dates_columns: Check if the required columns and data types are
        present in the dates data.
    - make_features_regression: Create lag features for regression analysis
        based on the target column.
    - make_features_dates: Extract date-related features from the date column.
    - make_features: Generate all features by merging lag features with
        date-related features.
    """

    def __init__(self):
        pass

    def _check_store_sales_columns(self, data: pd.DataFrame) -> bool:
        """
        Check if the required columns and data types are present in the
        store sales data.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing store sales data.

        Returns:
        - bool: True if all required columns and data types are present,
            False otherwise.

        Raises:
        - ValueError: If any of the required columns are missing or have
            incorrect data types.
        """
        required_columns = ['store_id', 'item_id', 'date', 'cnt']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'store_id': ['int64', 'object'],
            'item_id': ['int64', 'object'],
            'date': ['datetime64[ns]'],
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
        Check if the required columns and data types are present in the
        dates data.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing dates data.

        Returns:
        - bool: True if all required columns and data types are present,
            False otherwise.

        Raises:
        - ValueError: If any of the required columns are missing or have
            incorrect data types.
        """
        required_columns = ['date', 'event_name', 'event_type']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'date': ['datetime64[ns]'],
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

    def make_features_regression(
            self,
            data: pd.DataFrame,
            target_col: str,
            max_pred_period: int,
            num_lags: int,
    ) -> pd.DataFrame:
        """
        Create lag features for regression analysis based on the target column.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the data.
        - target_col (str): The name of the target column for which lag
            features are to be created.
        - max_pred_period (int): The maximum prediction period for lag
            features.
        - num_lags (int): The number of lag features to create.

        Returns:
        - pd.DataFrame: The input DataFrame with lag features added as new
            columns.

        """
        feature_lags = [max_pred_period + i for i in range(1, num_lags + 1)]
        for lag in feature_lags:
            data.loc[:, f'{target_col}_lag_{lag}'] = data[target_col].shift(
                periods=lag,
                fill_value=0,
            ).values

        return data

    def make_features_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create date-related features from the date column.

        Parameters:
        - data (pd.DataFrame): The input DataFrame containing the date column.

        Returns:
        - pd.DataFrame: The input DataFrame with additional columns for
            weekday, day, month, and year extracted from the date column.
        """
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])

        data['weekday'] = data['date'].dt.weekday
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year

        return data

    def make_features(
        self, store_sales: pd.DataFrame,
        target_col: str, max_pred_period: int, num_lags: int,
        store_sales_dates: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create features by merging lag features with date-related features.

        Parameters:
        - store_sales (pd.DataFrame): The input DataFrame containing store
            sales data.
        - target_col (str): The name of the target column for which lag
            features are to be created.
        - max_pred_period (int): The maximum prediction period for lag
            features.
        - num_lags (int): The number of lag features to create.
        - store_sales_dates (Optional[pd.DataFrame]): The input DataFrame
            containing dates data. Default is None.

        Returns:
        - pd.DataFrame: The input DataFrame with lag features and date-related
            features merged.

        """
        if self._check_store_sales_columns(store_sales):
            store_sales_lags = self.make_features_regression(
                store_sales, target_col, max_pred_period, num_lags,
            )

            if store_sales_dates is not None:
                if self._check_dates_columns(store_sales_dates):
                    dates_features = self.make_features_dates(
                        store_sales_dates,
                    )
                    all_features = store_sales_lags.merge(
                        dates_features, 'left', 'date',
                    )
                    return all_features
            else:
                return store_sales_lags
