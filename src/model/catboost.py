from typing import Dict, List, Optional

import pandas as pd
from catboost import CatBoostRegressor, Pool


class CatBoostPredictor:
    """
CatBoostPredictor class for fitting and predicting using CatBoostRegressor.

Methods:
- __init__: Initialize the CatBoostPredictor object.
- _check_store_sales_columns: Check if the required columns and data types are
    present in the input DataFrame.
- _split_stores: Split the input DataFrame by 'store_id'.
- _split_items: Split the input DataFrame by 'item_id'.
- _get_feature_names: Get the feature names from the input DataFrame.
- _get_text_columns: Get the text columns from the input DataFrame.
- make_pools: Create CatBoost Pools for each store-item combination in the
    input DataFrame.
- fit: Fit CatBoostRegressor models for each store-item combination in the
    input DataFrame.
- predict: Make predictions using the fitted models for each store-item
    combination in the input DataFrame.

Attributes:
- model_dict: A dictionary to store the fitted CatBoostRegressor models.

Note:
- This class assumes that the input DataFrame has columns 'store_id',
    'item_id', 'date', and the target column for fitting.
- The 'date' column is expected to be of type 'datetime64[ns]'.
- The 'store_id' and 'item_id' columns can be of type 'int64' or 'object'.
- The target column is expected to be of type 'int64'.
- The class uses CatBoostRegressor for fitting and prediction tasks.
"""

    def __init__(self):
        self.model_dict = {}

    def _check_store_sales_columns(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = None,
    ) -> bool:
        """
        Check if the required columns and data types are present in the input
        DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to check.
        - target_col (str, optional): The target column to include in the
            required columns. Default is None.

        Returns:
        - bool: True if all required columns and data types are present,
            False otherwise.

        Raises:
        - ValueError: If any of the required columns are missing or have
            incorrect data types.

        Notes:
        - This method checks for the presence of 'store_id', 'item_id', and
            'date' columns in the DataFrame.
        - If 'target_col' is provided, it is also checked for presence and
            correct data type.
        - 'store_id' and 'item_id' columns can have data types 'int64' or
            'object'.
        - 'date' column is expected to have data type 'datetime64[ns]'.
        - If 'target_col' is provided, it is expected to have data type
            'int64'.
        """
        required_columns = ['store_id', 'item_id', 'date']
        if target_col:
            required_columns.append(target_col)
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'store_id': ['int64', 'object'],
            'item_id': ['int64', 'object'],
            'date': ['datetime64[ns]'],
        }
        if target_col:
            expected_dtypes[target_col] = ['int64']

        for col, expected_types in expected_dtypes.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type not in expected_types:
                    raise ValueError(
                        f"Column '{col}' should have dtype "
                        f"{' or '.join(expected_types)}, "
                        f"but found '{actual_type}'.",
                    )

        return True

    def _split_stores(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the input DataFrame by 'store_id' and return a dictionary where
        keys are store IDs and values are DataFrames without the 'store_id'
        column.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to split by 'store_id'.

        Returns:
        - Dict[str, pd.DataFrame]: A dictionary where keys are store IDs and
            values are DataFrames without the 'store_id' column.

        Notes:
        - This method uses the pandas 'groupby' function to group the DataFrame
            by 'store_id'.
        - Each group is then stored in the dictionary with the store ID as the
            key.
        - The 'store_id' column is dropped from each group before storing in
            the dictionary.
        """
        store_df_dict = {}

        grouped = df.groupby('store_id')
        for s_id, group in grouped:
            store_df_dict[s_id] = group.drop('store_id', axis=1)

        return store_df_dict

    def _split_items(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Split the input DataFrame by 'item_id' and return a dictionary where
        keys are item IDs and values are DataFrames without the 'item_id'
        column.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to split by 'item_id'.

        Returns:
        - Dict[str, pd.DataFrame]: A dictionary where keys are item IDs and
            values are DataFrames without the 'item_id' column.

        Notes:
        - This method uses the pandas 'groupby' function to group the DataFrame
            by 'item_id'.
        - Each group is then stored in the dictionary with the item ID as the
            key.
        - The 'item_id' column is dropped from each group before storing in
            the dictionary.
        """
        item_df_dict = {}

        grouped = df.groupby('item_id')
        for i_id, group in grouped:
            item_df_dict[i_id] = group.drop('item_id', axis=1)

        return item_df_dict

    def _get_feature_names(
            self,
            df: pd.DataFrame,
            to_drop: Optional[List[str]],
    ) -> List[str]:
        """
        Get the feature names from the input DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame from which to extract feature
            names.
        - to_drop (List[str], optional): A list of column names to drop from
            the feature names. Default is None.

        Returns:
        - List[str]: A list of feature names extracted from the input DataFrame
            after dropping the specified columns.

        Notes:
        - If 'to_drop' is provided, the specified columns will be excluded from
            the final list of feature names.
        - The feature names are extracted directly from the columns of the
            input DataFrame.
        """
        feature_names = df.columns
        if to_drop:
            feature_names = feature_names.drop(to_drop, errors='ignore')

        return feature_names.tolist()

    def _get_text_columns(
            self,
            df: pd.DataFrame,
            to_drop: Optional[List[str]],
    ) -> List[str]:
        """
        Get the text columns from the input DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame from which to extract text
            columns.
        - to_drop (List[str], optional): A list of column names to excludefrom
            the extracted text columns. Default is None.

        Returns:
        - List[str]: A list of text columns extracted from the input DataFrame
            after excluding the specified columns.

        Notes:
        - If 'to_drop' is provided, the specified columns will be excluded from
            the final list of text columns.
        - Text columns are identified based on the 'object' data type in the
            input DataFrame.
        """
        text_columns = df.select_dtypes(include=['object']).columns
        if to_drop:
            text_columns = text_columns.drop(to_drop, errors='ignore')

        return text_columns.tolist()

    def make_pools(
            self,
            df: pd.DataFrame,
            target_col: Optional[str] = None,
    ) -> Dict[str, Dict[str, Pool]]:
        """
        Create CatBoost Pools for each store-item combination in the input
            DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing store, item, date,
            and target columns.
        - target_col (str, optional): The target column for fitting. Default is
            None.

        Returns:
        - Dict[str, Dict[str, Pool]]: A dictionary where keys are store IDs and
            values are dictionaries where keys are item IDs and values are
            Pools containing the data for each store-item combination.

        Notes:
        - This method checks the input DataFrame for required columns and data
            types using '_check_store_sales_columns'.
        - It extracts feature names and text columns using '_get_feature_names'
            and '_get_text_columns' methods.
        - The input DataFrame is split by 'store_id' and 'item_id' using
            '_split_stores' and '_split_items' methods.
        - CatBoost Pools are created for each store-item combination with the
            specified features and target column.
        - The resulting dictionary contains Pools and dates for each store-item
            combination.
        """
        if self._check_store_sales_columns(df, target_col):
            feature_names = self._get_feature_names(
                df, ['store_id', 'item_id', 'cnt'],
            )
            cat_features = self._get_text_columns(df, ['store_id', 'item_id'])

            pool_dict = {}

            store_df_dict = self._split_stores(df)
            for store, store_df in store_df_dict.items():
                item_df_dict = self._split_items(store_df)
                for item, item_df in item_df_dict.items():
                    dates = item_df['date'].tolist()

                    if target_col:
                        item_pool = Pool(
                            item_df[feature_names],
                            item_df[target_col],
                            cat_features=cat_features,
                            feature_names=feature_names,
                        )
                    else:
                        item_pool = Pool(
                            item_df[feature_names],
                            cat_features=cat_features,
                            feature_names=feature_names,
                        )
                    pool_dict[store] = {
                        item: {'pool': item_pool, 'dates': dates},
                    }

            return pool_dict

    def fit(self, df: pd.DataFrame, target_col: str) -> None:
        """
        Fit CatBoostRegressor models for each store-item combination in the
        input DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing store, item, date,
            and target columns.
        - target_col (str): The target column for fitting.

        Returns:
        - None

        Notes:
        - This method creates CatBoost Pools for each store-item combination
            using the 'make_pools' method.
        - It fits a CatBoostRegressor model for each store-item combination
            with specified parameters.
        - The fitted models are stored in the 'model_dict' attribute of the
            class.
        - A message is printed for each fitted model showing the store and item
            IDs.
        """
        pool_dict = self.make_pools(df, target_col)
        for store, item_dict in pool_dict.items():
            for item, item_pool in item_dict.items():
                model = CatBoostRegressor(
                    random_state=52,
                    early_stopping_rounds=75,
                    verbose=0,
                    eval_metric='MAE',
                )
                model.fit(item_pool['pool'])
                print(f"Model for store {store} item {item} fitted.")
                self.model_dict[store] = {item: model}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using the fitted models for each store-item
        combination in the input DataFrame.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing store, item, date,
            and target columns.

        Returns:
        - pd.DataFrame: A DataFrame containing the predictions for each
            store-item combination, including store ID, item ID, date, and
            predicted count.

        Notes:
        - This method relies on the 'make_pools' method to create CatBoost
            Pools for each store-item combination.
        - It iterates over each store-item combination, retrieves the
            corresponding model, and makes predictions using the model.
        - The predictions are stored in a DataFrame with columns for store ID,
            item ID, date, and the predicted count.
        - The final DataFrame contains predictions for all store-item
            combinations in the input DataFrame.
        """
        all_forecasts = pd.DataFrame()

        pool_dict = self.make_pools(df)
        for store, item_dict in pool_dict.items():
            for item, item_pool in item_dict.items():
                model = self.model_dict[store][item]
                forecast = model.predict(item_pool['pool'])

                forecast_data = {
                    'store_id': [store] * len(forecast),
                    'item_id': [item] * len(forecast),
                    'date': item_pool['dates'],
                    'cnt_predict': forecast,
                }
                forecast_df = pd.DataFrame(forecast_data)
                all_forecasts = pd.concat([all_forecasts, forecast_df])

        return all_forecasts
