from typing import Dict, List, Optional

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.linear_model import LinearRegression


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
        self.deseas_detr_dict = {}
        self.lr_trend_dict = {}
        self.df_agg_dict = {}

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
        required_columns = ['store_id', 'item_id']
        if target_col:
            required_columns.append(target_col)
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in the dataset.")

        expected_dtypes = {
            'store_id': ['int64', 'object'],
            'item_id': ['int64', 'object'],
            # 'date': ['datetime64[ns]'],
        }
        if target_col:
            expected_dtypes[target_col] = ['int64', 'float64']

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
            store_df_dict[s_id] = group  # .drop('store_id', axis=1)

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
            item_df_dict[i_id] = group  # .drop('item_id', axis=1)

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
                df, ['store_id', 'item_id', target_col],
            )
            cat_features = self._get_text_columns(df, ['store_id', 'item_id'])

            pool_dict = {}

            store_df_dict = self._split_stores(df)
            for store, store_df in store_df_dict.items():
                item_df_dict = self._split_items(store_df)
                for item, item_df in item_df_dict.items():
                    item_df = item_df.drop(['store_id', 'item_id'], axis=1)
                    dates = item_df.index.tolist()

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

    def detrending(
            self,
            df,
            trend_model,
            target_col='cnt',
            seas_type='mult',
            dt_col='date',
    ):
        """
        Detrend the input DataFrame using the specified trend model.

        Parameters:
        - df (pd.DataFrame): The input DataFrame to detrend.
        - trend_model: The regression model used for detrending.
        - target_col (str): The target column to detrend. Default is 'cnt'.
        - seas_type (str): The type of detrending to perform, either 'mult' for
            multiplicative or 'add' for additive.
        - dt_col (str): The name of the column to set as the index in the
            output DataFrame. Default is 'date'.

        Returns:
        - pd.DataFrame: A DataFrame with the detrended values, trend
            predictions, and the specified index column.

        Raises:
        - ValueError: If seas_type is not 'mult' or 'add'.

        Notes:
        - The input DataFrame is reset to have a numeric index for detrending.
        - The trend model is fitted on the index values against the target
            column values.
        - The detrended values are calculated based on the specified seas_type.
        - The output DataFrame includes the detrended values, trend
            predictions, and the specified index column.
        """
        df_trend = df.reset_index()
        X = df_trend.index.values
        y = df_trend[target_col].values
        trend_model.fit(X.reshape(-1, 1), y)
        y_pred = trend_model.predict(X.reshape(-1, 1))
        target_col_detrended = target_col + '_detrended'
        if seas_type not in ['add', 'mult']:
            raise ValueError('seas_type should be mult or add')
        elif seas_type == 'mult':
            df_trend[target_col_detrended] = df_trend[target_col] / y_pred
        else:
            df_trend[target_col_detrended] = df_trend[target_col] - y_pred
        df_trend = df_trend.set_index(dt_col)[[target_col_detrended]]
        df_trend['trend'] = y_pred
        return df_trend, trend_model, y_pred

    def predict_trend(self, y_train, df_test, trend_model, target_col='cnt'):
        """
        Predict the trend values using the provided trend model.

        Parameters:
        - y_train (pd.DataFrame): The training data used to predict the trend
            values.
        - df_test (pd.DataFrame): The test data for which trend values are
            predicted.
        - trend_model: The regression model used for predicting the trend
            values.
        - target_col (str): The target column for which trend values are
            predicted. Default is 'cnt'.

        Returns:
        - pd.DataFrame: A DataFrame containing the predicted trend values added
            as a new column 'trend'.

        Notes:
        - The method iterates over the test data and predicts the trend values
            based on the provided trend model.
        - The last value from the training data is used as the initial value
            for prediction.
        - The predicted trend values are added as a new column 'trend' in the
            test data DataFrame.
        """
        last_val = y_train.iloc[-1]
        trend_predicts = []
        for _ in range(df_test.shape[0]):
            val = last_val + trend_model.coef_[0]
            last_val = val
            trend_predicts.append(val)
        df_test['trend'] = trend_predicts

    def restore_trend(
            self,
            y_pred_detr,
            df_test,
            seas_type='mult',
            col='trend',
    ):
        """
        Restore the trend values to the predicted values based on the detrended
        predictions.

        Parameters:
        - y_pred_detr (array-like): The detrended predictions to which the
            trend values are restored.
        - df_test (pd.DataFrame): The DataFrame containing the test data with
            the trend values.
        - seas_type (str): The type of detrending used, either 'mult' for
            multiplicative or 'add' for additive. Default is 'mult'.
        - col (str): The column name in 'df_test' that contains the trend
            values. Default is 'trend'.

        Returns:
        - array-like: An array containing the predicted values with the trend
            restored.

        Raises:
        - ValueError: If 'seas_type' is not 'mult' or 'add'.

        Notes:
        - The method restores the trend values to the detrended predictions
            based on the specified 'seas_type'.
        - If 'seas_type' is 'mult', the trend values are multiplied by the
            detrended predictions.
        - If 'seas_type' is 'add', the trend values are added to the
            detrended predictions.
        """
        if seas_type not in ['add', 'mult']:
            raise ValueError('seas_type should be mult or add')
        elif seas_type == 'mult':
            y_pred = y_pred_detr * df_test[col].values
        else:
            y_pred = y_pred_detr + df_test[col].values
        return y_pred

    def get_seasonal(self, df, seas_type='mult', target_col='cnt'):
        """
        Calculate the seasonal component of the time series data.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the time series
            data.
        - seas_type (str): The type of seasonal component to calculate, either
            'mult' for multiplicative or 'add' for additive. Default is 'mult'.
        - target_col (str): The column in the DataFrame representing the target
            values. Default is 'cnt'.

        Returns:
        - pd.Series: A Series containing the seasonal component values for each
            period.
        - pd.DataFrame: A DataFrame with the aggregated seasonal component
            values for each period.

        Raises:
        - ValueError: If seas_type is not 'mult' or 'add'.

        Notes:
        - The method fits a Linear Regression model to the time series data to
            calculate the trend.
        - The detrended values are computed based on the trend predictions.
        - The data is grouped by quarter to calculate the mean detrended values
            for each period.
        - The seasonal component is derived by normalizing the mean detrended
            values.
        - The final output includes the seasonal component values for each
            period and the aggregated seasonal values.
        """
        lr = LinearRegression()
        dt_col = df.index.name
        df_trend = df.reset_index()
        X = df_trend.index.values
        y = df_trend[target_col].values
        lr.fit(X.reshape(-1, 1), y)
        y_pred = lr.predict(X.reshape(-1, 1))
        if seas_type not in ['add', 'mult']:
            raise ValueError('seas_type should be mult or add')
        elif seas_type == 'mult':
            df_trend['detrended'] = df_trend[target_col] / y_pred
        else:
            df_trend['detrended'] = df_trend[target_col] - y_pred
        df_trend['period'] = df_trend[dt_col].dt.quarter
        df_agg = df_trend\
            .groupby('period')\
            .agg({'detrended': 'mean'})\
            .reset_index()
        mean_mean = df_agg['detrended'].mean()
        if seas_type not in ['add', 'mult']:
            raise ValueError('seas_type should be mult or add')
        elif seas_type == 'mult':
            df_agg['seasonal'] = df_agg['detrended'] / mean_mean
        else:
            df_agg['seasonal'] = df_agg['detrended'] - mean_mean
        df_trend = df_trend.merge(
            df_agg[['period', 'seasonal']], on='period', how='inner',
        )
        return df_trend.set_index(dt_col)['seasonal'], df_agg

    def deseason(self, df, df_seasonal, seas_type='mult', target_col='cnt'):
        """
        Calculate the deseasoned values based on the seasonal component.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the original
            values.
        - df_seasonal (pd.DataFrame): The DataFrame containing the seasonal
            component values.
        - seas_type (str): The type of deseasoning to perform, either 'mult'
            for multiplicative or 'add' for additive. Default is 'mult'.
        - target_col (str): The column in the input DataFrame representing the
            original values. Default is 'cnt'.

        Returns:
        - pd.DataFrame: A DataFrame with the deseasoned values calculated based
            on the seasonal component.

        Raises:
        - ValueError: If seas_type is not 'mult' or 'add'.

        Notes:
        - The method concatenates the original DataFrame with the seasonal
            component DataFrame.
        - If seas_type is 'mult', the deseasoned values are calculated by
            dividing the original values by the seasonal component.
        - If seas_type is 'add', the deseasoned values are calculated by
            subtracting the seasonal component from the original values.
        """
        df_deseas = pd.concat((df, df_seasonal), axis=1)
        if seas_type not in ['add', 'mult']:
            raise ValueError('seas_type should be mult or add')
        elif seas_type == 'mult':
            df_deseas['deseason'] = df_deseas[target_col] / \
                df_deseas['seasonal']
        else:
            df_deseas['detrended'] = df_deseas[target_col] - \
                df_deseas['seasonal']
        return df_deseas

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
        df_deseason_detrended = pd.DataFrame()

        store_df_dict = self._split_stores(df)
        for store, store_df in store_df_dict.items():
            item_df_dict = self._split_items(store_df)
            for item, item_df in item_df_dict.items():
                # dates = item_df.index.tolist()

                features = item_df.drop(target_col, axis=1)
                target = item_df[[target_col]]

                df_seas, df_agg = self.get_seasonal(target, seas_type='mult')

                df_deseas = self.deseason(target, df_seas)

                df_deseas_detr, lr_trend, trend_ex = self.detrending(
                    df_deseas,
                    LinearRegression(),
                    target_col='deseason',
                    seas_type='mult',
                )

                self.deseas_detr_dict[store] = {item: df_deseas_detr}
                self.lr_trend_dict[store] = {item: lr_trend}
                self.df_agg_dict[store] = {item: df_agg}

                features[f"{target_col}_deseason_detrended"] = df_deseas_detr[
                    'deseason_detrended'
                ]
                df_deseason_detrended = pd.concat(
                    [df_deseason_detrended, features],
                )

        pool_dict = self.make_pools(
            df_deseason_detrended, f"{target_col}_deseason_detrended",
        )
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
                df_deseas_detr = self.deseas_detr_dict[store][item]
                lr_trend = self.lr_trend_dict[store][item]
                df_agg = self.df_agg_dict[store][item]
                model = self.model_dict[store][item]

                forecast = model.predict(item_pool['pool'])
                forecast_data = {
                    'date': item_pool['dates'],
                    'cnt_predict': forecast,
                }
                forecast_df = pd.DataFrame(forecast_data)

                self.predict_trend(
                    df_deseas_detr['trend'],
                    forecast_df,
                    lr_trend,
                    target_col='deseason',
                )
                forecast_df['pred'] = forecast_df['cnt_predict'] * \
                    forecast_df['trend']
                forecast_df['quarter'] = forecast_df['date'].dt.quarter

                forecast_df = forecast_df.merge(
                    df_agg[['period', 'seasonal']],
                    left_on='quarter',
                    right_on='period',
                    how='inner',
                )
                forecast_df['pred'] = forecast_df['seasonal'] * \
                    forecast_df['pred']
                y_pred = forecast_df['pred'].values

                forecast_data_final = {
                    'store_id': [store] * len(y_pred),
                    'item_id': [item] * len(y_pred),
                    'date': item_pool['dates'],
                    'cnt_predict': y_pred,
                }
                forecast_df_final = pd.DataFrame(forecast_data_final)
                all_forecasts = pd.concat([all_forecasts, forecast_df_final])

        return all_forecasts
