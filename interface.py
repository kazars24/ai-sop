from datetime import timedelta

import plotly.express as px
import streamlit as st

from src.data.loading import DataLoader
from src.data.split import train_test_split_by_date
from src.features.feature_engineering import FeaturesMaker
from src.model.catboost import CatBoostPredictor

# Заголовок приложения
st.title("AI S&OP")

with st.sidebar:
    # Блок выбора модели для прогнозирования
    st.subheader("Выбор модели для прогнозирования")
    model_choice = st.selectbox(
        "Выберите модель",
        ["Gradient Boosting", "Exponential Smoothing", "Prophet"],
    )

    # Блок загрузки данных
    st.subheader("Загрузка данных")
    store_sales_path = st.file_uploader(
        "Загрузить данные по продажам", type=["csv"],
    )
    store_sales_dates_path = st.file_uploader(
        "Загрузить данные по датам", type=["csv"],
    )
    forecast_period = st.number_input(
        "Период прогнозирования", min_value=1, step=1,
    )


# Загрузка и отображение данных
if store_sales_path is not None and store_sales_dates_path is not None:
    dl = DataLoader()
    store_sales = dl.load_store_sales(store_sales_path)
    store_sales_dates = dl.load_dates(store_sales_dates_path)

    fm = FeaturesMaker()
    dataset = fm.make_features(
        store_sales, 'cnt', forecast_period, 30, store_sales_dates,
    )

    max_date = dataset.index.max()
    delta = timedelta(days=forecast_period)
    new_date = max_date - delta
    new_date_str = new_date.strftime('%Y-%m-%d')

    # Подготовка данных для моделей
    train_df, test_df = train_test_split_by_date(dataset, 'date', new_date_str)

    # Применение выбранной модели для прогнозирования
    st.subheader("Прогнозирование")
    if st.button("Начать прогнозирование"):
        if model_choice == "Prophet":
            pass

        elif model_choice == "Exponential Smoothing":
            pass

        elif model_choice == "Gradient Boosting":
            cb = CatBoostPredictor()
            cb.fit(train_df, 'cnt')
            preds = cb.predict(test_df)

        # Отображение интерактивного графика
        for store in preds['store_id'].unique():
            store_pred_df = preds[preds['store_id'] == store]

            st.subheader("Интерактивный график")
            forecast_fig = px.line(
                store_pred_df, x='date', y='cnt_predict', title="Прогноз",
                labels={'date': 'Дата', 'cnt_predict': 'Прогноз'},
            )
            st.plotly_chart(forecast_fig)

        # Кнопка загрузки прогнозов
        st.subheader("Скачать прогнозы")
        if st.button("Скачать прогнозы"):
            forecast_download = preds.to_csv().encode('utf-8')
            st.download_button(
                label="Скачать как CSV",
                data=forecast_download,
                file_name='forecast.csv',
                mime='text/csv',
            )
