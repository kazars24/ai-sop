import os
from datetime import timedelta

import plotly.express as px
import streamlit as st
import ollama
import matplotlib.pyplot as plt

from src.data.loading import DataLoader
from src.data.split import train_test_split_by_date
from src.features.feature_engineering import FeaturesMaker
from src.model.catboost import CatBoostPredictor
from data.rag.text_processing import process
from data.rag.common.settings import SingletonConfig
from data.rag.common.prompts import *

# Function definitions
def analyze_graph(model_name, image_path):
    response = ollama.chat(
        model=model_name,
        messages=[{
            'role': 'user',
            'content': GRAPH_EXPLANATION,
            'images': [image_path]
        }]
    )
    return response['message']['content']

# Streamlit app
st.set_page_config(page_title="AI S&OP", page_icon="📊", layout="wide")

# Sidebar for page selection
page = st.sidebar.selectbox("Choose a page", ["Forecasting", "Data Analysis"])

if page == "Forecasting":
    st.title("AI S&OP - Forecasting")

    with st.sidebar:
        # Model selection
        st.subheader("Выбор модели для прогнозирования")
        model_choice = st.selectbox(
            "Выберите модель",
            ["Gradient Boosting", "Exponential Smoothing", "Prophet"],
        )

        # Data loading
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

    # Data loading and display
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

        # Data preparation for models
        train_df, test_df = train_test_split_by_date(dataset, 'date', new_date_str)

        # Forecasting
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

            # Interactive graph display
            for store in preds['store_id'].unique():
                store_pred_df = preds[preds['store_id'] == store]

                st.subheader("Интерактивный график")
                forecast_fig = px.line(
                    store_pred_df, x='date', y='cnt_predict', title="Прогноз",
                    labels={'date': 'Дата', 'cnt_predict': 'Прогноз'},
                )
                st.plotly_chart(forecast_fig)

            # Download forecasts button
            st.subheader("Скачать прогнозы")
            if st.button("Скачать прогнозы"):
                forecast_download = preds.to_csv().encode('utf-8')
                st.download_button(
                    label="Скачать как CSV",
                    data=forecast_download,
                    file_name='forecast.csv',
                    mime='text/csv',
                )

elif page == "Data Analysis":
    st.title("AI S&OP - Data Analysis")
    first_run = True

    llm_name = st.text_input("LLM Model Name", value="ollama_phi-3-medium-128k")
    vision_model_name = st.text_input("Vision Model Name", value="llava-phi3")
    data_path = st.text_input("Data Path")
    if st.button("Select Folder"):
        data_path = st.text_input("Selected Folder/File", value=data_path, type="default")

    question = st.text_area("Question", value="What is that data about?")

    if st.button("Process"):
        if not data_path:
            st.error("Please select a folder for the data path.")
        else:
            llm_config = SingletonConfig(llm_name)
            response = process(data_path, llm_config, question)
            st.write("Response:")
            st.write(response)

    graph_prompt = st.text_area("Graph Prompt", value="Generate a bar graph showing the total sales count for each store.")
    image_path = 'graph.png'

    if st.button("Generate Graph"):
        if not data_path:
            st.error("Please select a folder for the data path.")
        else:
            llm_config = SingletonConfig(llm_name)
            data_description = process(data_path, llm_config, DATA_EXPLANATION, top_lines=True)
            graph_code = process(data_path, llm_config, GRAPH_GENERATION(graph_prompt, data_path, data_description), top_lines=True)
            
            with open('tmp.py', 'w') as f:
                f.write(graph_code.strip())

            while(True):
                try:
                    exec(graph_code.strip(), globals())
                    fig = plt.gcf()
                    plt.savefig(image_path)
                    st.pyplot(fig)
                    st.session_state['graph_generated'] = True
                    first_run = True
                    break
                except Exception as e:
                    st.error(f"Error generating graph: {str(e)}")
                    break

    # Display the graph if it has been generated
    if 'graph_generated' in st.session_state and st.session_state['graph_generated'] and not first_run:
        st.image(image_path)

    if os.path.exists(image_path):
        if st.button("Summarize Graph Info"):
            first_run = False
            st.image(image_path)
            analysis = analyze_graph(vision_model_name, image_path)
            st.write("Graph Analysis:")
            st.write(analysis)
