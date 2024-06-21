import streamlit as st
import matplotlib.pyplot as plt

from data.rag.text_processing import process
from data.rag.common.settings import SingletonConfig
from data.rag.common.prompts import *

def main():
    st.title("S&OP AI")

    model_name = st.text_input("Model Name", value="ollama_phi-3-medium-128k")
    data_path = st.text_input("Data Path")
    if st.button("Select Folder"):
        data_path = st.text_input("Selected Folder/File", value=data_path, type="default")

    question = st.text_area("Question", value="What is that data about?")

    if st.button("Process"):
        if not data_path:
            st.error("Please select a folder for the data path.")
        else:
            llm_config = SingletonConfig(model_name)
            response = process(data_path, llm_config, question)
            st.write("Response:")
            st.write(response)

    graph_prompt = st.text_area("Graph Prompt", value="Generate a bar graph showing the total sales count for each store.")

    if st.button("Generate Graph"):
        if not data_path:
            st.error("Please select a folder for the data path.")
        else:
            llm_config = SingletonConfig(model_name)
            data_description = process(data_path, llm_config, DATA_EXPLANATION, top_lines=True)
            graph_code = process(data_path, llm_config, GRAPH_GENERATION(graph_prompt, data_path, data_description), top_lines=True)
            with open('tmp.py', 'w') as f:
                f.write(graph_code.strip())
            try:
                exec(graph_code.strip(), globals())
                fig = plt.gcf()
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating graph: {str(e)}")

if __name__ == "__main__":
    main()
