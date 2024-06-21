import os
import logging as log

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from .text_loader import data_loader


def load_data(data_path, top_lines=False):
    log.info(f'[RAG][Preprocess]: Reading documents from {data_path}')
    loaders = data_loader(data_path, top_lines)
    text = ''
    for loader in loaders:
        text += loader.get_text()
    return text

def split_text(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vector_store(chunks, embed_model):
    vector_store = Chroma.from_texts(chunks, embed_model)
    log.info('[RAG][DB]: Vector store created successfully')
    return vector_store


def create_retrieval_qa_chain(llm, retriever, langchain_model="langchain-ai/retrieval-qa-chat"):
    retrieval_qa_chat_prompt = hub.pull(langchain_model)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    log.info(f'[RAG][R-QA]: Retrieval chain created, using {langchain_model}')
    return retrieval_chain


def answer_question(question, retrieval_chain):
    response = retrieval_chain.invoke({"input": question})
    return response['answer']


def process(data_path, llm_config, question, top_lines=False):
    backend = llm_config['backend']
    log.info(f'[RAG] Using backend: {backend}')
    if backend == 'ollama':
        model_name = llm_config['model_name']

        log.info(f'[RAG][INIT]: Loading model: {backend}')
        llm = Ollama(model=model_name, base_url=llm_config['base_url'])
        embed_model = OllamaEmbeddings(
            model=model_name,
            base_url=llm_config['base_url']
        )

        text = load_data(data_path, top_lines=top_lines)
        chunks = split_text(text, llm_config['chunk_size'], llm_config['chunk_overlap'])
        vector_store = create_vector_store(chunks, embed_model)
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_qa_chain(llm, retriever)
        response = answer_question(question, retrieval_chain)
        return response
