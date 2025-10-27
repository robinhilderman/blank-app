####################################################################
#                         import
####################################################################

import os, glob, tiktoken
from pathlib import Path


import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any

from python_ags4 import AGS4
from python_ags4.data import load_test_data

from operator import itemgetter

# Import openai as main LLM services
import openai
from openai import AzureOpenAI

# PDF loader
from langchain_community.document_loaders import PyPDFLoader

# Embedding
from langchain_community.embeddings import AzureOpenAIEmbeddings

# text_splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Vector store
import tkinter as tk
from tkinter import filedialog

# Retriever
from langchain_core.retrievers import BaseRetriever

# Import chroma as the vector store
from langchain_chroma import Chroma

# Contextual Compression
from langchain_community.document_transformers import EmbeddingsRedundantFilter,LongContextReorder

# Chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Prompt 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# OutputParser
from langchain_core.output_parsers import StrOutputParser

# Chat
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureChatOpenAI

 #?
from langchain_core.documents import Document

# Streamlit and visualisation
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import tkinter as tk
from tkinter import filedialog


def load_ags_convert_df_to_numeric(directory_path) -> pd.DataFrame:
    ags_files = glob.glob(os.path.join(directory_path, '*.ags'))
    all_numeric_tables = {}
    for file_path in ags_files:
        tables, _ = AGS4.AGS4_to_dataframe(file_path)
        numeric_tables = {k: AGS4.convert_to_numeric(v) for k, v in tables.items()}
        for k, df in numeric_tables.items():
            if k in all_numeric_tables:
                all_numeric_tables[k] = pd.concat([all_numeric_tables[k], df], ignore_index=True)
            else:
                all_numeric_tables[k] = df
    return all_numeric_tables

numeric_tables = load_ags_convert_df_to_numeric('/workspaces/blank-app/.devcontainer/data')

def create_loca_geol_table(numeric_tables):
    loca_cols = ['LOCA_ID', 'LOCA_NATE', 'LOCA_NATN', 'LOCA_GL', 'LOCA_REM']
    geol_cols = ['LOCA_ID', 'GEOL_TOP', 'GEOL_BASE', 'GEOL_DESC','GEOL_LEG', 'GEOL_GEOL', 'GEOL_STAT']
    loca_df = numeric_tables['LOCA'][loca_cols]
    geol_df = numeric_tables['GEOL'][geol_cols]
    merged_df = geol_df.merge(loca_df, on='LOCA_ID', how='left')
    # Drop rows where both 'LOCA_NATE' and 'LOCA_NATN' are null
    merged_df = merged_df[~(merged_df['LOCA_NATE'].isnull() & merged_df['LOCA_NATN'].isnull())]
    return merged_df

result_df = create_loca_geol_table(numeric_tables)

####################################################################
#              Config: LLM services, assistant language,...
####################################################################

list_LLM_providers = [
    ":rainbow[**OpenAI**]"]

dict_welcome_message = {
    "english": "How can I assist you today?",
    "french": "Comment puis-je vous aider aujourd’hui ?",
    "spanish": "¿Cómo puedo ayudarle hoy?",
    "german": "Wie kann ich Ihnen heute helfen?",
    "russian": "Чем я могу помочь вам сегодня?",
    "chinese": "我今天能帮你什么？",
    "arabic": "كيف يمكنني مساعدتك اليوم؟",
    "portuguese": "Como posso ajudá-lo hoje?",
    "italian": "Come posso assistervi oggi?",
    "Japanese": "今日はどのようなご用件でしょうか?",
}

list_retriever_types = [
    "Vectorstore backed retriever",
    "Contextual compression",
        "Cohere reranker"
    ''
]

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")
LOCAL_VECTOR_STORE_DIR = (
    Path(__file__).resolve().parent.joinpath("data", "vector_stores")
)


####################################################################
#            Create app interface with streamlit
####################################################################

st.title("Ground Hazard Dashboard")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
# API keys
st.session_state.openai_embedding_key = ""
st.session_state.openai_chat_key = ""

# Model and parameters expander
def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["text-embedding-ada-002", "gpt-4o-mini"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI_Embeddings":
        st.session_state.openai_embeddings_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_chat_key = ""

    if LLM_provider == "OpenAI_Chat":
        st.session_state.openai_chat_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.openai_embedding_key = ""

    with st.expander("**Models and parameters**"):
        st.session_state.selected_model = st.selectbox(
            f"Choose {LLM_provider} model", list_models
        )

        # model parameters
        st.session_state.temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )
        st.session_state.top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            value=0.95,
            step=0.05,
        )

model_set_up = expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["text-embedding-ada-002", "gpt-4o-mini"],
)

#sidebar

def sidebar_and_documentChooser():
    """Create the sidebar and the a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an old vectorstore)."""

    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, OpenAI, and Ground Engineering"
        )
        st.write("")

        llm_chooser = st.radio(
            "Select provider",
            list_LLM_providers,
            captions=[
                "[OpenAI pricing page](https://openai.com/pricing)",
                "Rate limit: 60 requests per minute.",
                "**Free access.**",
            ],
        )

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                LLM_provider="OpenAI",
                text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
                list_models=[
                    "text-embedding-ada-002", 
                    "gpt-4o-mini"
                ],
            )

        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types

        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", retrievers
        )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, \
            and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )

    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore
       
    tab_new_vectorstore, tab_open_vectorstore = st.tabs(
        ["Create a new Vectorstore", "Open a saved Vectorstore"]
    )
    with tab_new_vectorstore:
        # 1. Select documnets
        st.session_state.uploaded_file_list = st.file_uploader(
            label="**Select documents**",
            accept_multiple_files=True,
            type=(["pdf", "txt", "docx", "csv"]),
        )
        # 2. Process documents
        st.session_state.vector_store_name = st.text_input(
            label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
            placeholder="Vectorstore name",
        )
        # 3. Add a button to process documnets and create a Chroma vectorstore

        st.button("Create Vectorstore", on_click=chain_RAG_blocks)
        try:
            if st.session_state.error_message != "":
                st.warning(st.session_state.error_message)
        except:
            pass

    with tab_open_vectorstore:
        # Open a saved Vectorstore
        # https://github.com/streamlit/streamlit/issues/1019
        st.write("Please select a Vectorstore:")

        clicked = st.button("Vectorstore chooser")
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)  # Make dialog appear on top of other windows

        st.session_state.selected_vectorstore_name = ""

        if clicked:
            # Check inputs
            error_messages = []
            if (
                not st.session_state.openai_api_key
            ):
                error_messages.append(
                    f"insert your {st.session_state.LLM_provider} API key"
                )

            if len(error_messages) == 1:
                st.session_state.error_message = "Please " + error_messages[0] + "."
                st.warning(st.session_state.error_message)
            elif len(error_messages) > 1:
                st.session_state.error_message = (
                    "Please "
                    + ", ".join(error_messages[:-1])
                    + ", and "
                    + error_messages[-1]
                    + "."
                )
                st.warning(st.session_state.error_message)

            # if API keys are inserted, start loading Chroma index, then create retriever and ConversationalRetrievalChain
            else:
                selected_vectorstore_path = filedialog.askdirectory(master=root)

                if selected_vectorstore_path == "":
                    st.info("Please select a valid path.")

                else:
                    with st.spinner("Loading vectorstore..."):
                        st.session_state.selected_vectorstore_name = (
                            selected_vectorstore_path.split("/")[-1]
                        )
                        try:
                            # 1. load Chroma vectorestore
                            embeddings_model = select_embeddings_model()
                            st.session_state.vector_store = Chroma(
                                persist_directory = LOCAL_VECTOR_STORE_DIR.as_posix() + "/Vit_All_OpenAI_Embeddings",
                                embedding_function=embeddings_model)

                            # 2. create retriever
                            st.session_state.retriever = create_retriever(
                                vector_store=st.session_state.vector_store,
                                embeddings=embeddings_model,
                                retriever_type=st.session_state.retriever_type,
                                base_retriever_search_type="similarity",
                                base_retriever_k=16,
                                compression_retriever_k=20

                            )

                            # 3. create memory and ConversationalRetrievalChain
                            (
                                st.session_state.chain,
                                st.session_state.memory,
                            ) = create_ConversationalRetrievalChain(
                                retriever=st.session_state.retriever,
                                chain_type="stuff",
                                language=st.session_state.assistant_language,
                            )

                            # 4. clear chat_history
                            clear_chat_history()

                            st.info(
                                f"**{st.session_state.selected_vectorstore_name}** is loaded successfully."
                            )

                        except Exception as e:
                            st.error(e)
        


def plot_well_locations(result_df):
    st.title("Well Location Finder")

    input_type = st.radio("Input coordinate type:", ("Easting/Northing", "Latitude/Longitude"))
    if input_type == "Easting/Northing":
        x = st.number_input("Enter Easting (LOCA_NATE):", value=0.0)
        y = st.number_input("Enter Northing (LOCA_NATN):", value=0.0)
    else:
        x = st.number_input("Enter Longitude:", value=0.0)
        y = st.number_input("Enter Latitude:", value=0.0)

    # Calculate Euclidean distance (assumes projected coordinates, e.g., British National Grid)
    dists = np.sqrt((result_df['LOCA_NATE'] - x)**2 + (result_df['LOCA_NATN'] - y)**2)
    filtered = result_df[dists <= 400]

    # Prepare dataframe for mapping: treat LOCA_NATE as lon and LOCA_NATN as lat
    map_df = result_df.dropna(subset=['LOCA_NATE', 'LOCA_NATN']).copy()
    map_df['lon'] = map_df['LOCA_NATE']
    map_df['lat'] = map_df['LOCA_NATN']

    filtered_map = filtered.copy()
    filtered_map['lon'] = filtered_map['LOCA_NATE']
    filtered_map['lat'] = filtered_map['LOCA_NATN']

    # Create map with all wells (light grey) and highlight filtered wells + search point
    fig = px.scatter_map(
        map_df,
        lat='lat',
        lon='lon',
        hover_name='LOCA_ID' if 'LOCA_ID' in map_df.columns else None,
        color_discrete_sequence=['lightgrey'],
        zoom=10,
        height=600,
    )

    # add filtered wells
    fig.add_trace(go.Scattergeo(
        lat=filtered_map['lat'],
        lon=filtered_map['lon'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Wells within 400m'
    ))

    # add search point
    fig.add_trace(go.Scattergeo(
        lat=[y],
        lon=[x],
        mode='markers',
        marker=dict(size=14, color='red'),
        name='Search point'
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_center={"lat": float(y), "lon": float(x)},
        mapbox_zoom=12,
        title="Well Locations (map)"
    )

    st.plotly_chart(fig, use_container_width=True)
    return filtered

filtered_wells = plot_well_locations(result_df)


####################################################################
#        Process documents and create vectorstor (Chroma dB)
####################################################################

def load_all_pdfs_from_directory(directory_path):
    pdf_paths = glob.glob(f"{directory_path}/*.pdf")
    all_documents = [os.path.basename(pdf) for pdf in pdf_paths]
    all_data_metadata = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(file_path=pdf)
        document = loader.load()
        for page in document:
            doc = Document(
                page_content = page.page_content,
                metadata = {"source": pdf}
            )
            all_data_metadata.append(doc)
    return all_documents, all_data_metadata

pdf_directory = '/workspaces/blank-app/.devcontainer/data'
_, all_data_metadata = load_all_pdfs_from_directory(pdf_directory)

def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model():
    """Select embeddings models: OpenAIEmbeddings ."""
    if st.session_state.LLM_provider == "OpenAI_Embeddings":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_embedding_key)

    if st.session_state.LLM_provider == "OpenAI_Chat":
        embeddings = OpenAIEmbeddings(api_key=st.session_state.openai_chat_key)

    return embeddings


def create_retriever(
    vector_store,
    embeddings,
    retriever_type="Contextual compression",
    base_retriever_search_type="semilarity",
    base_retriever_k=16,
    compression_retriever_k=20
):
    """
    create a retriever which can be a:
        - Vectorstore backed retriever: this is the base retriever.
        - Contextual compression retriever: We wrap the the base retriever in a ContextualCompressionRetriever.
            The compressor here is a Document Compressor Pipeline, which splits documents
            to smaller chunks, removes redundant documents, filters the top relevant documents,
            and reorder the documents so that the most relevant are at beginning / end of the list.
        - Cohere_reranker: CohereRerank endpoint is used to reorder the results based on relevance.

    Parameters:
        vector_store: Chroma vector database.
        embeddings: OpenAIEmbeddings or GoogleGenerativeAIEmbeddings.

        retriever_type (str): in [Vectorstore backed retriever,Contextual compression,Cohere reranker]. default = Cohere reranker

        base_retreiver_search_type: search_type in ["similarity", "mmr", "similarity_score_threshold"], default = similarity.
        base_retreiver_k: The most similar vectors are returned (default k = 16).

        compression_retriever_k: top k documents returned by the compression retriever, default = 20

    """

    base_retriever = vectorstore_backed_retriever(
        vectorstore=vector_store,
        search_type=base_retriever_search_type,
        k=base_retriever_k,
        score_threshold=None,
    )

    if retriever_type == "Vectorstore backed retriever":
        return base_retriever

    else:
        pass


def vectorstore_backed_retriever(
    vectorstore, search_type="similarity", k=4, score_threshold=None
):
    """create a vectorsore-backed retriever
    Parameters:
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold"
        k: number of documents to return (Default: 4)
        score_threshold: Minimum relevance threshold for similarity_score_threshold (default=None)
    """
    search_kwargs = {}
    if k is not None:
        search_kwargs["k"] = k
    if score_threshold is not None:
        search_kwargs["score_threshold"] = score_threshold

    retriever = vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )
    return retriever


def chain_RAG_blocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = []
        if (
            not st.session_state.openai_api_key
        ):
            error_messages.append(
                f"insert your {st.session_state.LLM_provider} API key"
            )

        if (
            st.session_state.retriever_type == list_retriever_types[0]
            and not st.session_state.cohere_api_key
        ):
            error_messages.append(f"insert your Cohere API key")
        if not st.session_state.uploaded_file_list:
            error_messages.append("select documents to upload")
        if st.session_state.vector_store_name == "":
            error_messages.append("provide a Vectorstore name")

        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = (
                "Please "
                + ", ".join(error_messages[:-1])
                + ", and "
                + error_messages[-1]
                + "."
            )
        else:
            st.session_state.error_message = ""
            try:


                # 2. Upload selected documents to temp directory
                if st.session_state.uploaded_file_list is not None:
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        try:
                            temp_file_path = os.path.join(
                                TMP_DIR.as_posix(), uploaded_file.name
                            )
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = load_all_pdfs_from_directory(pdf_directory)

                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    # 5. Embeddings
                    embeddings = select_embeddings_model()

                    # 6. Create a vectorstore
                    persist_directory = (
                        LOCAL_VECTOR_STORE_DIR.as_posix()
                        + "/"
                        + st.session_state.vector_store_name
                    )

                    try:
                        st.session_state.vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory=persist_directory,
                        )
                        st.info(
                            f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully."
                        )

                        # 7. Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        (
                            st.session_state.chain,
                            st.session_state.memory,
                        ) = create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                        )

                        # 9. Cclear chat_history
                        clear_chat_history()

                    except Exception as e:
                        st.error(e)

            except Exception as error:
                st.error(f"An error occurred: {error}")

####################################################################
#                       Create memory
####################################################################

class CustomSummaryBufferMemory:
    def __init__(
        self,
        llm,
        max_token_limit=1024,
        memory_key='chat_history',
        input_key='question',
        output_key='answer',
        return_messages=True,
        ai_prefix='AI',
        human_prefix='Human'
    ):
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.chat_history = []
        self.summary = ""

    def _summarize(self, new_lines):
        prompt = (
            "Progressively summarize the lines of conversation provided, "
            "adding onto the previous summary returning a new summary.\n\n"
            "EXAMPLE\n"
            "Current summary:\n"
            "The human asks what the AI thinks of artificial intelligence. "
            "The AI thinks artificial intelligence is a force for good.\n\n"
            "New lines of conversation:\n"
            "Human: Why do you think artificial intelligence is a force for good?\n"
            "AI: Because artificial intelligence will help humans reach their full potential.\n\n"
            "New summary:\n"
            "The human asks what the AI thinks of artificial intelligence. "
            "The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.\n"
            "END OF EXAMPLE\n\n"
            f"Current summary:\n{self.summary}\n\n"
            f"New lines of conversation:\n{new_lines}\n\n"
            "New summary:"
        )
        # Use the LLM to generate a new summary
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def save_context(self, inputs, outputs):
        # Add new message to history
        question = inputs.get(self.input_key, "")
        answer = outputs.get(self.output_key, "")
        self.chat_history.append(
            {"role": self.human_prefix, "content": question}
        )
        self.chat_history.append(
            {"role": self.ai_prefix, "content": answer}
        )
        # Prune if over token limit
        self._prune()

    def _prune(self):
        # Simple token counting using number of words (replace with tiktoken for accuracy)
        def count_tokens(messages):
            return sum(len(m["content"].split()) for m in messages)
        while count_tokens(self.chat_history) > self.max_token_limit and len(self.chat_history) > 2:
            # Summarize the oldest two messages and replace them with a summary
            oldest = self.chat_history[:2]
            new_lines = "\n".join([f"{m['role']}: {m['content']}" for m in oldest])
            self.summary = self._summarize(new_lines)
            # Remove the oldest two messages
            self.chat_history = self.chat_history[2:]
        # Optionally, keep the summary as a system message
        if self.summary:
            self.chat_history = (
                [{"role": "system", "content": self.summary}] + self.chat_history
            )

    def load_memory_variables(self, inputs):
        # Return the chat history and summary
        if self.return_messages:
            return {self.memory_key: self.chat_history}
        else:
            return {self.memory_key: "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history])}

    def clear(self):
        self.chat_history = []
        self.summary = ""

class CustomConversationBufferMemory:
    def __init__(
        self,
        memory_key='chat_history',
        input_key='question',
        output_key='answer',
        return_messages=True,
        ai_prefix='AI',
        human_prefix='Human'
    ):
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.ai_prefix = ai_prefix
        self.human_prefix = human_prefix
        self.chat_history = []

    def save_context(self, inputs, outputs):
        question = inputs.get(self.input_key, "")
        answer = outputs.get(self.output_key, "")
        self.chat_history.append({"role": self.human_prefix, "content": question})
        self.chat_history.append({"role": self.ai_prefix, "content": answer})

    def load_memory_variables(self, inputs):
        if self.return_messages:
            return {self.memory_key: self.chat_history}
        else:
            return {
                self.memory_key: "\n".join(
                    [f"{m['role']}: {m['content']}" for m in self.chat_history]
                )
            }

    def clear(self):
        self.chat_history = []

    @property
    def buffer(self):
        return self.chat_history

    @property
    def buffer_as_messages(self):
        return self.chat_history

    @property
    def buffer_as_str(self):
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.chat_history])

def create_memory(model_name='gpt-4o-mini', memory_max_token=None):
    """Creates a custom summary buffer memory for gpt-4o-mini,
    or a ConversationBufferMemory for other models."""
    if model_name == "gpt-4o-mini":
        if memory_max_token is None:
            memory_max_token = 1024
        memory = CustomSummaryBufferMemory(
            llm=ChatOpenAI(
                openai_api_key=st.session_state.openai_chat_key,  
                model_name="gpt-4o-mini",
                temperature=0.1
            ),
            max_token_limit=memory_max_token,
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question"
        )
    else:
        memory = CustomConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            output_key="answer",
            input_key="question",
        )
    return memory


####################################################################
#          Create ConversationalRetrievalChain with memory
####################################################################


def answer_template(language="english"):
    """Pass the standalone question along with the chat history and context
    to the `LLM` wihch will answer."""

    template = f"""Answer the question at the end, using only the following context (delimited by <context></context>).
Your answer must be in the language at the end. 

<context>
{{chat_history}}

{{context}} 
</context>

Question: {{question}}

Language: {language}.
"""
    return template


def create_ConversationalRetrievalChain(
    retriever,
    chain_type="stuff",
    language="english",
):
    """Create a ConversationalRetrievalChain.
    First, it passes the follow-up question along with the chat history to an LLM which rephrases
    the question and generates a standalone query.
    This query is then sent to the retriever, which fetches relevant documents (context)
    and passes them along with the standalone question and chat history to an LLM to answer.
    """

    # 1. Define the standalone_question prompt.
    # Pass the follow-up question along with the chat history to the `condense_question_llm`
    # which rephrases the question and generates a standalone question.

    condense_question_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template="""Given the following conversation and a follow up question, 
rephrase the follow up question to be a standalone question, in its original language.\n\n
Chat History:\n{chat_history}\n
Follow Up Input: {question}\n
Standalone question:""",
    )

    # 2. Define the answer_prompt
    # Pass the standalone question + the chat history + the context (retrieved documents)
    # to the `LLM` wihch will answer

    answer_prompt = ChatPromptTemplate.from_template(answer_template(language=language))

    # 3. Add ConversationSummaryBufferMemory for gpt-3.5, and ConversationBufferMemory for the other models
    memory = create_memory(st.session_state.selected_model)

    # 4. Instantiate LLMs: standalone_query_generation_llm & response_generation_llm
    if st.session_state.LLM_provider == "OpenAI":
        standalone_query_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=0.1,
        )
        response_generation_llm = ChatOpenAI(
            api_key=st.session_state.openai_api_key,
            model=st.session_state.selected_model,
            temperature=st.session_state.temperature,
            model_kwargs={"top_p": st.session_state.top_p},
        )

    # 5. Create the ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(
        condense_question_prompt=condense_question_prompt,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        condense_question_llm=standalone_query_generation_llm,
        llm=response_generation_llm,
        memory=memory,
        retriever=retriever,
        chain_type=chain_type,
        verbose=False,
        return_source_documents=True,
    )

    return chain, memory

def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass

def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        if st.session_state.LLM_provider == "HuggingFace":
            answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

        # 2. Display results
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

            # 2.2. Display source documents:
            with st.expander("**Source documents**"):
                documents_content = ""
                for document in response["source_documents"]:
                    try:
                        page = " (Page: " + str(document.metadata["page"]) + ")"
                    except:
                        page = ""
                    documents_content += (
                        "**Source: "
                        + str(document.metadata["source"])
                        + page
                        + "**\n\n"
                    )
                    documents_content += document.page_content + "\n\n\n"

                st.markdown(documents_content)

    except Exception as e:
        st.warning(e)



####################################################################
#                         Chatbot
####################################################################
def chatbot():
    sidebar_and_documentChooser()
    st.divider()
    col1, col2 = st.columns([7, 3])
    with col1:
        st.subheader("Chat with your data")
    with col2:
        st.button("Clear Chat History", on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": dict_welcome_message[st.session_state.assistant_language],
            }
        ]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if (
            not st.session_state.openai_api_key
            and not st.session_state.google_api_key
            and not st.session_state.hf_api_key
        ):
            st.info(
                f"Please insert your {st.session_state.LLM_provider} API key to continue."
            )
            st.stop()
        with st.spinner("Running..."):
            get_response_from_LLM(prompt=prompt)


if __name__ == "__main__":
    chatbot()