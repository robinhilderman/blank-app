import streamlit as st
import plotly.graph_objects as go
import numpy as np
import glob
import pandas as pd
import os
from python_ags4 import AGS4
from python_ags4.data import load_test_data
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
    "Cohere reranker",
    "Contextual compression"
    ''
]

####################################################################
#            Create app interface with streamlit
####################################################################

st.title("Ground Hazard Dashboard")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
# API keys
st.session_state.openai_api_key = ""

# Model and parameters expander
def expander_model_parameters(
    LLM_provider="OpenAI",
    text_input_API_key="OpenAI API Key - [Get an API key](https://platform.openai.com/account/api-keys)",
    list_models=["text-embedding-ada-002", "gpt-4o-mini"],
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = LLM_provider

    if LLM_provider == "OpenAI":
        st.session_state.openai_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.google_api_key = ""
        st.session_state.hf_api_key = ""

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

   #sidebar = sidebar_and_documentChooser()

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

            if (
                st.session_state.retriever_type == list_retriever_types[0]
                and not st.session_state.cohere_api_key
            ):
                error_messages.append(f"insert your Cohere API key")

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
                            embeddings = select_embeddings_model()
                            st.session_state.vector_store = Chroma(
                                embedding_function=embeddings,
                                persist_directory=selected_vectorstore_path,
                            )

                            # 2. create retriever
                            st.session_state.retriever = create_retriever(
                                vector_store=st.session_state.vector_store,
                                embeddings=embeddings,
                                retriever_type=st.session_state.retriever_type,
                                base_retriever_search_type="similarity",
                                base_retriever_k=16,
                                compression_retriever_k=20,
                                cohere_api_key=st.session_state.cohere_api_key,
                                cohere_model="rerank-multilingual-v2.0",
                                cohere_top_n=10,
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