import streamlit as st
from pages import data_loading, data_preprocessing, exploratory_data_analysis, model_training, hyperparameter_tuning


st.title("Auto EDA and Pre Processing Project")
# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "data_loading"
if "df" not in st.session_state:
    st.session_state.df = None  # Ensure dataset is stored

# Sidebar with only radio buttons
page = st.sidebar.radio("Steps", [
    "Data Loading", "Data Preprocessing", "Exploratory Data Analysis", "Model Training", "Hyperparameter Tuning"
])

# Update session state
st.session_state.page = page.lower().replace(" ", "_")

# Page routing
if st.session_state.page == "data_loading":
    data_loading.data_loading_page()
elif st.session_state.page == "data_preprocessing":
    data_preprocessing.data_preprocessing_page()
elif st.session_state.page == "exploratory_data_analysis":
    exploratory_data_analysis.eda_page()
elif st.session_state.page == "model_training":
    model_training.model_training_page()
elif st.session_state.page == "hyperparameter_tuning":
    hyperparameter_tuning.hyperparameter_tuning_page()
