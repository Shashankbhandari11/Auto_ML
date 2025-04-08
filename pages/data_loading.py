import streamlit as st
import pandas as pd

def data_loading_page():
    st.title("📂 Data Loading")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df = df  # Store in session state
            st.success("✅ Dataset loaded successfully!")

            st.write("📌 **Dataset Preview:**")
            st.dataframe(df.head(20))

        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

    if st.button("Proceed to Data Preprocessing ➡️"):
        if st.session_state.df is None:
            st.warning("⚠️ Please upload a dataset first.")
        else:
            st.session_state.page = "data_preprocessing"
            st.rerun()
