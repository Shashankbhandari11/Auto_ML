# Code by Shashank Bhandari

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def eda_page():
    st.title(" Exploratory Data Analysis")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset found. Please upload a dataset first.")
        return

    df = st.session_state.df.copy()

    # 🔹 Display missing values
    st.subheader(" Missing Values")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    
    if not missing_values.empty:
        st.write("Total Missing Values per Column:")
        st.dataframe(missing_values)
        
        if st.button("Drop Missing Values"):
            df.dropna(inplace=True)
            st.session_state.df = df  # ✅ Update dataset
            st.success("✅ Missing values dropped successfully!")
            st.rerun()
    else:
        st.success("✅ No missing values found!")

    # Feature Distribution (Numerical Variables)
    st.subheader("Feature Distribution (Numerical Columns)")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(6, len(numeric_cols) * 2))
        if len(numeric_cols) == 1:
            axes = [axes]  # Ensure axes is iterable
        for ax, col in zip(axes, numeric_cols):
            sns.histplot(df[col], bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("ℹ️ No numerical columns found.")

     # 🔹 Display Skewness Values
    st.subheader(" Skewness Analysis")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    skewness = df[numeric_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
    st.write("**Skewness of Numeric Features (Before Transformation):**")
    st.dataframe(skewness.to_frame("Skewness"))

    # Detect Highly Skewed Columns
    highly_skewed_cols = skewness[abs(skewness) > 1].index.tolist()
    if highly_skewed_cols:
        st.warning(f"⚠️ The following columns are highly skewed: {highly_skewed_cols}")

        if st.button(" Auto-Apply Best Transformations"):
            for col in highly_skewed_cols:
                if skewness[col] > 1:  # Right-Skewed
                    df[col] = np.log1p(df[col])  # Log Transformation
                elif skewness[col] < -1:  # Left-Skewed
                    df[col] = df[col] ** 2  # Square Transformation
            
            # ✅ Update session state with transformed data
            st.session_state.df = df

            # ✅ Show Skewness After Transformation
            new_skewness = df[highly_skewed_cols].apply(lambda x: x.skew()).sort_values(ascending=False)
            st.subheader(" Skewness After Automatic Transformation")
            st.dataframe(new_skewness.to_frame("Skewness"))

            st.success("✅ Best transformations applied automatically!")
            st.rerun()

    else:
        st.success("✅ No highly skewed columns detected.")

    #Boxplot for Outlier Detection
    st.subheader("Outlier Detection (Boxplot)")
    if numeric_cols:
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)
    else:
        st.info("ℹ️ No numerical columns found for boxplots.")

    #Outlier Removal (IQR Method)
    st.subheader("Remove Outliers")
    selected_outlier_cols = st.multiselect("Select columns for outlier removal", numeric_cols)
    if st.button("Apply Outlier Removal"):
        for col in selected_outlier_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        st.session_state.df = df  # Updates dataset in preprocessing
        st.success(" Outliers removed successfully!")
        st.rerun()



    #Correlation Matrix
    st.subheader("Correlation Matrix")
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 4))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        st.pyplot(plt)
    else:
        st.info("ℹ️ Not enough numerical columns for correlation matrix.")

    # Proceed to Model Training
    if st.button("Proceed to Model Training "):
        st.session_state.page = "model_training"
        st.rerun()
