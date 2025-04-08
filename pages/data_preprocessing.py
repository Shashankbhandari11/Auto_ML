#Code by Shashank Bhandari

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from io import BytesIO

def data_preprocessing_page():
    st.title("🔄 Data Preprocessing")

    # Check if dataset is uploaded
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset found. Please upload a dataset in the Data Loading section first.")
        return

    df = st.session_state.df.copy()

    #removing the white space from the columns
    df.columns = df.columns.str.strip()
    st.session_state.df = df

    st.write(f"### 📂 The original dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")


    #Column Renaming Option
    st.subheader(" Rename Columns")
    rename_option = st.radio("Do you want to rename column names?", ("No", "Yes"), index=0)

    if rename_option == "Yes":
        new_column_names = {}

        for col in df.columns:
            new_name = st.text_input(f"Rename `{col}`:", value=col)
            new_column_names[col] = new_name

        if st.button("Apply Column Renaming"):
            df.rename(columns=new_column_names, inplace=True)
            st.session_state.df = df  # Update session state with new column names
            st.success("Column names updated successfully!")
            st.rerun()

        # Show Updated Column Names only if renaming was applied
            st.write("### Updated Columns:")
            st.write(list(df.columns))

    # 🔹 Removing Duplicates
    st.subheader(" Removal of Duplicate Records")
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        st.warning(f"⚠️ Found {duplicate_count} duplicate records.")
        if st.button("Remove Duplicates"):
            df.drop_duplicates(inplace=True)
            st.session_state.df = df  # Save back to session state
            st.success(f"✅ The dataset now contains {df.shape[0]} rows and {df.shape[1]} columns.")
            st.rerun()
    else:
        st.success("✅ No duplicate records found.")

     # 🔹 Handling Missing Values
    st.subheader("Handling Missing Values")
    total_null = df.isnull().sum().sum()
    st.write(f"**Total missing values in the dataset: {total_null}**")

    # Define column types **before** imputation
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if total_null > 0:
        st.write("#### Columns with Missing Values:")
        missing_counts = df.isnull().sum()[df.isnull().sum() > 0]
        st.dataframe(missing_counts)

        st.subheader("Imputation - Fill Missing Values")
        fill_mean_cols = st.multiselect("Fill missing values with Mean (Numeric Only)", numeric_cols)
        fill_median_cols = st.multiselect("Fill missing values with Median (Numeric Only)", numeric_cols)
        fill_mode_cols = st.multiselect("Fill missing values with Mode (Categorical Only)", categorical_cols)

        if st.button("Apply Imputation"):
            for col in fill_mean_cols:
                df[col] = df[col].fillna(df[col].mean())
            for col in fill_median_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in fill_mode_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
                df[col] = df[col].astype(str)  # Convert categorical back to object

            # 🔹 Recalculate column types **AFTER** imputation
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

            st.session_state.df = df
            st.success("✅ Imputation applied successfully.")
            st.rerun()

    # Debugging prints **before encoding and scaling**
    else:
        st.success("✅ No Missing value found.")
    st.write("Data Types After Imputation:")
    st.dataframe(df.dtypes.astype(str))

    #Deleting the columns
    st.subheader("Deletion of Columns")
    cols_to_drop = st.multiselect("Select columns to drop", df.columns.tolist())
    if st.button("Drop Selected Columns"):
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df = df  # Save back to session state
            st.success(f"✅ Dropped columns: {cols_to_drop}")
            st.rerun()
        else:
            st.warning("⚠️ Please select at least one column to drop.")

    # Display the updated dataset
    st.write("### Updated Dataset:")
    st.dataframe(df.head())

     #Encoding Categorical Variables
    st.subheader("Encoding Categorical Variables")
    if len(categorical_cols) > 0:
        selected_cat_cols = st.multiselect("Select categorical columns for Label Encoding", categorical_cols)
        if st.button("Apply Label Encoding") and selected_cat_cols:
            encoder = LabelEncoder()
            for col in selected_cat_cols:
                df[col] = encoder.fit_transform(df[col])
            st.session_state.df = df
            st.success(f"✅ Label encoding applied to: {selected_cat_cols}")
            st.rerun()
    else:
        st.success("✅ Encoding categorical has done successfully.")
        st.info("ℹ️ No categorical columns found for encoding.")

    # Display the updated dataset
    st.write("###  Updated Dataset:")
    st.dataframe(df.head())


    # 🎯 **Target Variable Selection**
    st.subheader("Target Variable Selection")
    target_variable = st.selectbox("Select the target variable", df.columns.tolist(), key="target_var")

    # 📌 **Problem Type Selection**
    st.subheader(" Problem Type Selection")
    problem_type = st.radio("Select problem type", ["Regression", "Classification"], key="problem_type")

    st.subheader("Train-Test Split")
    test_size = st.slider("Select test size", min_value=0.1, max_value=0.5, step=0.05, value=0.2, key="test_size")
    
    if st.button("Split Dataset"):
        if target_variable not in df.columns:
            st.error(" Please select a valid target variable before splitting.")
        else:
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            st.session_state["X_train"] = X_train
            st.session_state["X_test"] = X_test
            st.session_state["y_train"] = y_train
            st.session_state["y_test"] = y_test
            
            st.success(f"✅ Train-Test split completed! Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # Feature Scaling Standardization and Normalization
    # Initialize session state variables if they don't exist
    # Initialize session state variables if they don’t exist
    for key in ["X_train_standardized", "X_test_standardized", "X_train_normalized", "X_test_normalized"]:
        if key not in st.session_state:
            st.session_state[key] = None
    
    st.subheader("Feature Scaling")
    
    if "X_train" in st.session_state and "X_test" in st.session_state:
        X_train = st.session_state["X_train"]
        X_test = st.session_state["X_test"]
    
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
        if numeric_cols:
            standard_cols = st.multiselect("Select columns for Standardization", numeric_cols)
            if st.button("Apply Standardization") and standard_cols:
                scaler = StandardScaler()
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                X_train_scaled[standard_cols] = scaler.fit_transform(X_train[standard_cols])
                X_test_scaled[standard_cols] = scaler.transform(X_test[standard_cols])
    
                st.session_state["X_train_standardized"] = X_train_scaled
                st.session_state["X_test_standardized"] = X_test_scaled
                st.success(f" Standardization applied to: {standard_cols}")
                st.rerun()
    
            normalize_cols = st.multiselect("Select columns for Normalization", numeric_cols)
            if st.button("Apply Normalization") and normalize_cols:
                scaler = MinMaxScaler()
                X_train_scaled = X_train.copy()
                X_test_scaled = X_test.copy()
                X_train_scaled[normalize_cols] = scaler.fit_transform(X_train[normalize_cols])
                X_test_scaled[normalize_cols] = scaler.transform(X_test[normalize_cols])
    
                st.session_state["X_train_normalized"] = X_train_scaled
                st.session_state["X_test_normalized"] = X_test_scaled
                st.success(f"✅ Normalization applied to: {normalize_cols}")
                st.rerun()
        else:
            st.info("ℹ️ No numeric columns found for scaling.")
    
    # ✅ Safe Check Before Displaying Data
    if st.session_state["X_train_standardized"] is not None:
        st.write("###  Standardized Train Data (First 5 Rows)")
        st.dataframe(st.session_state["X_train_standardized"].head())
    
    if st.session_state["X_train_normalized"] is not None:
        st.write("###  Normalized Train Data (First 5 Rows)")
        st.dataframe(st.session_state["X_train_normalized"].head())
    # 💾 **Download Processed Dataset**
    st.subheader("💾 Download Processed Dataset")

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    def convert_df_to_excel(df):
        if df.empty:
            return None
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        processed_data = output.getvalue()  # Convert to bytes
        return processed_data

    st.download_button(" Download as CSV", data=convert_df_to_csv(df), file_name="processed_dataset.csv", mime="text/csv")
    st.download_button(
    " Download as Excel",
    data=convert_df_to_excel(df),
    file_name="processed_dataset.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # proceed to Next Step
    if st.button("Proceed to EDA ➡️"):
        st.session_state.page = "exploratory_data_analysis"
        st.rerun()
