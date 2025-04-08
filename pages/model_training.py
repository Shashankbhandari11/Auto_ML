import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def model_training_page():
    st.title("Model Training")

    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset found. Please upload a dataset first.")
        return

    df = st.session_state.df.copy()
    
    # Select problem type
    problem_type = st.session_state.get("problem_type", "Classification")
    st.subheader("📌 Problem Type: " + problem_type)
    
    # Select target column
    target_column = st.selectbox(" Select Target Column", df.columns)
    
    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Drop non-numeric features for models
    X = df[numeric_features]
    y = df[target_column]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model selection based on problem type
    model_options = {
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor (SVR)": SVR()
        },
        "Classification": {
            "Logistic Regression": LogisticRegression(),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
            "Support Vector Classifier (SVC)": SVC()
        }
    }
    
    model_choice = st.selectbox(" Select Model", list(model_options[problem_type].keys()))
    model = model_options[problem_type][model_choice]
    
    # Hyperparameter tuning option
    tuning_option = st.radio(" Do you want to apply hyperparameter tuning?", ["No", "Yes"])
    
    if tuning_option == "Yes":
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
            n_estimators = st.slider(" Number of Trees (n_estimators)", 50, 500, 100, step=50)
            max_depth = st.slider(" Max Depth", 2, 20, 10, step=2)
            model.set_params(n_estimators=n_estimators, max_depth=max_depth)
        elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            n_neighbors = st.slider(" Number of Neighbors (K)", 1, 20, 5, step=1)
            model.set_params(n_neighbors=n_neighbors)
    
    # Train Model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if problem_type == "Regression":
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.success(f"✅ Model trained!\n MSE: {mse:.4f}\n R² Score: {r2:.4f}")
        else:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            st.success(f"✅ Model trained!\n Accuracy: {accuracy:.2%}\n Precision: {precision:.2%}\n Recall: {recall:.2%}\n F1 Score: {f1:.2%}")
        
        # Cross-validation score
        scores = cross_val_score(model, X, y, cv=5)
        st.write(f"Cross-validation score: {scores.mean():.2%} ± {scores.std():.2%}")
        
        # Feature Importance (only for tree-based models)
        if hasattr(model, "feature_importances_"):
            st.subheader("Feature Importance")
            feature_importances = pd.Series(model.feature_importances_, index=numeric_features).sort_values(ascending=False)
            fig, ax = plt.subplots()
            feature_importances.plot(kind='bar', ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)