import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def hyperparameter_tuning_page():
    st.title("🎛️ Hyperparameter Tuning")
    
    if "df" not in st.session_state or st.session_state.df is None:
        st.warning("⚠️ No dataset found. Please upload a dataset first.")
        return
    
    df = st.session_state.df.copy()
    
    # Select problem type
    problem_type = st.session_state.get("problem_type", "Classification")
    st.subheader("📌 Problem Type: " + problem_type)
    
    # Select target column
    target_column = st.selectbox("🎯 Select Target Column", df.columns)
    
    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Drop non-numeric features for models
    X = df[numeric_features]
    y = df[target_column]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "Classification" else None
    )
    
    # Model selection
    model_options = {
        "Regression": {
            "Random Forest Regressor": RandomForestRegressor(),
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "KNN Regressor": KNeighborsRegressor(),
            "Support Vector Regressor (SVR)": SVR()
        },
        "Classification": {
            "Random Forest Classifier": RandomForestClassifier(),
            "Gradient Boosting Classifier": GradientBoostingClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(),
            "KNN Classifier": KNeighborsClassifier(),
            "Support Vector Classifier (SVC)": SVC()
        }
    }
    
    model_choice = st.selectbox("Select Model for Tuning", list(model_options[problem_type].keys()))
    model = model_options[problem_type][model_choice]
    
    # Select search method
    tuning_method = st.radio(" Choose Hyperparameter Tuning Method", ["Grid Search", "Randomized Search", "Manual Tuning"])
    
    # Define default hyperparameter grid
    param_grid = {}
    if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10]
        }
    elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
        param_grid = {
            'n_neighbors': [3, 5, 10]
        }
    elif isinstance(model, (SVC, SVR)):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    
    # Manual tuning option
    if tuning_method == "Manual Tuning":
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor)):
            n_estimators = st.slider("🌲 Number of Trees (n_estimators)", 50, 500, 100, step=50)
            max_depth = st.slider(" Max Depth", 2, 20, 10, step=2)
            model.set_params(n_estimators=n_estimators, max_depth=max_depth)
        elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
            n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5, step=1)
            model.set_params(n_neighbors=n_neighbors)
        elif isinstance(model, (SVC, SVR)):
            C = st.slider(" Regularization Parameter (C)", 0.1, 10.0, 1.0, step=0.1)
            kernel = st.selectbox("Kernel Type", ["linear", "rbf"])
            model.set_params(C=C, kernel=kernel)
    
    if st.button(" Run Hyperparameter Tuning") and tuning_method != "Manual Tuning":
        if tuning_method == "Grid Search":
            search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
        else:
            search = RandomizedSearchCV(model, param_grid, cv=5, n_jobs=-1, n_iter=5)
        
        search.fit(X_train, y_train)
        best_params = search.best_params_
        
        st.success(f"✅ Best Hyperparameters: {best_params}")
        st.session_state.best_model = search.best_estimator_
    
    if st.button("➡️ Proceed to Model Training"):
        st.session_state.page = "model_training"
        st.session_state.df = df
        st.rerun()
