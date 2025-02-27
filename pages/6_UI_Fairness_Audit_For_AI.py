import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import aif360
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf

# Streamlit UI
st.title("Fairness and Bias Analysis Tool")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of the dataset:")
    st.write(df.head())
    
    # Select target variable
    target_variable = st.selectbox("Select the target variable", df.columns)
    
    # Select protected attribute
    protected_attribute = st.selectbox("Select the protected attribute", df.columns)
    
    # Select optimization objective
    objective = st.radio("Select optimization objective", ("Fairness", "Accuracy"))
    
    # Select type of conceptual fairness
    fairness_type = st.selectbox("Select fairness concept", ["Demographic Parity", "Equalized Odds", "Predictive Parity"])
    
    # Select model
    model_choice = st.selectbox("Select a machine learning model", ["Logistic Regression", "Random Forest", "Adversarial Debiasing"])
    
    # Run analysis button
    if st.button("Run Analysis"):
        st.write("### Running Analysis...")
        
        # Preprocess dataset
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if model_choice == "Logistic Regression":
            model = LogisticRegression()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "Adversarial Debiasing":
            privileged_group = [{protected_attribute: 1}]
            unprivileged_group = [{protected_attribute: 0}]
            dataset = StandardDataset(df, label_name=target_variable, protected_attribute_names=[protected_attribute])
            train, test = dataset.split([0.8], shuffle=True)
            sess = tf.Session()
            model = AdversarialDebiasing(privileged_groups=privileged_group, unprivileged_groups=unprivileged_group, sess=sess)
        
        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Model Accuracy: {accuracy:.2f}")
        
        # Fairness Analysis
        dataset_metric = BinaryLabelDatasetMetric(dataset, privileged_groups=privileged_group, unprivileged_groups=unprivileged_group)
        
        if fairness_type == "Demographic Parity":
            metric = dataset_metric.statistical_parity_difference()
        elif fairness_type == "Equalized Odds":
            metric = dataset_metric.equal_opportunity_difference()
        elif fairness_type == "Predictive Parity":
            metric = dataset_metric.average_abs_odds_difference()
        
        st.write(f"Fairness Metric ({fairness_type}): {metric:.4f}")
        
        if objective == "Fairness" and model_choice != "Adversarial Debiasing":
            st.write("Consider using Adversarial Debiasing for improved fairness.")
        elif objective == "Accuracy" and accuracy < 0.8:
            st.write("Consider tuning your model for better accuracy.")
