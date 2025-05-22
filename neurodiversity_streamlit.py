import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Page configuration
st.set_page_config(
    page_title="Neurodiversity Analysis Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Application title and description
st.title("Neurodiversity Dataset Analysis")
st.markdown("""
This dashboard provides interactive analysis and visualizations of the neurodiversity dataset.
Explore various aspects of neurodiversity across different demographics, diagnoses, and life outcomes.
""")

# Load data with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("neurodiversity_dataset_no_country.csv")
        return data, None
    except Exception as e:
        return None, str(e)

# Load the dataset
df, error = load_data()

# Error handling for data loading
if error:
    st.error(f"Error loading the dataset: {error}")
    st.stop()

if df is None or df.empty:
    st.error("The dataset is empty or could not be loaded.")
    st.stop()

# Display basic dataset information
st.subheader("Dataset Overview")
st.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Display column names for debugging
with st.expander("View Column Names"):
    st.write("Available columns:", list(df.columns))

# Create a function to check if columns exist
def get_columns_if_exist(dataframe, column_list):
    """Return only columns that exist in the dataframe"""
    return [col for col in column_list if col in dataframe.columns]

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Dataset Overview", "Demographics Analysis", "Diagnosis Insights", "SVM Classification"]
)

# Get available demographic columns
demographic_columns = get_columns_if_exist(df, [
    "Age", "Gender", "Socioeconomic Status", "Job Type", "Employment History (Years)"
])

# Get available diagnosis columns
diagnosis_columns = get_columns_if_exist(df, [
    "Diagnoses", "Age at Diagnosis", "Type of Diagnosis", 
    "has_ASD", "has_ADHD", "has_Dyslexia"
])

# Get available performance columns
performance_columns = get_columns_if_exist(df, [
    "Attention Span (minutes)", "Problem-solving Skills (Score)",
    "Memory Retention (Score)", "Academic Performance (GPA)",
    "Standardized Test Scores"
])

# Add filters only if relevant columns exist
st.sidebar.title("Filters")

# Age filter (if Age column exists)
age_filter = None
if "Age" in df.columns:
    age_min = int(df["Age"].min())
    age_max = int(df["Age"].max())
    age_filter = st.sidebar.slider("Age Range", 
                                  min_value=age_min, 
                                  max_value=age_max,
                                  value=(age_min, age_max))

# Gender filter (if Gender column exists)
gender_filter = None
if "Gender" in df.columns:
    gender_options = df["Gender"].unique().tolist()
    gender_filter = st.sidebar.multiselect("Gender", gender_options, default=gender_options)

# Diagnosis filter (if Diagnoses column exists)
diagnosis_filter = None
if "Diagnoses" in df.columns:
    diagnosis_options = df["Diagnoses"].unique().tolist()
    diagnosis_filter = st.sidebar.multiselect("Diagnosis Type", diagnosis_options, default=diagnosis_options)

# Apply filters to create filtered dataframe
filtered_df = df.copy()

if age_filter and "Age" in df.columns:
    filtered_df = filtered_df[(filtered_df["Age"] >= age_filter[0]) & (filtered_df["Age"] <= age_filter[1])]

if gender_filter and "Gender" in df.columns:
    filtered_df = filtered_df[filtered_df["Gender"].isin(gender_filter)]

if diagnosis_filter and "Diagnoses" in df.columns:
    filtered_df = filtered_df[filtered_df["Diagnoses"].isin(diagnosis_filter)]

# Main content based on selected page
if page == "Dataset Overview":
    st.header("Dataset Overview")
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(10))
    
    # Calculate and display summary statistics for numeric columns
    st.subheader("Summary Statistics")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    st.dataframe(filtered_df[numeric_cols].describe())
    
    # Data completeness
    st.subheader("Data Completeness")
    missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_data / len(filtered_df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
    
    # Only show columns with missing values
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    if not missing_df.empty:
        st.dataframe(missing_df)
        
        # Plot missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_df.index, y='Percentage', data=missing_df)
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Column')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No missing values in the filtered dataset!")

elif page == "Demographics Analysis" and demographic_columns:
    st.header("Demographics Analysis")
    
    # Display available demographic columns
    st.write("Available demographic features:", demographic_columns)
    
    # Age distribution (if available)
    if "Age" in demographic_columns:
        st.subheader("Age Distribution")
        fig = px.histogram(filtered_df, x="Age", nbins=20,
                         title="Age Distribution")
        st.plotly_chart(fig)
    
    # Gender distribution (if available)
    if "Gender" in demographic_columns:
        st.subheader("Gender Distribution")
        gender_counts = filtered_df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig = px.pie(gender_counts, names="Gender", values="Count",
                    title="Gender Distribution")
        st.plotly_chart(fig)
    
    # Socioeconomic status (if available)
    if "Socioeconomic Status" in demographic_columns:
        st.subheader("Socioeconomic Status")
        ses_counts = filtered_df["Socioeconomic Status"].value_counts().reset_index()
        ses_counts.columns = ["Status", "Count"]
        fig = px.pie(ses_counts, names="Status", values="Count",
                    title="Socioeconomic Status Distribution")
        st.plotly_chart(fig)

elif page == "Diagnosis Insights" and diagnosis_columns:
    st.header("Diagnosis Insights")
    
    # Display available diagnosis columns
    st.write("Available diagnosis features:", diagnosis_columns)
    
    # Diagnoses distribution (if available)
    if "Diagnoses" in diagnosis_columns:
        st.subheader("Diagnosis Distribution")
        diag_counts = filtered_df["Diagnoses"].value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Count"]
        fig = px.pie(diag_counts, names="Diagnosis", values="Count",
                   title="Distribution of Neurodiversity Diagnoses")
        st.plotly_chart(fig)
    
    # Age at diagnosis (if available)
    if "Age at Diagnosis" in diagnosis_columns and "Diagnoses" in diagnosis_columns:
        st.subheader("Age at Diagnosis")
        valid_age_diag = filtered_df[filtered_df["Age at Diagnosis"].notna()]
        if not valid_age_diag.empty:
            fig = px.box(valid_age_diag, x="Diagnoses", y="Age at Diagnosis",
                       title="Age at Diagnosis by Diagnosis Type")
            st.plotly_chart(fig)
        else:
            st.write("No age at diagnosis data available with current filters.")

elif page == "SVM Classification":
    st.header("SVM Classification")
    
    st.write("""
    This page uses Support Vector Machine (SVM) classification to analyze patterns in the neurodiversity dataset
    and predict neurodevelopmental conditions based on features you select.
    """)
    
    # Target selection
    st.subheader("Target Selection")
    
    # Option 1: Using "has_" prefix condition columns (if available)
    target_options = []
    if "has_ASD" in df.columns:
        target_options.append("Autism Spectrum (ASD)")
    if "has_ADHD" in df.columns:
        target_options.append("ADHD")
    if "has_Dyslexia" in df.columns:
        target_options.append("Dyslexia")
    
    # Option 2: If no "has_" columns, use Diagnoses column if available
    if not target_options and "Diagnoses" in df.columns:
        diagnoses = df["Diagnoses"].unique().tolist()
        target_options = diagnoses
    
    if not target_options:
        st.error("No suitable target columns found for classification.")
        st.stop()
    
    target_condition = st.selectbox(
        "Select condition to classify:",
        target_options
    )
    
    # Map user selection to column name
    if "has_ASD" in df.columns:
        target_mapping = {
            "Autism Spectrum (ASD)": "has_ASD",
            "ADHD": "has_ADHD",
            "Dyslexia": "has_Dyslexia"
        }
        target_column = target_mapping.get(target_condition, "")
    else:
        # Binary classification: selected diagnosis vs others
        df["target"] = (df["Diagnoses"] == target_condition).astype(int)
        target_column = "target"
    
    # Feature selection
    st.subheader("Feature Selection")
    
    # Get all available numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Remove any target columns from feature list
    feature_cols = [col for col in numeric_cols if col != target_column and not col.startswith("has_")]
    
    # Group features by category
    cognitive_features = get_columns_if_exist(df, [
        "verbal_IQ", "performance_IQ", "full_scale_IQ", "working_memory_index", 
        "processing_speed_index", "reading_score", "phonological_processing_score"
    ])
    
    behavioral_features = get_columns_if_exist(df, [
        "attention_score", "hyperactivity_score", "social_communication_score",
        "Attention Span (minutes)"
    ])
    
    performance_features = get_columns_if_exist(df, [
        "Problem-solving Skills (Score)", "Memory Retention (Score)", 
        "Academic Performance (GPA)", "Standardized Test Scores"
    ])
    
    # Let user select feature groups
    feature_groups = []
    
    if cognitive_features:
        feature_groups.append("Cognitive")
    if behavioral_features:
        feature_groups.append("Behavioral")
    if performance_features:
        feature_groups.append("Performance")
    
    if not feature_groups:
        st.warning("No categorized features found. Using all numeric features instead.")
        selected_features = feature_cols
    else:
        feature_options = st.multiselect(
            "Select feature groups to include:",
            feature_groups,
            default=feature_groups[:2] if len(feature_groups) >= 2 else feature_groups
        )
        
        # Combine selected features
        selected_features = []
        if "Cognitive" in feature_options:
            selected_features.extend(cognitive_features)
        if "Behavioral" in feature_options:
            selected_features.extend(behavioral_features)
        if "Performance" in feature_options:
            selected_features.extend(performance_features)
    
    # Display selected features
    st.write(f"Selected {len(selected_features)} features:")
    st.write(", ".join(selected_features))
    
    # Model parameters
    st.subheader("Model Parameters")
    test_size = st.slider("Test set size (percentage)", 10, 50, 25)
    test_size = test_size / 100  # Convert to proportion
    
    kernel_type = st.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    
    # Function for SVM Classification with error handling
    def run_svm_classification(dataframe, target_col, features, test_size, kernel):
        try:
            # Create a copy of the dataframe
            df_copy = dataframe.copy()
            
            # Filter features to only include columns that exist
            existing_features = [feat for feat in features if feat in df_copy.columns]
            
            if not existing_features:
                return None, "No valid features found for classification."
            
            # Prepare features (X) and target (y)
            X = df_copy[existing_features].copy()
            
            # Ensure target column exists
            if target_col not in df_copy.columns:
                return None, f"Target column '{target_col}' not found in the dataset."
            
            y = df_copy[target_col]
            
            # Handle missing values in features
            for col in X.columns:
                X[col] = X[col].fillna(X[col].median())
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create and train model
            model = SVC(kernel=kernel, probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Prepare results
            results = {
                "model": model,
                "accuracy": accuracy,
                "confusion_matrix": cm,
                "feature_names": existing_features,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "y_pred": y_pred
            }
            
            return results, None
            
        except Exception as e:
            return None, str(e)
    
    # Run classification when button is clicked
    if st.button("Run SVM Classification"):
        with st.spinner("Training and evaluating model..."):
            results, error = run_svm_classification(
                filtered_df, target_column, selected_features, test_size, kernel_type
            )
            
            if error:
                st.error(f"Error during classification: {error}")
            elif results:
                # Display results
                st.subheader("Classification Results")
                
                # Display accuracy
                st.metric("Accuracy", f"{results['accuracy']:.4f}")
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                          xticklabels=['Negative', 'Positive'],
                          yticklabels=['Negative', 'Positive'])
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix for {target_condition}')
                st.pyplot(fig)
                
                # Display features used
                st.subheader("Features Used")
                st.write(f"Successfully used {len(results['feature_names'])} features:")
                st.write(", ".join(results['feature_names']))
                
                # If linear kernel, show feature importance
                if kernel_type == "linear":
                    st.subheader("Feature Importance")
                    
                    # Get coefficients
                    coefficients = results['model'].coef_[0]
                    
                    # Create DataFrame for visualization
                    feature_importance = pd.DataFrame({
                        'Feature': results['feature_names'],
                        'Importance': np.abs(coefficients),
                        'Coefficient': coefficients
                    })
                    
                    # Sort by importance
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
                    
                    # Color based on coefficient sign
                    for i, bar in enumerate(bars):
                        if feature_importance['Coefficient'].iloc[i] < 0:
                            bar.set_color('salmon')
                    
                    ax.set_xlabel('Feature Importance (absolute coefficient value)')
                    ax.set_title('Feature Importance')
                    ax.invert_yaxis()  # Highest importance at the top
                    
                    st.pyplot(fig)
else:
    st.warning("Please select a valid page or ensure relevant data columns are available.") 