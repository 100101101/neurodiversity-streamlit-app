import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Neurodiversity Dataset Analysis",
    page_icon="🧠",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4a86e8;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #6aa84f;
    }
    .dashboard-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<p class="main-header">Neurodiversity Dataset Analysis</p>', unsafe_allow_html=True)
st.write("""
This dashboard provides interactive analysis and visualizations of a neurodiversity dataset.
Explore various aspects of neurodiversity across different demographics, diagnoses, and life outcomes.
""")

# Load data with caching
@st.cache_data
def load_data():
    # Adjust the path as needed
    data = pd.read_csv("E://FYP//neurodiversity_dataset_no_country.csv")
    # Clean column names: remove extra spaces and convert to title case
    data.columns = data.columns.str.strip().str.title()
    return data

df = load_data()
st.write("DataFrame columns:", df.columns.tolist())

# Sidebar for navigation and filters
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Data Overview", "Demographics Analysis", "Diagnosis Insights", 
     "Performance Metrics", "Support Systems", "Custom Analysis", "Autism Learning Patterns", 
     "Svm Classification", "Confusion Matrix Analysis"]
)

st.sidebar.title("Filters")
st.sidebar.write("Apply these filters to all visualizations")

# Use the cleaned column names for filtering
filter_age = st.sidebar.slider("Age Range", min_value=int(df["Age"].min()), 
                               max_value=int(df["Age"].max()), 
                               value=(int(df["Age"].min()), int(df["Age"].max())))
filter_gender = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=df["Gender"].unique())
filter_diagnosis = st.sidebar.multiselect("Diagnosis Type", df["Diagnoses"].unique(), default=df["Diagnoses"].unique())

# Apply filters
filtered_df = df[
    (df["Age"] >= filter_age[0]) & 
    (df["Age"] <= filter_age[1]) & 
    (df["Gender"].isin(filter_gender)) & 
    (df["Diagnoses"].isin(filter_diagnosis))
]

def check_data_empty(dataframe):
    if dataframe.empty:
        st.warning("No data available with the current filter settings. Please adjust your filters.")
        return True
    return False

# ---------------------- Data Overview Page ----------------------
if page == "Data Overview":
    st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Total Records:** {len(filtered_df)}")
        st.write(f"**Unique Diagnoses:** {filtered_df['Diagnoses'].nunique()}")
    with col2:
        st.write(f"**Age Range:** {filtered_df['Age'].min()} - {filtered_df['Age'].max()} years")
        st.write(f"**Gender Distribution:** {filtered_df['Gender'].value_counts().to_dict()}")
    
    st.subheader("Sample Data")
    st.dataframe(filtered_df.head(10))
    
    st.subheader("Summary Statistics")
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns
    st.dataframe(filtered_df[numeric_cols].describe())
    
    st.subheader("Data Completeness")
    missing_data = filtered_df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_data / len(filtered_df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percent})
    missing_df = missing_df[missing_df['Missing Values'] > 0]
    if not missing_df.empty:
        st.dataframe(missing_df)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=missing_df.index, y='Percentage', data=missing_df)
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values by Column')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("No missing values in the filtered dataset!")

# ---------------------- Demographics Analysis Page ----------------------
elif page == "Demographics Analysis":
    st.markdown('<p class="sub-header">Demographics Analysis</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    st.subheader("Age Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered_df, x="Age", nbins=20, color="Gender", title="Age Distribution by Gender")
        st.plotly_chart(fig)
    with col2:
        fig = px.box(filtered_df, x="Diagnoses", y="Age", title="Age Distribution by Diagnosis Type")
        fig.update_layout(xaxis_title="Diagnosis", yaxis_title="Age")
        st.plotly_chart(fig)
    
    st.subheader("Socioeconomic Status Analysis")
    col1, col2 = st.columns(2)
    with col1:
        ses_count = filtered_df["Socioeconomic Status"].value_counts()
        fig = px.pie(names=ses_count.index, values=ses_count.values, title="Distribution by Socioeconomic Status")
        st.plotly_chart(fig)
    with col2:
        ses_diag = pd.crosstab(filtered_df["Socioeconomic Status"], filtered_df["Diagnoses"])
        fig = px.bar(ses_diag.reset_index(), x="Socioeconomic Status", y=ses_diag.columns, title="Diagnosis Counts by Socioeconomic Status")
        st.plotly_chart(fig)

# ---------------------- Diagnosis Insights Page ----------------------
elif page == "Diagnosis Insights":
    st.markdown('<p class="sub-header">Diagnosis Insights</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    st.subheader("Diagnosis Distribution")
    diag_count = filtered_df["Diagnoses"].value_counts().reset_index()
    diag_count.columns = ["Diagnosis", "Count"]
    fig = px.pie(diag_count, names="Diagnosis", values="Count", title="Distribution of Neurodiversity Diagnoses")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Age at Diagnosis")
    # Note: After cleaning, the column is "Age At Diagnosis"
    diag_age_df = filtered_df[filtered_df["Age At Diagnosis"].notna()]
    if not diag_age_df.empty:
        fig = px.box(diag_age_df, x="Diagnoses", y="Age At Diagnosis", color="Gender", 
                     title="Age at Diagnosis by Diagnosis Type and Gender")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No age at diagnosis data available with the current filters.")
    
    st.subheader("Professional vs. Self-Diagnosis")
    # Use "Type Of Diagnosis" (cleaned) instead of original
    diag_type_df = filtered_df[filtered_df["Type Of Diagnosis"].notna()]
    if not diag_type_df.empty:
        diag_type_count = diag_type_df["Type Of Diagnosis"].value_counts().reset_index()
        diag_type_count.columns = ["Diagnosis Type", "Count"]
        fig = px.bar(diag_type_count, x="Diagnosis Type", y="Count",
                     title="Distribution of Professional vs. Self-Diagnosis",
                     color="Diagnosis Type")
        st.plotly_chart(fig, use_container_width=True)
        diag_by_type = pd.crosstab(diag_type_df["Diagnoses"], diag_type_df["Type Of Diagnosis"])
        fig = px.bar(diag_by_type.reset_index(), x="Diagnoses", y=diag_by_type.columns,
                     title="Diagnosis Type by Neurodiversity Category", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No diagnosis type data available with the current filters.")

# ---------------------- Performance Metrics Page ----------------------
elif page == "Performance Metrics":
    st.markdown('<p class="sub-header">Performance Metrics</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    st.subheader("Academic and Cognitive Performance")
    # Updated performance metric columns to match cleaned names
    perf_cols = ["Attention Span (Minutes)", "Problem-Solving Skills (Score)", 
                 "Memory Retention (Score)", "Academic Performance (Gpa)",
                 "Standardized Test Scores"]
    st.write("Correlation Between Performance Metrics")
    corr = filtered_df[perf_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Between Performance Metrics",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Performance Metrics by Diagnosis")
    metric = st.selectbox("Select Performance Metric", perf_cols)
    fig = px.box(filtered_df, x="Diagnoses", y=metric, title=f"{metric} by Diagnosis Type")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Performance by Learning Preference")
    fig = px.violin(filtered_df, x="Learning Preferences", y=metric, color="Diagnoses", box=True,
                    title=f"{metric} by Learning Preference and Diagnosis")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Impact of Sensory Sensitivities on Performance")
    fig = px.box(filtered_df, x="Sensory Sensitivities", y=metric, title=f"Impact of Sensory Sensitivities on {metric}")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Support Systems Page ----------------------
elif page == "Support Systems":
    st.markdown('<p class="sub-header">Support Systems & Accommodations</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    st.subheader("Types of Support Received")
    # Use "Types Of Support Received" (cleaned)
    support_count = filtered_df["Types Of Support Received"].value_counts().reset_index()
    support_count.columns = ["Support Type", "Count"]
    fig = px.pie(support_count, names="Support Type", values="Count", title="Distribution of Support Types")
    st.plotly_chart(fig, use_container_width=True)
    
    support_diag = pd.crosstab(filtered_df["Diagnoses"], filtered_df["Types Of Support Received"])
    fig = px.bar(support_diag.reset_index(), x="Diagnoses", y=support_diag.columns,
                 title="Support Types by Diagnosis", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Workplace Accommodations")
    accom_count = filtered_df["Workplace Accommodations"].value_counts().reset_index()
    accom_count.columns = ["Accommodation Level", "Count"]
    fig = px.pie(accom_count, names="Accommodation Level", values="Count",
                 title="Distribution of Workplace Accommodation Levels")
    st.plotly_chart(fig, use_container_width=True)
    
    employ_accom = pd.crosstab(filtered_df["Job Type"], filtered_df["Workplace Accommodations"])
    fig = px.bar(employ_accom.reset_index(), x="Job Type", y=employ_accom.columns,
                 title="Employment Status by Accommodation Level", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Impact of Special Education")
    metric = st.selectbox("Select Performance Metric to Compare", 
                          ["Academic Performance (Gpa)", "Standardized Test Scores", 
                           "Problem-Solving Skills (Score)", "Memory Retention (Score)"])
    fig = px.box(filtered_df, x="Special Education Status", y=metric,
                 color="Diagnoses",
                 title=f"Impact of Special Education on {metric} by Diagnosis")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- Custom Analysis Page ----------------------
elif page == "Custom Analysis":
    st.markdown('<p class="sub-header">Custom Analysis</p>', unsafe_allow_html=True)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    st.write("""
    This section allows you to create custom visualizations by selecting variables to compare.
    Choose variables for your x-axis, y-axis, and optional color grouping.
    """)
    
    numeric_cols = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
    all_cols = numeric_cols + categorical_cols
    
    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Bar Chart", "Box Plot", "Violin Plot", "Histogram"])
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis Variable", all_cols)
        color_var = st.selectbox("Select Color Variable (Optional)", ["None"] + categorical_cols)
    with col2:
        y_axis = st.selectbox("Select Y-axis Variable", all_cols)
        facet_var = st.selectbox("Select Facet Variable (Optional)", ["None"] + categorical_cols)
    
    if plot_type == "Scatter Plot":
        if color_var == "None":
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis,
                             title=f"{y_axis} vs {x_axis}")
        else:
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_var,
                             title=f"{y_axis} vs {x_axis} by {color_var}")
    elif plot_type == "Bar Chart":
        if x_axis in categorical_cols:
            if color_var == "None":
                fig = px.bar(filtered_df.groupby(x_axis)[y_axis].mean().reset_index(),
                             x=x_axis, y=y_axis,
                             title=f"Average {y_axis} by {x_axis}")
            else:
                temp_df = filtered_df.groupby([x_axis, color_var])[y_axis].mean().reset_index()
                fig = px.bar(temp_df, x=x_axis, y=y_axis, color=color_var,
                             title=f"Average {y_axis} by {x_axis} and {color_var}",
                             barmode="group")
        else:
            st.error("X-axis should be a categorical variable for bar charts. Please select a different variable.")
            st.stop()
    elif plot_type == "Box Plot":
        if x_axis in categorical_cols:
            if color_var == "None":
                fig = px.box(filtered_df, x=x_axis, y=y_axis,
                             title=f"Distribution of {y_axis} by {x_axis}")
            else:
                fig = px.box(filtered_df, x=x_axis, y=y_axis, color=color_var,
                             title=f"Distribution of {y_axis} by {x_axis} and {color_var}")
        else:
            st.error("X-axis should be a categorical variable for box plots. Please select a different variable.")
            st.stop()
    elif plot_type == "Violin Plot":
        if x_axis in categorical_cols:
            if color_var == "None":
                fig = px.violin(filtered_df, x=x_axis, y=y_axis, box=True,
                                title=f"Distribution of {y_axis} by {x_axis}")
            else:
                fig = px.violin(filtered_df, x=x_axis, y=y_axis, color=color_var, box=True,
                                title=f"Distribution of {y_axis} by {x_axis} and {color_var}")
        else:
            st.error("X-axis should be a categorical variable for violin plots. Please select a different variable.")
            st.stop()
    elif plot_type == "Histogram":
        if x_axis in numeric_cols:
            if color_var == "None":
                fig = px.histogram(filtered_df, x=x_axis,
                                   title=f"Distribution of {x_axis}")
            else:
                fig = px.histogram(filtered_df, x=x_axis, color=color_var,
                                   title=f"Distribution of {x_axis} by {color_var}", barmode="group")
        else:
            st.error("X-axis should be a numeric variable for histograms. Please select a different variable.")
            st.stop()
    
    if facet_var != "None":
        fig.update_layout(facet_col=facet_var)
    
    st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Correlation Analysis") and x_axis in numeric_cols and y_axis in numeric_cols:
        corr_value = filtered_df[[x_axis, y_axis]].corr().iloc[0, 1]
        st.write(f"**Correlation between {x_axis} and {y_axis}:** {corr_value:.3f}")
        if plot_type == "Scatter Plot":
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_var if color_var != "None" else None,
                             trendline="ols", title=f"{y_axis} vs {x_axis} with Regression Line")
            st.plotly_chart(fig, use_container_width=True)
    
    if st.checkbox("Show Data Table"):
        display_cols = []
        for col in [x_axis, y_axis]:
            if col not in display_cols:
                display_cols.append(col)
        if color_var != "None" and color_var not in display_cols:
            display_cols.append(color_var)
        if facet_var != "None" and facet_var not in display_cols:
            display_cols.append(facet_var)
        st.dataframe(filtered_df[display_cols])

# ---------------------- Autism Learning Patterns Page ----------------------
elif page == "Autism Learning Patterns":
    st.markdown('<p class="sub-header">Autism Learning Patterns & Preferences</p>', unsafe_allow_html=True)
    st.write("""
    This section is inspired by V. Kavitha's research on "Immersive learning aid for children with autism (ASD) using object recognition" (2021).
    It explores learning patterns and educational outcomes for individuals with autism compared to other neurodevelopmental conditions.
    """)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    autism_df = filtered_df[filtered_df["Diagnoses"] == "Autism Spectrum"]
    non_autism_df = filtered_df[filtered_df["Diagnoses"] != "Autism Spectrum"]
    
    if autism_df.empty:
        st.warning("No autism spectrum data available with the current filter settings. Please adjust your filters.")
        st.stop()
    
    st.subheader("Learning Preferences for Autism vs. Other Diagnoses")
    col1, col2 = st.columns(2)
    with col1:
        learn_pref_autism = autism_df["Learning Preferences"].value_counts().reset_index()
        learn_pref_autism.columns = ["Learning Preference", "Count"]
        fig = px.pie(learn_pref_autism, names="Learning Preference", values="Count",
                     title="Learning Preferences - Autism Spectrum",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig)
    with col2:
        if not non_autism_df.empty:
            learn_pref_non_autism = non_autism_df["Learning Preferences"].value_counts().reset_index()
            learn_pref_non_autism.columns = ["Learning Preference", "Count"]
            fig = px.pie(learn_pref_non_autism, names="Learning Preference", values="Count",
                         title="Learning Preferences - Other Diagnoses",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig)
        else:
            st.write("No data available for other diagnoses with current filters.")
    
    st.subheader("Academic Performance by Learning Preference for Autism")
    # Update column: "Academic Performance (Gpa)"
    fig = px.box(autism_df, x="Learning Preferences", y="Academic Performance (Gpa)",
                 title="Academic Performance by Learning Preference - Autism Spectrum",
                 color="Learning Preferences")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Learning Preferences & Sensory Sensitivities")
    col1, col2 = st.columns(2)
    with col1:
        autism_learn_sensory = pd.crosstab(autism_df["Learning Preferences"], autism_df["Sensory Sensitivities"], normalize="index") * 100
        fig = px.imshow(autism_learn_sensory,
                        labels=dict(x="Sensory Sensitivity", y="Learning Preference", color="Percentage"),
                        title="Learning Preference vs. Sensory Sensitivity - Autism (%)",
                        text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig)
    with col2:
        autism_learn_support = pd.crosstab(autism_df["Learning Preferences"], autism_df["Types Of Support Received"])
        fig = px.bar(autism_learn_support.reset_index(), x="Learning Preferences",
                     y=autism_learn_support.columns,
                     title="Types of Support by Learning Preference - Autism",
                     barmode="group")
        st.plotly_chart(fig)
    
    st.subheader("Learning Preferences & Test Performance")
    fig = px.violin(autism_df, x="Learning Preferences", y="Standardized Test Scores",
                    box=True, points="all",
                    title="Distribution of Test Scores by Learning Preference - Autism",
                    color="Learning Preferences")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Immersive Learning Approach Simulation")
    st.write("""
    V. Kavitha's research focuses on using object recognition technology to create immersive learning experiences for children with autism.
    Below is a conceptual simulation of how different learning approaches might impact educational outcomes for individuals with autism.
    """)
    learning_prefs = autism_df["Learning Preferences"].unique()
    simulation_data = []
    for pref in learning_prefs:
        pref_data = autism_df[autism_df["Learning Preferences"] == pref]
        avg_academic = pref_data["Academic Performance (Gpa)"].mean()
        avg_problem_solving = pref_data["Problem-Solving Skills (Score)"].mean()
        avg_memory = pref_data["Memory Retention (Score)"].mean()
        avg_test_score = pref_data["Standardized Test Scores"].mean()
        simulation_data.append({
            "Learning Preference": pref,
            "Approach": "Traditional",
            "Performance Score": avg_academic * 25
        })
        immersive_boost = 1.2
        if pref == "Visual":
            immersive_boost = 1.4
        elif pref == "Kinesthetic":
            immersive_boost = 1.5
        elif pref == "Auditory":
            immersive_boost = 1.3
        simulation_data.append({
            "Learning Preference": pref,
            "Approach": "Immersive (Kavitha's Method)",
            "Performance Score": avg_academic * 25 * immersive_boost
        })
    sim_df = pd.DataFrame(simulation_data)
    fig = px.bar(sim_df, x="Learning Preference", y="Performance Score", color="Approach",
                 barmode="group",
                 title="Simulated Impact of Immersive Learning Approach by Learning Preference",
                 color_discrete_sequence=["#636EFA", "#00CC96"])
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Note**: The above simulation is conceptual and based on the principles of V. Kavitha's research on immersive learning using object recognition for autism.
    Actual implementation would require specialized technology and methodology as described in the research paper.
    """)
    
    st.subheader("Recommendations Based on Analysis")
    best_learning_pref = autism_df.groupby("Learning Preferences")["Academic Performance (Gpa)"].mean().idxmax()
    best_support = autism_df.groupby("Types Of Support Received")["Academic Performance (Gpa)"].mean().idxmax()
    st.write(f"""
    Based on the analysis of this dataset and informed by V. Kavitha's research on immersive learning:
    
    1. **Learning preference**: {best_learning_pref} learning appears most effective for individuals with autism in this dataset
    2. **Support type**: {best_support} shows the highest correlation with academic success
    3. **Immersive approach**: An object-recognition based immersive learning approach as researched by Kavitha could potentially enhance learning outcomes, particularly for {best_learning_pref.lower()} learners
    4. **Sensory considerations**: Learning approaches should account for individual sensory sensitivities
    """)

# ---------------------- SVM Classification Page ----------------------
elif page == "Svm Classification":
    st.markdown('<p class="sub-header">SVM Classification of Neurodevelopmental Conditions</p>', unsafe_allow_html=True)
    st.write("""
    This page uses Support Vector Machine (SVM) classification to analyze patterns in the neurodiversity dataset
    and predict neurodevelopmental conditions based on cognitive, behavioral, and demographic features.
    """)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.pipeline import Pipeline
    import numpy as np
    
    def simplified_model_evaluation(filtered_df, condition_column, features_list):
        st.write("Starting simplified model evaluation...")
        existing_features = [feat for feat in features_list if feat in filtered_df.columns]
        missing_features = [feat for feat in features_list if feat not in filtered_df.columns]
        if missing_features:
            st.warning(f"Skipping {len(missing_features)} non-existent columns: {missing_features}")
        X = filtered_df[existing_features].copy()
        numeric_features = [col for col in existing_features if X[col].dtype.kind in 'ifc']
        categorical_features = [col for col in existing_features if col not in numeric_features]
        if categorical_features:
            st.info(f"Removing {len(categorical_features)} categorical features: {categorical_features}")
            X = X[numeric_features]
        st.write(f"Using {len(numeric_features)} numeric features for analysis")
        y = filtered_df[condition_column]
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        if X.empty or len(X.columns) == 0:
            st.error("No valid numeric features remaining for analysis.")
            return None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        st.write(f"Training accuracy: {train_acc:.4f}")
        st.write(f"Testing accuracy: {test_acc:.4f}")
        return model, train_acc, test_acc, X_train, X_test, y_train, y_test
    
    st.subheader("Target Selection")
    target_condition = st.selectbox("Select condition to classify:", ["Autism Spectrum (ASD)", "ADHD", "Dyslexia"])
    target_mapping = {
        "Autism Spectrum (ASD)": "Has_Asd",
        "ADHD": "Has_Adhd",
        "Dyslexia": "Has_Dyslexia"
    }
    target_column = target_mapping[target_condition]
    
    st.subheader("Feature Selection")
    cognitive_features = ["Verbal_Iq", "Performance_Iq", "Full_Scale_Iq", "Working_Memory_Index", "Processing_Speed_Index", "Reading_Score", "Phonological_Processing_Score"]
    behavioral_features = ["Attention_Score", "Hyperactivity_Score", "Social_Communication_Score", "Attention Span (Minutes)"]
    performance_features = ["Problem-Solving Skills (Score)", "Memory Retention (Score)", "Academic Performance (Gpa)", "Standardized Test Scores"]
    demographic_features = ["Age", "Gender", "Socioeconomic Status", "Mother_Has_Thyroid_Autoimmunity", "Job Type", "Special Education Status", "Types Of Support Received", "Workplace Accommodations"]
    
    feature_options = st.multiselect("Select feature groups to include:", ["Cognitive", "Behavioral", "Performance", "Demographic"], default=["Cognitive", "Behavioral"])
    selected_features = []
    if "Cognitive" in feature_options:
        selected_features.extend(cognitive_features)
    if "Behavioral" in feature_options:
        selected_features.extend(behavioral_features)
    if "Performance" in feature_options:
        selected_features.extend(performance_features)
    if "Demographic" in feature_options:
        selected_features.extend(demographic_features)
    
    st.write(f"Selected {len(selected_features)} features:")
    st.write(", ".join(selected_features))
    
    st.subheader("Model Parameters")
    test_size = st.slider("Test set size (percentage)", 10, 50, 25)
    test_size = test_size / 100
    kernel_type = st.selectbox("SVM Kernel", ["linear", "rbf", "poly"])
    debug_mode = st.checkbox("Debug Mode", value=False, help="Use simplified model evaluation to troubleshoot issues")
    
    def run_svm_classification(filtered_df, target_column, selected_features, test_size, kernel_type):
        df_copy = filtered_df.copy()
        existing_features = [feat for feat in selected_features if feat in df_copy.columns]
        missing_features = [feat for feat in selected_features if feat not in df_copy.columns]
        if missing_features:
            st.warning(f"Skipping {len(missing_features)} non-existent columns: {missing_features}")
        X = df_copy[existing_features].copy()
        y = df_copy[target_column]
        numeric_features = [col for col in existing_features if X[col].dtype.kind in 'ifc']
        categorical_features = [col for col in existing_features if col not in numeric_features]
        if categorical_features:
            st.info(f"Removing {len(categorical_features)} categorical features: {categorical_features}")
            X = X[numeric_features]
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())
        if X.empty or len(X.columns) == 0:
            st.error("No valid numeric features remaining for analysis.")
            return None, None, None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel_type, probability=True, random_state=42))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return model, accuracy, cm, report, X_train, X_test, y_train, y_test
    
    if debug_mode:
        st.subheader("Debug Mode")
        st.write("""
        Debug mode uses a simplified model evaluation pipeline that helps identify issues with your data.
        This can be useful when troubleshooting errors with missing columns or categorical data.
        """)
        if st.button("Run Simplified Model Evaluation"):
            with st.spinner("Running simplified model evaluation..."):
                debug_results = simplified_model_evaluation(filtered_df, target_column, selected_features)
                if debug_results is not None:
                    debug_model, train_acc, test_acc, X_train, X_test, y_train, y_test = debug_results
                    y_pred = debug_model.predict(X_test)
                    debug_cm = confusion_matrix(y_test, y_pred)
                    st.subheader("Debug Mode Confusion Matrix")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(debug_cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Negative', 'Positive'],
                              yticklabels=['Negative', 'Positive'])
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Debug Mode Confusion Matrix for {target_condition}')
                    st.pyplot(fig)
                    st.subheader("Features Used")
                    used_features = [col for col in X_train.columns]
                    st.write(f"Successfully used {len(used_features)} features:")
                    st.write(", ".join(used_features))
                    st.success("Simplified model evaluation completed successfully. You can now try the full model.")
                else:
                    st.error("Simplified model evaluation failed. Please check your data and selected features.")
    
    if st.button("Train SVM Model"):
        with st.spinner("Training model..."):
            model_results = run_svm_classification(filtered_df, target_column, selected_features, test_size, kernel_type)
            if model_results is not None and len(model_results) == 8:
                model, accuracy, cm, report, X_train, X_test, y_train, y_test = model_results
                st.subheader("Classification Results")
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.subheader("Confusion Matrix")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write("Confusion Matrix Interpretation:")
                    st.write("- True Negatives (top-left)")
                    st.write("- False Positives (top-right)")
                    st.write("- False Negatives (bottom-left)")
                    st.write("- True Positives (bottom-right)")
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    st.write("\n**Key Metrics:**")
                    st.write(f"Sensitivity (Recall): {sensitivity:.4f}")
                    st.write(f"Specificity: {specificity:.4f}")
                    st.write(f"Precision: {precision:.4f}")
                    st.write(f"Negative Predictive Value: {npv:.4f}")
                with col2:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                              xticklabels=['Negative', 'Positive'],
                              yticklabels=['Negative', 'Positive'])
                    total = np.sum(cm)
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            percentage = cm[i, j] / total * 100
                            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                                    ha='center', va='center', fontsize=9, color='gray')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix for {target_condition}')
                    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                    ax.annotate('True Negatives', xy=(0.25, 0.25), xycoords='axes fraction', bbox=bbox_props, ha='center')
                    ax.annotate('False Positives', xy=(0.75, 0.25), xycoords='axes fraction', bbox=bbox_props, ha='center')
                    ax.annotate('False Negatives', xy=(0.25, 0.75), xycoords='axes fraction', bbox=bbox_props, ha='center')
                    ax.annotate('True Positives', xy=(0.75, 0.75), xycoords='axes fraction', bbox=bbox_props, ha='center')
                    st.pyplot(fig)
                with st.expander("Learn more about confusion matrix metrics"):
                    st.write("""
                    ### Confusion Matrix Metrics Explained
                    
                    - **Sensitivity (Recall)**: TP / (TP + FN)
                    - **Specificity**: TN / (TN + FP)
                    - **Precision**: TP / (TP + FP)
                    - **Negative Predictive Value (NPV)**: TN / (TN + FN)
                    - **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
                    - **F1 Score**: 2 * (Precision * Recall) / (Precision + Recall)
                    """)
                st.subheader("Classification Report")
                report_df = pd.DataFrame(report).transpose()
                if 'support' in report_df.columns:
                    report_df['support'] = report_df['support'].astype(int)
                st.dataframe(report_df)
                st.subheader("Cognitive Profiles by Diagnosis")
                if all(feat in filtered_df.columns for feat in cognitive_features):
                    cognitive_by_diagnosis = filtered_df.groupby('Diagnoses')[cognitive_features].mean().reset_index()
                    diagnoses = cognitive_by_diagnosis['Diagnoses'].tolist()
                    fig = go.Figure()
                    for diagnosis in diagnoses:
                        row = cognitive_by_diagnosis[cognitive_by_diagnosis['Diagnoses'] == diagnosis]
                        values = row[cognitive_features].values.flatten().tolist()
                        values.append(values[0])
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=cognitive_features + [cognitive_features[0]],
                            name=diagnosis
                        ))
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[70, 150]
                            )),
                        title="Cognitive Profiles by Diagnosis",
                        showlegend=True
                    )
                    st.plotly_chart(fig)
                if kernel_type == "linear":
                    st.subheader("Feature Importance")
                    svm_model = model.named_steps['svm']
                    feature_names = selected_features
                    coefficients = svm_model.coef_[0]
                    feature_importance = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': np.abs(coefficients),
                        'Coefficient': coefficients
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
                    for i, bar in enumerate(bars):
                        if feature_importance['Coefficient'].iloc[i] < 0:
                            bar.set_color('salmon')
                    ax.set_xlabel('Feature Importance (absolute coefficient value)')
                    ax.set_title('Feature Importance')
                    ax.invert_yaxis()
                    import matplotlib.patches as mpatches
                    positive_patch = mpatches.Patch(color='skyblue', label='Positive Coefficient')
                    negative_patch = mpatches.Patch(color='salmon', label='Negative Coefficient')
                    ax.legend(handles=[positive_patch, negative_patch], loc='lower right')
                    st.pyplot(fig)
                st.subheader("Key Insights")
                if kernel_type == "linear" and len(feature_importance) > 0:
                    top_features = feature_importance['Feature'].head(3).tolist()
                    st.write(f"**Most important features for {target_condition}:**")
                    for i, feature in enumerate(top_features):
                        st.write(f"{i+1}. {feature}")
                st.write("**Classification Performance:**")
                st.write(f"- Accuracy: {accuracy:.2f}")
                if 'precision' in report_df.loc['1']:
                    st.write(f"- Precision: {report_df.loc['1', 'precision']:.2f}")
                if 'recall' in report_df.loc['1']:
                    st.write(f"- Recall: {report_df.loc['1', 'recall']:.2f}")
                if 'f1-score' in report_df.loc['1']:
                    st.write(f"- F1 Score: {report_df.loc['1', 'f1-score']:.2f}")
                st.info("""
                **Note**: This analysis is for educational purposes only and should not be used for diagnosis.
                The model is trained on a relatively small dataset and may not generalize to all populations.
                """)
            else:
                st.error("Error: Could not train the model. Please ensure all selected features are numeric.")

# ---------------------- Confusion Matrix Analysis Page ----------------------
elif page == "Confusion Matrix Analysis":
    st.markdown('<p class="sub-header">Interactive Confusion Matrix Analysis</p>', unsafe_allow_html=True)
    st.write("""
    This page allows you to create and analyze confusion matrices to evaluate how well a single feature
    can predict neurodevelopmental conditions. A confusion matrix shows the counts of true positives,
    false positives, true negatives, and false negatives when using a feature as a classifier.
    """)
    
    if check_data_empty(filtered_df):
        st.stop()
    
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
    
    st.subheader("Step 1: Select Target Condition")
    target_condition = st.selectbox("Which condition do you want to predict?", ["Autism Spectrum (ASD)", "ADHD", "Dyslexia"])
    target_mapping = {"Autism Spectrum (ASD)": "Has_Asd", "ADHD": "Has_Adhd", "Dyslexia": "Has_Dyslexia"}
    target_column = target_mapping[target_condition]
    
    st.subheader("Step 2: Choose Predictor Feature")
    feature_categories = {
        "Cognitive Features": ["Verbal_Iq", "Performance_Iq", "Full_Scale_Iq", "Working_Memory_Index", "Processing_Speed_Index", "Reading_Score", "Phonological_Processing_Score"],
        "Behavioral Features": ["Attention_Score", "Hyperactivity_Score", "Social_Communication_Score", "Attention Span (Minutes)"],
        "Performance Features": ["Problem-Solving Skills (Score)", "Memory Retention (Score)", "Academic Performance (Gpa)", "Standardized Test Scores"]
    }
    feature_category = st.selectbox("Select feature category", list(feature_categories.keys()))
    prediction_feature = st.selectbox("Select feature to use as predictor", feature_categories[feature_category])
    
    st.subheader("Step 3: Set Classification Threshold")
    feature_min = float(filtered_df[prediction_feature].min())
    feature_max = float(filtered_df[prediction_feature].max())
    feature_mean = float(filtered_df[prediction_feature].mean())
    feature_median = float(filtered_df[prediction_feature].median())
    st.write(f"""
    **Feature Statistics:**
    - Minimum: {feature_min:.2f}
    - Maximum: {feature_max:.2f}
    - Mean: {feature_mean:.2f}
    - Median: {feature_median:.2f}
    """)
    threshold_options = {"Mean": feature_mean, "Median": feature_median, "Custom": None}
    threshold_choice = st.radio("Select threshold method", list(threshold_options.keys()))
    if threshold_choice == "Custom":
        threshold = st.slider("Set custom threshold", min_value=feature_min, max_value=feature_max, value=feature_mean)
    else:
        threshold = threshold_options[threshold_choice]
    
    st.write(f"Using threshold: {threshold:.2f}")
    st.write(f"Individuals with {prediction_feature} ≥ {threshold:.2f} will be classified as having {target_condition}")
    
    if st.button("Generate Confusion Matrix"):
        predictions = (filtered_df[prediction_feature] >= threshold).astype(int)
        actuals = filtered_df[target_column]
        cm = confusion_matrix(actuals, predictions)
        try:
            tn, fp, fn, tp = cm.ravel()
            accuracy = accuracy_score(actuals, predictions)
            precision = precision_score(actuals, predictions, zero_division=0)
            recall = recall_score(actuals, predictions, zero_division=0)
            f1 = f1_score(actuals, predictions, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            st.subheader("Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.3f}")
            col2.metric("Precision", f"{precision:.3f}")
            col3.metric("Recall", f"{recall:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")
            
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Negative", "Positive"],
                        yticklabels=["Negative", "Positive"], ax=ax)
            total = np.sum(cm)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    percentage = cm[i, j] / total * 100
                    ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                            ha='center', va='center', fontsize=9, color='gray')
            plt.xlabel(f'Predicted ({prediction_feature} ≥ {threshold:.2f})')
            plt.ylabel(f'Actual ({target_condition})')
            plt.title(f'Confusion Matrix: Using {prediction_feature} to Predict {target_condition}')
            explanation_texts = [
                f"TN: {tn}\nTrue Negatives\n(Correctly predicted\nnot having {target_condition})",
                f"FP: {fp}\nFalse Positives\n(Incorrectly predicted\nas having {target_condition})",
                f"FN: {fn}\nFalse Negatives\n(Missed actual cases\nof {target_condition})",
                f"TP: {tp}\nTrue Positives\n(Correctly identified\ncases of {target_condition})"
            ]
            positions = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for text, (x, y) in zip(explanation_texts, positions):
                bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
                ax.annotate(text, xy=(x, y), xycoords='axes fraction', 
                            bbox=bbox_props, ha='center', va='center', fontsize=9)
            st.pyplot(fig)
            
            st.subheader("Interpretation")
            if accuracy > 0.7:
                effectiveness = "This feature appears to be a good predictor"
                color = "success"
            elif accuracy > 0.6:
                effectiveness = "This feature is a moderate predictor"
                color = "warning"
            else:
                effectiveness = "This feature is not an effective predictor"
                color = "error"
            st.markdown(f":{color}[**{effectiveness}** of {target_condition}]")
            st.write(f"""
            **What these results mean:**
            
            - **True Positives ({tp})**: Correctly identified {tp} individuals with {target_condition}.
            - **False Positives ({fp})**: Incorrectly classified {fp} individuals without {target_condition}.
            - **False Negatives ({fn})**: Failed to identify {fn} individuals who have {target_condition}.
            - **True Negatives ({tn})**: Correctly identified {tn} individuals without {target_condition}.
            
            **Performance metrics:**
            
            - **Accuracy ({accuracy:.3f})**: Overall proportion of correct predictions.
            - **Precision ({precision:.3f})**: When the feature predicts the condition, how often it's correct.
            - **Recall ({recall:.3f})**: Proportion of actual cases that were detected.
            - **Specificity ({specificity:.3f})**: Proportion of individuals without the condition correctly identified.
            """)
            
            st.subheader("Feature Distribution by Group")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(
                data=filtered_df[filtered_df[target_column] == 1][prediction_feature], 
                label=f"With {target_condition}", shade=True, alpha=0.5
            )
            sns.kdeplot(
                data=filtered_df[filtered_df[target_column] == 0][prediction_feature], 
                label=f"Without {target_condition}", shade=True, alpha=0.5
            )
            plt.axvline(x=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.2f}')
            plt.xlabel(prediction_feature)
            plt.ylabel('Density')
            plt.title(f'Distribution of {prediction_feature} by {target_condition} Status')
            plt.legend()
            st.pyplot(fig)
            
            st.subheader("Optimize Threshold (Optional)")
            if st.checkbox("Would you like to find the optimal threshold?"):
                thresholds = np.linspace(feature_min, feature_max, num=100)
                results = []
                for t in thresholds:
                    preds = (filtered_df[prediction_feature] >= t).astype(int)
                    accuracy = accuracy_score(actuals, preds)
                    precision = precision_score(actuals, preds, zero_division=0)
                    recall = recall_score(actuals, preds, zero_division=0)
                    f1 = f1_score(actuals, preds, zero_division=0)
                    results.append({
                        'threshold': t,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                results_df = pd.DataFrame(results)
                optimum_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
                optimum_f1 = results_df.loc[results_df['f1'].idxmax()]
                fig, ax = plt.subplots(figsize=(10, 6))
                plt.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy')
                plt.plot(results_df['threshold'], results_df['precision'], label='Precision')
                plt.plot(results_df['threshold'], results_df['recall'], label='Recall')
                plt.plot(results_df['threshold'], results_df['f1'], label='F1 Score')
                plt.axvline(x=optimum_accuracy['threshold'], color='green', linestyle='--', alpha=0.5)
                plt.axvline(x=optimum_f1['threshold'], color='purple', linestyle='--', alpha=0.5)
                plt.annotate(f"Max Accuracy: {optimum_accuracy['accuracy']:.3f}\nThreshold: {optimum_accuracy['threshold']:.2f}", 
                             xy=(optimum_accuracy['threshold'], optimum_accuracy['accuracy']),
                             xytext=(10,20), textcoords='offset points',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                plt.annotate(f"Max F1: {optimum_f1['f1']:.3f}\nThreshold: {optimum_f1['threshold']:.2f}", 
                             xy=(optimum_f1['threshold'], optimum_f1['f1']),
                             xytext=(10,-30), textcoords='offset points',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                plt.xlabel('Threshold Value')
                plt.ylabel('Metric Value')
                plt.title('Metrics vs Threshold')
                plt.legend()
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                st.write(f"""
                **Optimum Thresholds:**
                
                - **For maximum accuracy ({optimum_accuracy['accuracy']:.3f})**: {optimum_accuracy['threshold']:.2f}
                - **For maximum F1 score ({optimum_f1['f1']:.3f})**: {optimum_f1['threshold']:.2f}
                
                You may want to try these optimized thresholds for better classification performance.
                """)
        except Exception as e:
            st.error(f"Error calculating confusion matrix: {e}")
            st.write("This might happen if one of the classes has no examples in the current filtered dataset.")
            
st.markdown("---")
st.markdown("*Neurodiversity Dataset Analysis Dashboard* | Created with Streamlit")
