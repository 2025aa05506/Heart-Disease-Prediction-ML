import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score,
    recall_score, 
    f1_score, 
    matthews_corrcoef,
    confusion_matrix, 
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header"> Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Classification Models Comparison</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Configuration")
st.sidebar.markdown("---")

# Model selection
st.sidebar.subheader("Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a classification model:",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ],
    index=5  # Default to XGBoost
)

# Model descriptions
model_descriptions = {
    "Logistic Regression": "Linear model for binary classification using logistic function",
    "Decision Tree": "Tree-based model that splits data based on feature values",
    "KNN": "Instance-based learning using k-nearest neighbors",
    "Naive Bayes": "Probabilistic classifier based on Bayes' theorem",
    "Random Forest": "Ensemble of decision trees using bagging",
    "XGBoost": "Gradient boosting ensemble method"
}

st.sidebar.info(f"**About:** {model_descriptions[model_choice]}")

st.sidebar.markdown("---")

# Cache resource loading
@st.cache_resource
def load_model(model_name):
    """Load the selected model"""
    model_map = {
        "Logistic Regression": "model/saved_models/logistic_regression.pkl",
        "Decision Tree": "model/saved_models/decision_tree.pkl",
        "KNN": "model/saved_models/knn.pkl",
        "Naive Bayes": "model/saved_models/naive_bayes.pkl",
        "Random Forest": "model/saved_models/random_forest.pkl",
        "XGBoost": "model/saved_models/xgboost.pkl"
    }
    
    model_path = model_map[model_name]
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:  
            return pickle.load(f)
    else:
        st.error(f"Model file not found: {model_path}")
        st.stop()

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    scaler_path = "model/saved_models/scaler.pkl"
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:  
            return pickle.load(f)
    else:
        st.error(f"Scaler file not found: {scaler_path}")
        st.stop()

# File upload section
st.sidebar.subheader("Upload Test Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (test data only)",
    type=['csv'],
    help="Upload a CSV file with the same features as the training data"
)

# Info about sample data
if not uploaded_file:
    st.sidebar.info("**Tip:** Use 'test_data.csv' from the model folder for testing")

# Main content
if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Display dataset info
        st.subheader(" Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Features", df.shape[1] - 1)
        with col4:
            if 'target' in df.columns:
                st.metric("Has Target", "Yes")
            else:
                st.metric("Has Target", "No")
        
        st.markdown("---")
        
        # Show data preview
        with st.expander("View Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                st.dataframe(df.dtypes.astype(str).to_frame('Type'), use_container_width=True)
            with col2:
                st.write("**Missing Values:**")
                st.dataframe(df.isnull().sum().to_frame('Missing'), use_container_width=True)
        
        # Check if target column exists
        if 'target' not in df.columns:
            st.error("Error: 'target' column not found in the uploaded file!")
            st.info("Make sure your CSV file has a 'target' column with the true labels")
            st.stop()
        
        # Separate features and target
        X_test = df.drop('target', axis=1)
        y_test = df['target']
        
        # Load model and scaler
        model = load_model(model_choice)
        scaler = load_scaler()
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Get prediction probabilities if available
        try:
            y_proba = model.predict_proba(X_test_scaled)
        except:
            y_proba = None
        
        # Display results
        st.subheader(f"Model Performance: {model_choice}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Calculate AUC
        try:
            if y_proba is not None:
                if len(np.unique(y_test)) == 2:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            else:
                auc = None
        except:
            auc = None
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
        
        with col2:
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
        
        with col3:
            st.metric("MCC", f"{mcc:.4f}")
            if auc is not None:
                st.metric("AUC", f"{auc:.4f}")
            else:
                st.metric("AUC", "N/A")
        
        st.markdown("---")
        
        # Create two columns for visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues', 
                ax=ax,
                cbar_kws={'label': 'Count'},
                square=True,
                linewidths=1,
                linecolor='black'
            )
            ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {model_choice}', fontsize=14, fontweight='bold', pad=20)
            
            # Set labels
            ax.set_xticklabels(['No Disease (0)', 'Disease (1)'])
            ax.set_yticklabels(['No Disease (0)', 'Disease (1)'])
            
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("Prediction Distribution")
            
            # Create prediction distribution plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Count predictions
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            true_counts = pd.Series(y_test).value_counts().sort_index()
            
            x = np.arange(len(pred_counts))
            width = 0.35
            
            ax.bar(x - width/2, true_counts.values, width, label='True', color='#90EE90', edgecolor='black')
            ax.bar(x + width/2, pred_counts.values, width, label='Predicted', color='#FFB6C6', edgecolor='black')
            
            ax.set_xlabel('Class', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(['No Disease (0)', 'Disease (1)'])
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (t, p) in enumerate(zip(true_counts.values, pred_counts.values)):
                ax.text(i - width/2, t + 1, str(t), ha='center', va='bottom', fontweight='bold')
                ax.text(i + width/2, p + 1, str(p), ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("Detailed Classification Report")
        
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['No Disease', 'Disease'])
        report_df = pd.DataFrame(report).transpose()
        
        # Style the dataframe
        styled_report = report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score'])
        st.dataframe(styled_report, use_container_width=True)
        
        st.markdown("---")
        
        # Model Comparison Section
        st.subheader("All Models Comparison")
        
        # Try to load comparison results
        comparison_file = 'model/model_comparison.csv'
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file, index_col=0)
            
            # Convert AUC to numeric
            comparison_df['AUC'] = pd.to_numeric(comparison_df['AUC'], errors='coerce')
            
            # Highlight current model
            def highlight_current_model(row):
                if row.name == model_choice:
                    return ['background-color: #FFE66D; font-weight: bold'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_comparison = comparison_df.style.apply(highlight_current_model, axis=1)\
                                                   .background_gradient(cmap='RdYlGn', axis=0)
            
            st.dataframe(styled_comparison, use_container_width=True)
            
            st.info(f"**Current Model:** {model_choice} is highlighted in yellow")
            
            # Show best models
            with st.expander("🏅 Best Models Per Metric"):
                col1, col2 = st.columns(2)
                
                metrics_list = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
                
                for idx, metric in enumerate(metrics_list):
                    if metric in comparison_df.columns:
                        best_model = comparison_df[metric].idxmax()
                        best_score = comparison_df[metric].max()
                        
                        if idx % 2 == 0:
                            with col1:
                                st.success(f"**{metric}**: {best_model} ({best_score:.4f})")
                        else:
                            with col2:
                                st.success(f"**{metric}**: {best_model} ({best_score:.4f})")
        else:
            st.warning("Model comparison file not found. Train all models first.")
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.exception(e)

else:
    # Show instructions when no file is uploaded
    st.info("**Please upload a CSV file from the sidebar to begin**")
    
    st.markdown("---")
    
    # Show expected format
    st.subheader("Expected CSV Format")
    
    st.code("""
# Your CSV should have the following columns:
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target

# Example:
63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
    """, language="csv")
    
    st.markdown("---")
    
    # Show sample instructions
    st.subheader("🎯 How to Use This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Select Model**
        - Choose from 6 ML models in the sidebar
        - Each model has different strengths
        
        **Step 2: Upload Data**
        - Upload your test data CSV file
        - Must include 'target' column
        - Use `test_data.csv` for testing
        """)
    
    with col2:
        st.markdown("""
        **Step 3: View Results**
        - See evaluation metrics
        - View confusion matrix
        - Compare with other models
        
        **Step 4: Analyze**
        - Review classification report
        - Check model performance
        - Make informed decisions
        """)
    
    st.markdown("---")
    
    # Show feature descriptions
    with st.expander("Feature Descriptions"):
        st.markdown("""
        | Feature | Description |
        |---------|-------------|
        | **age** | Age in years |
        | **sex** | Gender (1 = male; 0 = female) |
        | **cp** | Chest pain type (0-3) |
        | **trestbps** | Resting blood pressure (mm Hg) |
        | **chol** | Serum cholesterol (mg/dl) |
        | **fbs** | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
        | **restecg** | Resting electrocardiographic results (0-2) |
        | **thalach** | Maximum heart rate achieved |
        | **exang** | Exercise induced angina (1 = yes; 0 = no) |
        | **oldpeak** | ST depression induced by exercise |
        | **slope** | Slope of the peak exercise ST segment (0-2) |
        | **ca** | Number of major vessels (0-4) |
        | **thal** | Thalassemia (0-3) |
        | **target** | Heart disease (0 = no; 1 = yes) |
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><b>Heart Disease Prediction System</b></p>
        <p>Built with Streamlit | M.Tech (AIML/DSE) - ML Assignment 2</p>
        <p>BITS Pilani - Work Integrated Learning Programme</p>
    </div>
    """, unsafe_allow_html=True)