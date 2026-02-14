# Heart Disease Prediction - ML Classification Project

## Problem Statement
Predict the presence of heart disease in patients based on clinical parameters. This is a binary classification problem where the goal is to identify whether a patient has heart disease (target=1) or not (target=0) using various medical attributes.

## Dataset Description

- **Source:** Kaggle - Heart Disease Dataset (johnsmith88/heart-disease-dataset)
- **Dataset Name:** Heart Disease UCI
- **Features:** 13 clinical features + 1 target variable
- **Instances:** 1,025 patient records
- **Target Variable:** Binary classification (0 = No disease, 1 = Disease present)
- **Class Distribution:** 
  - No Disease (0): 499 samples (48.7%)
  - Disease (1): 526 samples (51.3%)

### Feature Description:

1. **age**: Age of the patient (years)
2. **sex**: Gender (1 = male, 0 = female)
3. **cp**: Chest pain type (0-3)
   - 0: Typical angina
   - 1: Atypical angina
   - 2: Non-anginal pain
   - 3: Asymptomatic
4. **trestbps**: Resting blood pressure (mm Hg on admission)
5. **chol**: Serum cholesterol in mg/dl
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
   - 0: Normal
   - 1: ST-T wave abnormality
   - 2: Left ventricular hypertrophy
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment (0-2)
    - 0: Upsloping
    - 1: Flat
    - 2: Downsloping
12. **ca**: Number of major vessels (0-4) colored by fluoroscopy
13. **thal**: Thalassemia
    - 0: Normal
    - 1: Fixed defect
    - 2: Reversible defect

## Models Used

### Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----- |
| Logistic Regression | 0.8098 | 0.9298 | 0.8225 | 0.8098 | 0.8072 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 0.9858 | 0.9854 | 0.9854 | 0.9712 |
| KNN | 0.8634 | 0.9629 | 0.8636 | 0.8634 | 0.8634 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8315 | 0.8293 | 0.8288 | 0.6602 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Note:** Random Forest and XGBoost achieved perfect scores, indicating potential overfitting on test data. In production, cross-validation should be used for more robust evaluation.

### Model Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Provides a solid baseline with 80.98% accuracy. The model shows good AUC (0.9298), indicating strong discriminative ability despite moderate overall accuracy. Well-balanced precision and recall make it suitable for medical applications where both false positives and false negatives are critical. Interpretable coefficients allow understanding feature importance. |
| Decision Tree | Achieves exceptional performance with 98.54% accuracy on test data. While the high accuracy is impressive, there's a risk of overfitting to the training data. The max_depth=10 constraint helps but may not fully prevent memorization. Tree structure provides excellent interpretability showing exact decision rules. Performance validates that ensemble methods build on this foundation. |
| KNN | Shows strong performance with 86.34% accuracy using k=5 neighbors. The distance-based approach benefits significantly from StandardScaler normalization. High AUC (0.9629) indicates robust probability estimates. Computationally intensive for large datasets but provides intuitive "similar patient" based predictions. MCC of 0.7269 shows good correlation between predictions and true labels. |
| Naive Bayes | Achieves 82.93% accuracy despite strong feature independence assumption. Surprisingly competitive given the assumption violations in medical data. Very fast training (instantaneous) and prediction makes it ideal for real-time deployment. Lower MCC (0.6602) compared to tree-based models suggests some classification errors but overall reliable probabilistic predictions. |
| Random Forest | Perfect 100% accuracy on test set indicates likely overfitting. While impressive, this performance may not generalize to new unseen data. The ensemble of 100 trees with max_depth=10 may have memorized the training patterns. Feature importance analysis would be valuable. In production, cross-validation scores would provide more realistic performance estimates. Recommended to add regularization or reduce complexity. |
| XGBoost | Also achieves perfect 100% scores across all metrics, suggesting overfitting similar to Random Forest. The gradient boosting with default parameters may be too aggressive. Despite overfitting concerns, XGBoost's sophisticated boosting algorithm demonstrates its powerful learning capability. For deployment, should retrain with early stopping and cross-validation. The perfect scores validate the model's capacity but require validation on truly unseen data. |

## Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Heart_Disease_Prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# OR
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Download Dataset
```bash
# Run the download notebook
jupyter notebook model/00_download_dataset.ipynb
```

### 2. Explore Data
```bash
# Run data exploration
jupyter notebook model/01_data_exploration.ipynb
```

### 3. Train Models
```bash
# Train all 6 ML models
jupyter notebook model/02_train_models.ipynb
```

### 4. Run Streamlit App Locally
```bash
streamlit run app.py
```

### 5. Access Deployed App
**Live Application:** [Your Streamlit Cloud URL - will be added after deployment]

## Repository Structure

```
Heart_Disease_Prediction/
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
├── model/
│   ├── 00_download_dataset.ipynb      # Dataset download script
│   ├── 01_data_exploration.ipynb      # Exploratory data analysis
│   ├── 02_train_models.ipynb          # Model training and evaluation
│   ├── model_comparison.csv           # Results comparison table
│   ├── test_data.csv                  # Sample test data for app
│   ├── target_distribution.png        # Target class visualization
│   ├── correlation_matrix.png         # Feature correlation heatmap
│   ├── feature_distributions.png      # Feature distribution plots
│   └── saved_models/                  # Serialized trained models
│       ├── scaler.pkl                 # Feature scaler
│       ├── logistic_regression.pkl
│       ├── decision_tree.pkl
│       ├── knn.pkl
│       ├── naive_bayes.pkl
│       ├── random_forest.pkl
│       └── xgboost.pkl
└── data/
    └── heart.csv                       # Original dataset
```

## Key Findings

- **Random Forest and XGBoost achieved perfect scores (100%)** across all metrics, though this suggests potential overfitting
- **Decision Tree** showed excellent performance with 98.54% accuracy, demonstrating strong pattern recognition
- **KNN** achieved 86.34% accuracy with robust AUC of 0.9629 after proper feature scaling
- All models showed strong AUC scores (>0.90), indicating excellent discrimination ability
- **Logistic Regression** provided a reliable baseline with 80.98% accuracy and good interpretability
- **Naive Bayes** achieved 82.93% accuracy despite feature independence assumptions
- Feature scaling significantly improved distance-based models (KNN, Logistic Regression)
- Ensemble methods (Random Forest, XGBoost) consistently achieved the highest scores
- The dataset's balanced class distribution (48.7% vs 51.3%) eliminated need for class balancing
- Perfect scores on test data suggest need for cross-validation in production deployment

## Technologies Used

- **Language:** Python 3.9+
- **ML Framework:** scikit-learn 1.3.2, XGBoost 2.0.3
- **Web Framework:** Streamlit 1.31.0
- **Data Processing:** Pandas 2.1.4, NumPy 1.26.2
- **Visualization:** Matplotlib 3.8.2, Seaborn 0.13.0
- **Model Persistence:** Joblib 1.3.2

## Streamlit App Features

The deployed web application includes:

1. **Model Selection:** Choose from 6 different ML models via dropdown
2. **Data Upload:** Upload CSV test data for predictions
3. **Evaluation Metrics:** Display all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
4. **Confusion Matrix:** Visual representation of classification results
5. **Prediction Distribution:** Compare true vs predicted class distributions
6. **Classification Report:** Detailed per-class performance metrics
7. **Model Comparison:** Side-by-side comparison of all trained models
8. **Interactive Interface:** Real-time model switching and result updates

## Assignment Compliance

✅ **Dataset Requirements:**
- Features: 13 (exceeds minimum of 12)
- Instances: 1,025 (exceeds minimum of 500)
- Type: Binary classification

✅ **Models Implemented:**
1. Logistic Regression ✓
2. Decision Tree Classifier ✓
3. K-Nearest Neighbors ✓
4. Naive Bayes (Gaussian) ✓
5. Random Forest (Ensemble) ✓
6. XGBoost (Ensemble) ✓

✅ **Evaluation Metrics (all 6):**
- Accuracy ✓
- AUC Score ✓
- Precision ✓
- Recall ✓
- F1 Score ✓
- MCC Score ✓

✅ **Streamlit App Features:**
- CSV upload functionality ✓
- Model selection dropdown ✓
- Metrics display ✓
- Confusion matrix visualization ✓

## Author

[Your Name]  
M.Tech (AIML/DSE)  
BITS Pilani - Work Integrated Learning Programme  

**Roll Number:** [Your Roll Number]  
**Email:** [Your Email]

## Acknowledgments

- Dataset: UCI Machine Learning Repository / Kaggle (johnsmith88/heart-disease-dataset)
- BITS Pilani WILP - Machine Learning Course
- Assignment completed on BITS Virtual Lab
- Kagglehub for dataset download functionality

## License

This project is part of academic coursework for BITS Pilani M.Tech program.

## Deployment

**GitHub Repository:** [Your GitHub link - will be added]  
**Live Streamlit App:** [Your Streamlit Cloud link - will be added]

---

**Last Updated:** February 2026