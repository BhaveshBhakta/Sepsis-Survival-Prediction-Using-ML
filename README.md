## Sepsis Survival Prediction

### Project Overview

This project aims to predict the **hospital outcome (survival or death)** for patients diagnosed with sepsis, utilizing a minimal set of clinical records. By leveraging features like age, sex, and episode number, the goal is to develop a machine learning model that can assess the likelihood of survival, which is critical for early intervention and improving patient outcomes in critical care.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Sepsis Survival Minimal Clinical Records](https://www.kaggle.com/datasets/joebeachcapital/sepsis-survival-minimal-clinical-records)
  * **Size**: 110204 entries (initial), 4 columns. After dropping duplicates and `SMOTE` application, the dataset size changes. The described dataframe for EDA indicates a smaller sample (e.g., after `df.drop_duplicates()` there are 1511 entries before SMOTE).
  * **Key Features**:
      * age\_years, sex\_0male\_1female, episode\_number
  * **Approach**:
      * Data Cleaning: Renamed columns for clarity. Dropped duplicates (from original 110204 entries, down to 1511 unique rows of features). No missing values.
      * Exploratory Data Analysis: Histograms, Boxplots, and Heatmaps were used for visualization.
      * Handling Class Imbalance with `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data. This is crucial as the original dataset has a significant imbalance (975 alive vs 536 dead after deduplication).
      * Binary Classification: The target variable `hospital_outcome_1alive_0dead` (renamed to `dead`) indicates survival (1: alive, 0: dead).
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 63.8% with AdaBoost Classifier.
      * 62.6% with Ridge Classifier.
      * 62.3% with SVC.
      * 62.1% with Logistic Regression.
      * Note: The relatively moderate accuracies suggest the challenge of predicting sepsis survival with a minimal set of clinical records.

-----

### Purpose and Applications

  * Assist healthcare providers in **identifying sepsis patients at higher risk of mortality** for targeted interventions.
  * Support clinical decision-making in emergency and intensive care settings.
  * Potentially inform resource allocation and patient monitoring strategies in hospitals.
  * Contribute to ongoing efforts to improve sepsis management and reduce mortality rates.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Sepsis-Survival-Prediction-Using-ML.git
cd Sepsis-Survival-Prediction-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Improving model performance through advanced hyperparameter tuning and exploring different model architectures.
  * Investigating the impact of different resampling strategies or ensemble methods on class imbalance.
  * Exploring the potential benefits of incorporating more clinical features if available (e.g., vital signs, lab results, comorbidities).
  * Adding explainability (e.g., SHAP or LIME) to understand which factors most influence survival prediction.
