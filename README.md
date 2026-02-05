# Heart Disease Prediction using Machine Learning

This project aims to predict the presence of heart disease using machine learning techniques on a real-world medical dataset.  
The study focuses on data preprocessing, feature engineering, model training, and performance comparison.

---

## Project Members
- [@omerros04](https://github.com/omerros04)
- [@DrXendria](https://github.com/DrXendria)

---

## Project Domain

**Healthcare – Machine Learning**

---

## Motivation and Importance

Heart disease is one of the most critical global health problems.

- Approximately **17 million people die every year** due to cardiovascular diseases.
- This represents nearly **31% of all global deaths**.
- Early diagnosis can increase treatment success by up to **70%** and significantly reduce mortality rates.

With the help of machine learning, disease risk can be evaluated:
- automatically,
- quickly,
- and with high accuracy.

---

## Dataset

**UCI Heart Disease Dataset**

The dataset contains real patient records collected from the following medical centers:

- Cleveland Clinic
- Hungarian Institute of Cardiology
- Switzerland
- Long Beach V.A. Medical Center

### Dataset Characteristics

- Around **920 patient records**
- Includes **14 important medical features**, such as:
  - age
  - cholesterol level
  - maximum heart rate
  - chest pain type
  - resting blood pressure
  - and others

### Data Quality Issues

- The dataset contains missing and noisy values.
- Especially the following features contain a high percentage of missing values:
  - `ca` (number of major vessels)
  - `thal`
- These two columns were excluded from the study.

Source: UCI Heart Disease Data

---

## Data Preprocessing

Data preprocessing is a critical part of this project.

### 1. Missing Value Handling

- **Numerical features** were filled using the **median** strategy.
- **Categorical features** were filled using the **most frequent value (mode)**.

This approach is more robust than using the mean, especially for medical data.

---

### 2. Categorical Data Transformation

Categorical features such as chest pain type were converted into numerical format using:

- **One-Hot Encoding**

---

### 3. Feature Scaling

Because medical features have different units and ranges (for example, age vs cholesterol),  
features were standardized using:

- **StandardScaler**

---

## Target Definition

The original target variable (`num`) was converted into a binary classification task:

- `0` → Healthy
- `1` → Disease

Values greater than 0 were labeled as disease.

---

## Exploratory Findings

The following relationships were observed:

- **Age** and **cholesterol level** show a positive relationship with heart disease risk.
- **Maximum heart rate** shows a negative relationship with heart disease risk.

---

## Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest (base model)
- Random Forest with hyperparameter tuning

---

## Model Training Strategy

- The dataset was split into training and test sets using stratified sampling.
- Standardization was applied only on the training data and then transferred to the test data.
- Class imbalance was handled using `class_weight="balanced"`.

---

## Cross Validation

A 5-fold cross-validation strategy was applied on the Random Forest model using:

- **ROC-AUC** as the evaluation metric.

---

## Hyperparameter Optimization

Grid Search was used to tune the Random Forest model.

Tuned parameters include:

- number of trees (`n_estimators`)
- maximum depth (`max_depth`)
- minimum samples required to split a node (`min_samples_split`)

The final model was selected based on the best ROC-AUC score.

---

## Final Model

The final model is the optimized **Random Forest classifier** obtained via Grid Search.

The following outputs were produced:

- Confusion Matrix
- ROC Curve
- AUC Score
- Feature Importance ranking

---

## Performance Evaluation

All trained models were compared using accuracy scores.

### Important Note

The **Random Forest model** produced the most balanced and reliable results.

Thanks to its ensemble structure, it reduces overfitting and provides better generalization compared to single-tree models.

---

## Feature Importance

The most influential features were extracted from the final Random Forest model and visualized.  
The top 10 features were analyzed to better understand which medical factors contribute most to the prediction.

---

## Technologies and Libraries

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Conclusion

This project demonstrates that machine learning models, especially ensemble methods such as Random Forest, can be used effectively to predict heart disease risk using real clinical data.

With proper preprocessing, feature transformation, and model tuning, reliable and interpretable results can be obtained for healthcare-oriented decision support systems.

---

## Acknowledgment

Thank you for your interest in our project.
