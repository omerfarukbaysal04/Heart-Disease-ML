# -*- coding: utf-8 -*-
#Machine Learning Lesson - Heart Disease Prediction
#Dataset: UCI Heart Disease Dataset


# ===============================
# 1. LIBRARIES
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)

sns.set_style("whitegrid")


# ===============================
# 2. LOAD DATA
# ===============================
df = pd.read_csv(r'C:\your_path\Project\heart_disease_uci.csv')
df.columns = df.columns.str.strip()

print("Dataset Shape:", df.shape)


# ===============================
# 3. DROP COLUMNS
# ===============================
df.drop(columns=[c for c in ['id', 'dataset', 'ca', 'thal'] if c in df.columns], inplace=True)


# ===============================
# 4. TARGET
# ===============================
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.drop(columns='num', inplace=True)

print("\nTarget Distribution:")
print(df['target'].value_counts())


# ===============================
# 5. MISSING VALUES
# ===============================
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']

df[numeric_cols] = SimpleImputer(strategy='median').fit_transform(df[numeric_cols])
df[categorical_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[categorical_cols])


# ===============================
# 6. ENCODING
# ===============================
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ===============================
# 7. SPLIT & SCALE
# ===============================
X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ===============================
# 8. BASE MODELS
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest (Base)": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        random_state=42, class_weight='balanced'
    )
}

model_scores = {}

print("\nBASE MODEL RESULTS")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    model_scores[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, preds))


# ===============================
# 9. CROSS-VALIDATION
# ===============================
rf_cv = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)

cv_scores = cross_val_score(
    rf_cv,
    X_train,
    y_train,
    cv=5,
    scoring='roc_auc'
)

print("\nCV ROC-AUC Scores:", cv_scores)
print("Mean ROC-AUC:", cv_scores.mean())


# ===============================
# 10. GRIDSEARCH
# ===============================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 8, 12],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Grid Parameters:")
print(grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)


# ===============================
# 11. FINAL MODEL
# ===============================
best_rf = grid.best_estimator_
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

final_acc = accuracy_score(y_test, y_pred)

print("\nFINAL MODEL REPORT")
print(classification_report(y_test, y_pred))


# ===============================
# 12. CONFUSION MATRIX
# ===============================
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Disease"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Final Random Forest")
plt.show()


# ===============================
# 13. ROC-AUC
# ===============================
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Final Random Forest")
plt.legend()
plt.show()


# ===============================
# 14. FEATURE IMPORTANCE
# ===============================
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(10), x="Importance", y="Feature")
plt.title("Top 10 Feature Importance")
plt.show()


# ===============================
# 15. ALL MODELS PERFORMANCE (TEK GRAFİK)
# ===============================
performance = {
    "Logistic Regression": model_scores["Logistic Regression"],
    "Decision Tree": model_scores["Decision Tree"],
    "Random Forest (Base)": model_scores["Random Forest (Base)"],
    "Random Forest (GridSearch)": final_acc
}

perf_df = pd.DataFrame({
    "Model": performance.keys(),
    "Accuracy (%)": [v * 100 for v in performance.values()]
})

plt.figure(figsize=(10, 6))
sns.barplot(data=perf_df, x="Model", y="Accuracy (%)", palette="viridis")
plt.title("All Models Accuracy Comparison")
plt.ylim(0, 100)

for i, v in enumerate(perf_df["Accuracy (%)"]):
    plt.text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")

plt.xticks(rotation=20)
plt.show()