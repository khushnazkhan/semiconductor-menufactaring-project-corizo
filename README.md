# Capstone Project 2: Semiconductor Yield Classification

## 📌 Project Overview

This project aims to predict the **Pass/Fail yield** of semiconductor wafers based on hundreds of signal sensor features collected during the manufacturing process. The classification will help identify failed units early, reduce production costs, and optimize manufacturing decisions.

---

## 🧠 Problem Statement

Semiconductor fabrication involves collecting hundreds of signal readings. Not all signals contribute equally to the quality (yield) of the final product.

**Objective**:  
- Build a machine learning model that predicts whether a wafer will **Pass (-1)** or **Fail (1)** based on signal sensor data.

---

## 📁 Dataset Description

- **File**: `signal-data.csv`
- **Samples**: 1567 rows
- **Features**: 591 columns (signal features)
- **Target column**: `Pass/Fail`
  - `-1` → Pass  
  - `1` → Fail  
- **Missing values**: Present in several columns

---

## 🔧 Steps Performed

### 1. 📥 Data Loading & Cleaning
- Dropped non-informative `Time` column
- Removed features with >50% missing data
- Filled remaining missing values using median

### 2. 📊 Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Histograms/KDE for distributions
- **Bivariate Analysis**: Boxplots vs Pass/Fail
- **Multivariate Analysis**: Correlation heatmaps
- **Outlier Removal**: Applied IQR method
<img width="1600" height="860" alt="bai" src="https://github.com/user-attachments/assets/f12b0212-e2bb-486e-8c05-996389b9863a" />
<img width="1600" height="860" alt="heat" src="https://github.com/user-attachments/assets/8aef67d5-556d-4b1f-aca8-673fb199b687" />

<img width="1600" height="860" alt="uni" src="https://github.com/user-attachments/assets/61b129d1-1e79-47ed-a804-7b7576915f63" />


### 3. ⚖️ Preprocessing
- Handled class imbalance using **SMOTE**
- Scaled features using **StandardScaler**
- Split data into **Train/Test (80/20)**

### 4. 🤖 Model Training & Tuning
Trained the following 3 models:
- **Random Forest** (with GridSearchCV tuning)
- **SVM** (with GridSearchCV tuning)
- **Naive Bayes** (default parameters)

### 5. 📈 Model Evaluation
- Evaluated using **Accuracy, Precision, Recall, F1-Score**
- Used **classification reports** and **confusion matrices**
- Compared performance and selected **Random Forest** as best

### 6. 💾 Model Saving
- Saved final model using `joblib`
- Saved scaler for future inference

---
<img width="1600" height="860" alt="save moddl" src="https://github.com/user-attachments/assets/ef4acd11-6ef1-4d64-bb2a-1e93d4d5b00b" />


## ✅ Final Results

| Model         | Accuracy | F1-Score | Precision | Recall |
|---------------|----------|----------|-----------|--------|
| Random Forest | ✅ Best  | ✅ Best  | ✅ Best   | ✅ Best |
| SVM           | Medium   | Medium   | Medium    | Low    |
| Naive Bayes   | Low      | Low      | Low       | High   |

---

## 🔚 Conclusion

- Random Forest outperformed others due to its robustness with high-dimensional and noisy data.
- Outlier removal and SMOTE significantly improved performance.
- Feature importance or PCA can be used for future optimization.

---

## 📦 Tools & Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib

---

## 📁 Project Structure

