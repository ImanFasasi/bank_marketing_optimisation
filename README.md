# Bank Marketing Optimisation using Machine Learning

This project applies machine learning to improve the effectiveness of term deposit marketing campaigns for a Portuguese bank. It includes both classification and clustering tasks to predict customer behavior and segment client profiles, helping the bank design more efficient, data-driven marketing strategies.

---

## Project Objectives

- 🎯 **Classification**: Predict whether a client will subscribe to a term deposit.
- 🧠 **Clustering**: Group clients into segments based on similar characteristics.
- 🚀 **Deployment**: Provide a user-friendly interface for real-time predictions using Streamlit.

---

## Dataset Overview

- **Source**: UCI Bank Marketing Dataset
- **Records**: ~41,000
- **Features**: Demographics, call duration, previous outcomes, economic indicators
- **Target Variable**: `subscribed` (yes or no)

---

## Techniques Used

- Feature Engineering (e.g., `call_effectiveness`, `age_group`)
- Handling class imbalance with SMOTE
- PCA for dimensionality reduction
- Model Ensemble: Random Forest, XGBoost, LightGBM
- Clustering: K-Means, Hierarchical Clustering, DBSCAN
- Evaluation: Accuracy, F1 Score, ROC-AUC, Silhouette Score

---

## Run the App Locally

To launch the Streamlit app for real-time predictions:

```bash
streamlit run app/app.py
```

---

## Results

- **Classification Accuracy**: 94.8%
- **F1 Score**: 92.0%
- **Best Model**: Voting Classifier (Random Forest + XGBoost + LightGBM)
- **Clustering Insight**: Four distinct customer segments for marketing personalization

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ImanFasasi/bank_marketing_optimisation.git
cd bank_marketing_optimisation
pip install -r requirements.txt
```

---

## 👤 Author

**Iman Fasasi**  
GitHub: [@ImanFasasi](https://github.com/ImanFasasi)
