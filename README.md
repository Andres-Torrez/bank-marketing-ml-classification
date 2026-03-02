# Bank Marketing — Machine Learning Classification

## Project Objective
This project aims to build a machine learning classification model capable of predicting whether a client will subscribe to a term deposit (*y* variable) using the Bank Marketing dataset.

The solution includes:

- Structured exploratory data analysis (EDA)  
- Reproducible preprocessing and training pipeline  
- Model evaluation and overfitting control  
- Deployment through a Streamlit application  
- Production-ready repository structure  
- Scalable architecture for monitoring and retraining  

---

## Dataset
**Dataset:** Bank Marketing Dataset  

**Target variable:**  
`y ∈ {yes, no}`

**Class distribution:**

- ~88% **no**  
- ~12% **yes**

⚠️ *The dataset is imbalanced. Accuracy is not used as the primary metric.*

---

## Modeling Strategy

### Primary Metric
- **ROC-AUC**

### Secondary Metrics
- F1-score  
- Precision  
- Recall  
- Confusion Matrix  

### Overfitting Control
Overfitting is measured as:

\[
\text{gap} = \text{score}_{train} - \text{score}_{validation}
\]

**Constraint:**  
`gap ≤ 0.05`

---

## 🚨 Data Leakage Consideration
The variable **duration** represents the call duration.

Since this information is only available after the call is completed:

- ❌ It is **excluded** from the production model  
- ✅ It is **optionally included** in a benchmark comparison model  

---

## 🗂 Repository Structure
```
.
├── app/                # Streamlit application
├── data/               # Raw or processed datasets
├── models/             # Saved trained models
├── notebooks/          # EDA and analysis
├── reports/
│   ├── figures/        # Generated visualizations
│   ├── metrics/        # Stored evaluation results
│   └── model_cards/    # Technical reports
├── src/
│   ├── preprocessing/
│   ├── modeling/
│   ├── monitoring/
│   └── utils/
├── tests/              # Unit tests
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Installation
```bash
pip install -r requirements.txt
```

---

## Run Streamlit App
```bash
streamlit run app/app.py
```

---

## Roadmap

### 🟢 Essential
- Baseline model  
- EDA  
- Overfitting < 5%  
- Streamlit app  

### 🟡 Intermediate
- Ensemble models  
- Cross-validation  
- Hyperparameter tuning  
- Feedback logging  

### 🟠 Advanced
- Dockerization  
- Database integration  
- Deployment  
- Unit testing  

---

## Tech Stack
- Python  
- Pandas  
- Scikit-learn  
- Streamlit  
- Git & GitHub  
- Docker  


