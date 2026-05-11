# Student Math Grade Prediction
### DTSC 670 — Foundations of Machine Learning | Final Project

Regression pipeline to predict students' final math grades (G3, 0–20 scale) using demographic, social, and academic features from two Portuguese high schools.

---

## Project Overview

Early identification of students at risk of poor academic outcomes allows schools to intervene before it's too late. This project builds and evaluates supervised regression models on the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) to predict final math grades, comparing performance with and without mid-year grade features (G1, G2) to simulate different intervention timing scenarios.

---

## Repository Structure

\`\`\`
├── final_project.ipynb
├── data/
│   └── student-mat.csv
├── report/
│   └── DTSC670_ExecutiveSummary_AaronBinder.pdf
└── README.md
\`\`\`

---

## ML Pipeline Summary

The notebook follows the end-to-end checklist from *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (Géron, 3rd ed.) and covers:

| Stage | Details |
|---|---|
| **Data Loading & Splitting** | 80/20 train/test split, `random_state=42` |
| **EDA** | Distribution plots, correlation heatmap, boxplots |
| **Feature Engineering** | Custom `FinalProjectTransformer` — engineers `absences_sum`, conditionally drops G1/G2 |
| **Preprocessing** | Separate pipelines for numeric, categorical, and ordinal features via `ColumnTransformer` |
| **Models Evaluated** | Linear Regression, SVR, Lasso |
| **Validation** | 3-fold cross-validation RMSE |
| **Tuning** | `GridSearchCV` on SVR (`C`, `kernel`, `gamma`) |
| **Final Evaluation** | Test set RMSE and R² for best model |

---

## Key Results

### With G1/G2 (mid-year grades included)

| Model | CV RMSE | Test RMSE | R² |
|---|---|---|---|
| Linear Regression | ~2.1 | — | — |
| Lasso | ~2.2 | — | — |
| **SVR (tuned)** | — | **~2.07** | **0.79** |

### Without G1/G2 (early-intervention scenario)

| Model | CV RMSE | Test RMSE | R² |
|---|---|---|---|
| **SVR (tuned)** | — | **~4.35** | **0.08** |

> Mid-year grades (G1, G2) are by far the strongest predictors of final outcomes. Without them, model performance drops substantially — underscoring the difficulty of early-stage prediction using demographic and social features alone.

---

## Tech Stack

- Python 3.x
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn
- Jupyter Notebook

---

## Dataset

**UCI Student Performance Dataset**
Cortez, P., & Silva, A. (2008). *Using data mining to predict secondary school student performance.* EUROSIS.
[https://archive.ics.uci.edu/dataset/320/student+performance](https://archive.ics.uci.edu/dataset/320/student+performance)

395 student records | 33 features | Target: `G3` (final math grade, 0–20)

---

## Author

**Aaron Binder**  
MS Data Science — Eastern University
