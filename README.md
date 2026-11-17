# Fetal Health Classification 🩺

> A leakage-safe, imbalanced **multiclass classification** pipeline that predicts fetal health state (*Normal · Suspect · Pathological*) from cardiotocogram (CTG) exam data, to support automated prenatal screening.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview

This project builds and evaluates a machine-learning model that classifies the health status of a fetus from **21 numeric measurements** extracted from a **cardiotocogram (CTG)** — a non-invasive prenatal exam recording fetal heart rate and uterine contractions.

The target variable has three classes:

| Code | Class | Meaning |
|:---:|:---|:---|
| 1 | **Normal** | Reassuring trace |
| 2 | **Suspect** | Requires attention / monitoring |
| 3 | **Pathological** | Likely fetal distress |

The full analysis — from exploratory data analysis to model tuning — lives in a single, self-contained Jupyter notebook.

---

## 💼 Business Problem

CTG interpretation is traditionally manual, time-consuming, and dependent on specialist availability. A reliable automated classifier can:

- **Speed up prenatal screening** and reduce the manual review burden on clinicians.
- **Extend access to quality screening in under-resourced areas** where specialists are scarce.
- **Prioritise high-risk cases** by flagging *Suspect* and *Pathological* traces for urgent review.

Because this is a medical screening task, **missing a pathological case (a false negative) is far costlier than a false alarm.** This drives the central modelling decision of the project: optimise for a metric that rewards correct detection of the rare, dangerous classes — not raw accuracy.

---

## 📊 Dataset

- **2,126** CTG exams, each labelled by expert obstetricians.
- **21 numeric features** (baseline FHR, accelerations, decelerations, short/long-term variability, histogram statistics, …).
- **1 target**: `fetal_health` ∈ {1, 2, 3}.
- **Strongly imbalanced**: ~78% Normal, ~14% Suspect, ~8% Pathological.
- No missing values.

The dataset (`fetal_health.csv`) is the well-known public *Fetal Health Classification* CTG dataset.

---

## 🔬 Methodology / Pipeline

The notebook follows a rigorous, leakage-aware workflow:

1. **Train/test split first** — 80/20, **stratified**, performed *before any analysis* so the test set stays untouched and unbiased.
2. **Exploratory Data Analysis** (on the training set only): missing-value check, class-balance analysis, outlier detection (IQR, per feature), and a correlation matrix.
3. **Standardization** — `StandardScaler` fitted **only on the training set**, then applied to the test set.
4. **Feature selection** — two methods compared: **Mutual Information** vs **Chi-square** (`SelectKBest`), selecting the top-4 features.
5. **Metric selection** — **F1-macro** chosen and justified over accuracy for this imbalanced medical task.
6. **Modelling — two parallel "roads":**
   - **Road 1:** all 21 features.
   - **Road 2:** only the 4 most informative features (performance vs. interpretability trade-off).
   - For each road: a **StratifiedKFold spot-check** across 3 models → **GridSearchCV** hyperparameter tuning of the best two.
7. **Final evaluation** on the held-out test set, with confusion matrices and per-class reports.

**Models compared:** Logistic Regression · Random Forest · K-Nearest Neighbors.

---

## 🏆 Results

**Spot-check (5-fold CV, F1-macro) — Road 1:** Random Forest clearly led (≈0.90) over Logistic Regression (≈0.77) and KNN (≈0.76).

**Final test-set performance (F1-macro):**

| Model | Features | F1-macro |
|:---|:---:|:---:|
| 🥇 **Random Forest** | **All 21** | **0.88** |
| Random Forest | Top 4 | 0.83 |
| KNN | Top 4 | 0.83 |
| Logistic Regression | All 21 | 0.75 |

**Winning model: Random Forest** (`class_weight='balanced'`, all features)

| Metric | Value |
|:---|:---:|
| F1-macro | **0.88** |
| Accuracy | ~0.93 |
| **Recall — Pathological** | **0.91** |

The winner correctly identifies **91% of pathological cases** — precisely the clinical objective. Road 2 (only 4 features) reaches ~0.83 F1-macro, making it a strong, simpler, more interpretable alternative.

---

## 📁 Project Structure

```
fetal-health-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── fetal_health.csv                     # Dataset (kept in root: notebook reads it by relative path)
├── fetal_health_analysis.ipynb          # Clean analysis notebook (no outputs)
├── fetal_health_analysis_executed.ipynb # Executed notebook — renders with all plots & outputs on GitHub
└── docs/
    ├── presentation.pptx                # Slide deck (executive summary)
    └── project-report.pdf               # Written project report
```

> 💡 To read the analysis **without running anything**, open `fetal_health_analysis_executed.ipynb` directly on GitHub — it displays every chart and result inline.

---

## ▶️ How to Run / Reproduce

**Prerequisites:** Python 3.12 (3.10+ should work).

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd fetal-health-classification

# 2. (Recommended) create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter and open the notebook
jupyter notebook fetal_health_analysis.ipynb
```

Run all cells top-to-bottom. The notebook is fully reproducible (fixed `random_state`) and was verified end-to-end on Python 3.12 / scikit-learn 1.9.0 with **0 errors**.

---

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Data:** pandas, NumPy
- **ML:** scikit-learn (model selection, preprocessing, feature selection, metrics)
- **Visualization:** matplotlib, seaborn
- **Environment:** Jupyter Notebook

---

## 🎯 Key Takeaways / Skills Demonstrated

- End-to-end supervised ML workflow with **data-leakage prevention** (split-first, train-only scaling, CV-only tuning).
- Handling **class imbalance** through metric choice (F1-macro) and `class_weight='balanced'` — not naïve resampling.
- **Feature selection** with two independent methods (Mutual Information vs Chi-square) and critical comparison.
- **Model comparison + hyperparameter tuning** with `StratifiedKFold` cross-validation and `GridSearchCV`.
- Translating **clinical cost of errors** into a concrete modelling objective.

---

## 🚀 Possible Improvements

Future work to push this from a solid analysis toward a production-grade study:

- Add a **`DummyClassifier` baseline** to anchor the comparison to an explicit quantitative floor.
- Record **training/inference time** per model as a decision dimension.
- Strengthen **reproducibility**: global seeds + a **seed-stability analysis** over several seeds.
- Add a **Gradient Boosting** model (XGBoost / LightGBM) to the comparison.
- Enrich the EDA with **per-class feature distributions** (violin / overlapping histograms).
- Add **SHAP** explainability and **threshold tuning** to further boost pathological-case recall.

---

## 📝 Note on Language

The **analysis notebook and presentation are written in Italian**; this README documents the project in English for portfolio accessibility. The code, variable names, and results are universal.

---

## 👤 Author

**Oskar Les** — Data Science project (Start2Impact).

## 📄 License

Released under the **MIT License** — see [`LICENSE`](LICENSE).
