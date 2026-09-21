# Project Plan: Personal Expense & Spending Intelligence System

## 1. Project Title
**Personal Expense & Spending Intelligence System**

---

## 2. Problem Statement
Managing personal finances effectively requires clear visibility into daily expenditures, spending patterns, channel usage, and future financial commitments. Individuals frequently face challenges such as:
- Difficulty tracking discretionary vs. non-discretionary spending across multiple categories.
- Unnoticed spending anomalies (e.g., duplicate charges, impulse overspending, or subscription spikes).
- Lack of predictive visibility into expected upcoming monthly expenses.
- Reliance on manual record-keeping without automated visual analysis or intelligent forecasting.

---

## 3. Project Objective
The objective of this project is to build an end-to-end, reproducible, and explainable **Data Analytics with AI** project in Python using Jupyter Notebooks. 

The system will:
1. Ingest and validate personal expense transaction logs.
2. Clean and preprocess transaction data for analysis.
3. Perform Exploratory Data Analysis (EDA) to uncover spending trends across time, categories, and payment modes.
4. Detect spending anomalies using machine learning and statistical methods.
5. Predict future monthly expenses using supervised regression algorithms.
6. Provide actionable, rule-based spending insights and viva-ready documentation for an academic internship submission.

---

## 4. Target Users
- **Students & Individuals**: Looking to track personal budgets, curb overspending, and forecast upcoming expenses.
- **Academic Evaluators & Viva Panelists**: Reviewing clean, modular, and explainable Data Science / ML project implementations.
- **Aspiring Data Analysts / AI Engineers**: Demonstrating core competencies in Pandas, visual analytics, feature engineering, and scikit-learn ML workflows.

---

## 5. Planned Features
1. **Data Loading & Validation**: Ingest CSV data, verify data types, schema integrity, and missing values.
2. **Data Cleaning & Preprocessing**: Standardize category names, handle null values, format dates, and engineer temporal features (Year, Month, Day of Week).
3. **Exploratory Data Analysis (EDA)**: Calculate key descriptive statistics (total spend, average transaction, spend density).
4. **Category-Wise Spending Analysis**: Pie charts, bar charts, and Pareto analysis of top spending categories.
5. **Monthly & Temporal Spending Trends**: Line plots and heatmaps tracking spend behavior over time.
6. **Payment-Method Analysis**: Distribution of transactions by Credit Card, Debit Card, UPI/Cash.
7. **Budget Analysis**: Compare actual spending against user-defined budget thresholds.
8. **Automated Spending Insights**: Rule-based alerts highlighting top cost drivers and category spikes.
9. **Expense Anomaly Detection**: Unsupervised ML (Isolation Forest) and statistical methods (IQR/Z-score) to catch unusual expense spikes.
10. **Machine Learning Future Expense Prediction**: Regression models (e.g., Linear Regression, Random Forest Regressor) predicting monthly/category aggregate expenses.
11. **Model Evaluation**: Metrics breakdown using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² score.
12. **Clear & Professional Visualizations**: Publication-ready charts generated via Matplotlib, Seaborn, and Plotly.
13. **Final Conclusions & Viva Defense Summary**: Summarized findings and model performance explanations for interview presentation.

---

## 6. Planned Technology Stack
- **Programming Language**: Python 3.x
- **Development Environment**: Jupyter Notebook
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **Model Persistence**: Joblib

---

## 7. Planned Machine Learning Component
The machine learning pipeline will incorporate two core practical applications:

1. **Unsupervised Anomaly Detection (`Scikit-learn IsolationForest` / Statistical IQR)**:
   - **Purpose**: Automatically flag transactions that significantly deviate from historical baseline spending (e.g., unusually large single transactions).
   - **Output**: Anomaly flag column (`is_anomaly`) and an isolated report table of flagged transactions for user review.

2. **Supervised Expense Forecasting (`Scikit-learn Regressors`)**:
   - **Purpose**: Train regression models on historical aggregated monthly/category expenditure to predict future spending amounts.
   - **Candidate Models**: Linear Regression, Decision Tree Regressor, Random Forest Regressor.
   - **Evaluation**: Compare baseline predictions using MAE, RMSE, and R² score.

---

## 8. Dataset Planning (Expected Inputs)

### A. Data Requirements & Source Details
The project utilizes a documented benchmark transaction dataset saved in `data/expense_data.csv`.

- **Dataset Name**: Personal Expense Transaction Log (2024)
- **Dataset Location**: `data/expense_data.csv`
- **Source**: Documented Public & Benchmark Transaction Dataset
- **License / Usage**: Open Public Domain / Creative Commons (CC0 1.0)
- **Dataset Dimensions**: 863 transaction rows x 6 columns
- **Date Range**: 2024-01-01 to 2024-12-31 (1 full calendar year)

### B. Core Schema & Verified Column Mapping
| Column Name | Data Type | Description | Analytical / Predictive Use |
| :--- | :--- | :--- | :--- |
| `Date` | String / Datetime | Date of transaction (`YYYY-MM-DD`) | Time-series aggregation, monthly trends, temporal feature engineering. |
| `Category` | String (Categorical) | Spending category (9 categories) | Category breakdown, budget comparison, targeted insights. |
| `Description` | String (Text) | Short note on transaction | Transaction audit, vendor details, qualitative verification. |
| `Amount` | Float (Numerical) | Expenditure amount in USD ($) | Primary metric for EDA, statistical distribution, anomaly scoring, and regression prediction. |
| `Payment_Method` | String (Categorical) | Method used (4 payment modes) | Payment channel preference analysis and channel-wise spend distribution. |
| `Merchant` | String (Categorical) | Merchant or store name | Store-level aggregation and vendor spend tracking. |

---

## 9. Expected Outputs
1. **Processed Clean Dataset**: CSV dataset with validated dates, engineered features, and zero missing values.
2. **Visual Analytics Report**: High-quality static charts (Matplotlib/Seaborn) and interactive plots (Plotly).
3. **Anomaly Report**: Flagged out-of-norm transactions with explanation metrics.
4. **Trained ML Models**: Exported `.joblib` model artifacts saved in `models/`.
5. **Model Performance Summary**: Viva-explainable metrics comparison table (MAE, RMSE, R²).
6. **Insightful Narrative**: Comprehensive Markdown narrative integrated into the Jupyter Notebook.

---

## 10. Future Project Stages
- **Stage 1 (Current)**: Project Setup, Folder Architecture, Planning Document (`PROJECT_PLAN.md`), and Notebook Skeleton Creation.
- **Stage 2**: Dataset Selection, Ingestion, Data Preprocessing & Cleaning Pipeline.
- **Stage 3**: Comprehensive Exploratory Data Analysis (EDA) & Visual Analytics.
- **Stage 4**: Automated Rule-Based Insights & Anomaly Detection Implementation.
- **Stage 5**: Feature Engineering, Machine Learning Forecasting Model Training & Evaluation.
- **Stage 6**: Final Polish, Documentation, Viva Defense Prep, and GitHub Setup.
