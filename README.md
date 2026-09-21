# Personal Expense & Spending Intelligence System

An end-to-end data analytics, machine learning forecasting, and anomaly detection application built for personal finance management. Developed for the **IBM SkillsBuild Data Analytics with AI Academic Internship**.

---

## 1. Project Overview

The **Personal Expense & Spending Intelligence System** analyzes personal transaction logs to provide actionable spending insights, interactive category breakdowns, predictive monthly expense forecasting, and automated anomaly detection. 

Personal financial tracking often suffers from fragmented spending data, lack of forecasting tools, and undetected high-value anomalies. This system bridges that gap by combining Python data science libraries (`Pandas`, `Seaborn`, `Scikit-learn`) with an interactive `Streamlit` dashboard to empower users to make data-driven financial decisions.

---

## 2. Problem Statement

Individual budget management faces three key challenges:
1. **Lack of Granular Visibility**: Individuals often do not understand where their money goes across categories (e.g., Rent vs. Variable lifestyle expenses).
2. **Unpredictable Monthly Cash Flow**: Traditional budgeting relies on static manual estimates rather than predictive historical trends.
3. **Undetected Financial Anomalies**: Large, non-routine or unusual transactions often pass unnoticed, disrupting monthly savings goals.

This project solves these issues by automating data cleaning, exploratory data analysis, monthly expenditure forecasting, and unsupervised anomaly detection.

---

## 3. Objectives

- **Expense Data Analytics**: Ingest and clean transaction-level expense records, parsing datetime formats and categorizing payment modes.
- **Spending Intelligence**: Generate automated insights into top expenditure categories, payment method preferences, and weekday vs. weekend spending behavior.
- **Future Expense Prediction**: Build and evaluate machine learning regression models (Linear Regression and Random Forest Regressor) to forecast total monthly expenditure.
- **Unsupervised Anomaly Detection**: Implement an Isolation Forest algorithm to flag unusually high or irregular expense transactions.
- **Interactive Web Dashboard**: Deliver a modern, user-friendly Streamlit web application featuring KPI metric cards, Plotly charts, dynamic filtering, and custom budget simulations.

---

## 4. Key Features

- **Data Cleaning & Standardization**: Automatic date parsing, numeric validation, categorical text normalization, and feature extraction (`Year`, `Month`, `Day_of_Week`, `Is_Weekend`).
- **Exploratory Data Analytics**: Summary statistics (Mean, Median, Total, Min, Max), distribution plots, time-series daily trendlines, and payment mode breakdowns.
- **Predictive Machine Learning**: Monthly lag-feature engineering and regression modeling to predict future monthly expenditure.
- **Isolation Forest Anomaly Detection**: Unsupervised machine learning to highlight transaction outliers for user review.
- **Automated Intelligence Engine**: Rule-assisted narrative insights providing executive summaries and concrete financial advice.
- **Streamlit Web Application**: Multi-tab dashboard featuring real-time budget tracking, category donut charts, and interactive prediction calculators.

---

## 5. Dataset

- **Dataset Name**: Personal Expense Transactions Dataset
- **Dataset Source**: Kaggle Public Financial Sample / Academic Expense Dataset
- **Dataset Location**: `data/expense_data.csv` (Raw) -> `data/cleaned_expense_data.csv` (Cleaned)
- **License / Usage**: Open Data / Educational & Research License
- **Timeframe**: January 2023 – December 2024 (24 Months, 861 Transactions)
- **Main Columns Used**:
  - `Date`: Transaction date (`YYYY-MM-DD`)
  - `Category`: Expense category (`Rent`, `Food & Dining`, `Shopping`, `Travel`, `Groceries`, `Utilities`, `Healthcare`, `Entertainment`, `Miscellaneous`)
  - `Amount`: Numerical expenditure in Indian Rupees (₹ INR)
  - `Payment_Method`: Payment mode (`UPI`, `Credit Card`, `Debit Card`, `Net Banking`, `Cash`)
  - `Description`: Transaction narrative / vendor details

---

## 6. Technologies Used

- **Programming Language**: Python 3.13
- **Data Manipulation**: `Pandas`, `NumPy`
- **Data Visualization**: `Matplotlib`, `Seaborn`, `Plotly Express`
- **Machine Learning**: `Scikit-learn` (`LinearRegression`, `RandomForestRegressor`, `IsolationForest`)
- **Model Serialization**: `Joblib`
- **Web Application Framework**: `Streamlit`
- **Development Environment**: `Jupyter Notebook`

---

## 7. Machine Learning

### A. Spending Prediction (Regression)
- **Problem Formulation**: Predict total monthly spending (continuous numerical target in ₹ INR) using historical monthly aggregations, transaction frequency, max single transaction size, and lagged spending indicators.
- **Algorithms Evaluated**:
  - **Linear Regression**: Baseline parametric model.
  - **Random Forest Regressor**: Non-linear ensemble tree regressor (`n_estimators=100`, `random_state=42`).
- **Performance Metric**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and $R^2$ Score. Random Forest achieved superior performance with a Test MAE of **₹1,15,375.94**.

### B. Anomaly Detection (Isolation Forest)
- **Algorithm**: `IsolationForest` (`contamination=0.03`, `random_state=42`).
- **Features Used**: `Amount`, `Month`, `Day`, `Is_Weekend`.
- **Outcome**: Flagged **26 unusual high-value transactions** (e.g., International Vacation Booking, High-end Electronics) totaling **₹19,22,074.99** without requiring labelled historical fraud data.

---

## 8. Project Structure

```text
Personal-Expense-Spending-Intelligence/
│
├── app.py                                         # Streamlit Interactive Web Application
├── Pranav_PersonalExpenseSpendingIntelligence.ipynb # Final Submission Jupyter Notebook
├── README.md                                      # Project Documentation
├── requirements.txt                               # Project Python Dependencies
│
├── data/                                          # Data Directory
│   ├── expense_data.csv                           # Raw Expense Dataset
│   ├── cleaned_expense_data.csv                   # Cleaned Expense Dataset
│   └── expense_data_with_anomalies.csv            # Dataset with Anomaly Flags
│
├── models/                                        # Trained ML Models
│   └── spending_prediction_model.pkl              # Saved Random Forest Regressor Model
│
├── notebooks/                                     # Academic Notebook Directory
│   ├── Personal_Expense_Spending_Intelligence.ipynb # Main Working Notebook
│   └── Pranav_PersonalExpenseSpendingIntelligence.ipynb # Submission Copy
│
├── reports/                                       # Generated High-Res Visualizations
│   ├── category_spending_breakdown.png
│   ├── daily_spending_trend.png
│   ├── monthly_spending_trend.png
│   ├── spending_distribution.png
│   ├── anomaly_scatter_plot.png
│   └── actual_vs_predicted_monthly_spend.png
│
├── screenshots/                                   # Dashboard UI Captures
│   └── dashboard_overview.png
│
└── src/                                           # Modular Source Python Code
    ├── data_loader.py                             # Data Loading & Validation Module
    ├── data_cleaner.py                            # Datetime & Cleaning Pipeline
    ├── eda.py                                     # Statistical Aggregations Module
    ├── predictor.py                               # ML Model Training & Forecasting
    ├── anomaly_detector.py                        # Isolation Forest Anomaly Engine
    └── insights.py                                # Automated Intelligence Engine
```

---

## 9. Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pranavdharme2005-lgtm/Personal-Expense-Spending-Intelligence.git
   cd Personal-Expense-Spending-Intelligence
   ```

2. **Create and Activate a Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 10. Run the Notebook

Launch Jupyter Notebook to view and execute the academic submission workflow:

```bash
jupyter notebook Pranav_PersonalExpenseSpendingIntelligence.ipynb
```

Alternatively, open `notebooks/Personal_Expense_Spending_Intelligence.ipynb` inside VS Code or JupyterLab.

---

## 11. Run the Streamlit Dashboard

Launch the interactive web dashboard using Streamlit:

```bash
streamlit run app.py
```

Once executed, open your browser at `http://localhost:8501`.

---

## 12. Currency

All monetary values displayed across the Streamlit dashboard, Jupyter Notebooks, tables, and visualization charts are formatted in **Indian Rupees (₹ / INR)**.

---

## 13. Results & Insights

- **Total Expenditures**: Over 24 months (2023–2024), total spending reached **₹58,50,831.02** across 861 transactions.
- **Primary Cost Driver**: Fixed housing rent accounted for **44.29%** of total spending (**₹25,91,260.00**).
- **Payment Method Preference**: Credit Cards and UPI accounted for over 65% of total transaction volume.
- **Predictive Forecast**: The system projects a total monthly expenditure of **₹4,78,535.56** for January 2025.
- **Anomaly Flags**: Isolation Forest successfully identified 26 outlier transactions representing **₹19,22,074.99** in total spend.

---

## 14. Future Scope

- **Real-Time Bank API Integration**: Connect with open-banking APIs for automated daily expense ingestion.
- **Sub-Category Granular Forecasting**: Train hierarchical time-series models to predict category-wise budgets (e.g., Food vs. Travel separately).
- **Personalized Savings Goals**: Recommend automated monthly savings targets based on predicted discretionary spend.
- **Mobile-Friendly Deployment**: Containerize application using Docker and deploy to AWS / Streamlit Community Cloud for mobile accessibility.

---

## 15. Author

**Pranav**  
*IBM SkillsBuild Data Analytics with AI Academic Internship*
