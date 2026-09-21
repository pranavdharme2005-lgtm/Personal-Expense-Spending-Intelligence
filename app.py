import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Personal Expense & Spending Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model Loaders with Streamlit Caching
@st.cache_data
def load_clean_dataset():
    data_path = os.path.join('data', 'cleaned_expense_data.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'cleaned_expense_data.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_data
def load_anomaly_dataset():
    data_path = os.path.join('data', 'expense_data_with_anomalies.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join('..', 'data', 'expense_data_with_anomalies.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

@st.cache_resource
def load_ml_model():
    model_path = os.path.join('models', 'spending_prediction_model.pkl')
    if not os.path.exists(model_path):
        model_path = os.path.join('..', 'models', 'spending_prediction_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load Datasets & Model
df = load_clean_dataset()
df_anom = load_anomaly_dataset()
model = load_ml_model()

# Sidebar Navigation & Information
st.sidebar.title("💳 Expense Intelligence")
st.sidebar.markdown("**Academic Internship Submission Dashboard**")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📌 Overview",
        "📊 Spending Analytics",
        "🤖 Future Prediction",
        "⚠️ Anomaly Detection",
        "💡 Insights & Budget"
    ]
)

st.sidebar.markdown("---")

# Optional Sidebar Budget Input
st.sidebar.subheader("⚙️ Quick Settings")
user_budget_sidebar = st.sidebar.number_input(
    "Monthly Target Budget (₹):",
    min_value=10000.0,
    max_value=2000000.0,
    value=500000.0,
    step=25000.0
)

with st.sidebar.expander("ℹ️ Dataset Information"):
    if df is not None:
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Date Range:** 2024-01-01 to 2024-12-31")
        st.write(f"**Categories:** {df['Category'].nunique()}")
        st.write(f"**Payment Methods:** {df['Payment_Method'].nunique()}")
        st.write(f"**Currency:** INR (₹)")
    else:
        st.write("Dataset not loaded.")

# Header Title Banner
st.markdown("<div class='main-header'>Personal Expense & Spending Intelligence System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>End-to-End Financial Analytics, Machine Learning Forecasting & Anomaly Detection Dashboard</div>", unsafe_allow_html=True)

# Check Dataset Availability
if df is None:
    st.error("❌ Cleaned dataset (`data/cleaned_expense_data.csv`) not found! Please complete Stages 1-3 first.")
    st.stop()


# ==========================================
# SECTION 1: OVERVIEW
# ==========================================
if nav_choice == "📌 Overview":
    st.header("📌 Overview & KPI Metrics")
    st.markdown("High-level summary of financial activity across the 2024 transaction log.")

    # Calculate Core Metrics
    total_spending = df['Amount'].sum()
    total_txs = len(df)
    avg_tx = df['Amount'].mean()
    max_tx = df['Amount'].max()
    max_row = df.loc[df['Amount'].idxmax()]

    # 4 KPI Columns
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Total Outflow", value=f"₹{total_spending:,.2f}")
    with c2:
        st.metric(label="Total Transactions", value=f"{total_txs}")
    with c3:
        st.metric(label="Average Transaction", value=f"₹{avg_tx:,.2f}")
    with c4:
        st.metric(label="Highest Purchase", value=f"₹{max_tx:,.2f}")

    st.markdown("---")
    
    # Dataset Quick Summary Table
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("📄 Dataset Preview (Top 10 Records)")
        st.dataframe(df[['Date', 'Category', 'Description', 'Amount', 'Payment_Method', 'Merchant']].head(10), use_container_width=True)
    
    with col_right:
        st.subheader("📊 Key Structural Metrics")
        st.write(f"- **Total Expense Categories:** {df['Category'].nunique()}")
        st.write(f"- **Payment Modes Used:** {df['Payment_Method'].nunique()}")
        st.write(f"- **Median Transaction Size:** ₹{df['Amount'].median():,.2f}")
        st.write(f"- **Standard Deviation:** ₹{df['Amount'].std():,.2f}")
        st.write(f"- **Peak Single Item:** {max_row['Description']} (₹{max_tx:,.2f})")


# ==========================================
# SECTION 2: SPENDING ANALYTICS
# ==========================================
elif nav_choice == "📊 Spending Analytics":
    st.header("📊 Interactive Visual Spending Analytics")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🛒 Category Breakdown", 
        "📅 Monthly Trends", 
        "💳 Payment Methods", 
        "📆 Day of Week / Weekend"
    ])
    
    # TAB 1: CATEGORY BREAKDOWN
    with tab1:
        st.subheader("Category-Wise Spend Distribution")
        cat_df = df.groupby('Category')['Amount'].agg(['sum', 'count', 'mean']).reset_index().sort_values(by='sum', ascending=False)
        cat_df['Share'] = (cat_df['sum'] / df['Amount'].sum()) * 100
        
        c_left, c_right = st.columns(2)
        with c_left:
            fig_bar = px.bar(
                cat_df, 
                x='sum', 
                y='Category', 
                orientation='h',
                title="Total Expenditure by Category (₹)",
                labels={'sum': 'Total Spending (₹)', 'Category': 'Category'},
                color='sum',
                color_continuous_scale='Blues'
            )
            fig_bar.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c_right:
            fig_pie = px.pie(
                cat_df, 
                names='Category', 
                values='sum',
                title="Percentage Share of Overall Expenses",
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        st.dataframe(cat_df.rename(columns={'sum': 'Total Spend (₹)', 'count': 'Transaction Count', 'mean': 'Avg Spend (₹)', 'Share': 'Share (%)'}), use_container_width=True)

    # TAB 2: MONTHLY TRENDS
    with tab2:
        st.subheader("Monthly Outflow & Transaction Velocity")
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        m_df = df.groupby(['Month', 'Month_Name'])['Amount'].agg(['sum', 'count', 'mean']).reset_index().sort_values(by='Month')
        
        fig_line = px.line(
            m_df, 
            x='Month_Name', 
            y='sum', 
            markers=True,
            title="Monthly Total Spending Trend (2024)",
            labels={'Month_Name': 'Month', 'sum': 'Total Spending (₹)'}
        )
        fig_line.update_traces(line_color='#2563EB', line_width=3)
        st.plotly_chart(fig_line, use_container_width=True)
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            peak_m = m_df.loc[m_df['sum'].idxmax()]
            st.info(f"🏆 **Peak Spending Month:** {peak_m['Month_Name']} (₹{peak_m['sum']:,.2f})")
        with m_col2:
            low_m = m_df.loc[m_df['sum'].idxmin()]
            st.success(f"📉 **Lowest Spending Month:** {low_m['Month_Name']} (₹{low_m['sum']:,.2f})")

    # TAB 3: PAYMENT METHODS
    with tab3:
        st.subheader("Payment Channel Preference & Dollar Volume")
        pay_df = df.groupby('Payment_Method')['Amount'].agg(['sum', 'count', 'mean']).reset_index().sort_values(by='count', ascending=False)
        
        p_left, p_right = st.columns(2)
        with p_left:
            fig_pay_cnt = px.bar(
                pay_df, 
                x='Payment_Method', 
                y='count',
                title="Transaction Frequency by Payment Mode",
                labels={'count': 'Number of Transactions', 'Payment_Method': 'Payment Mode'},
                color='count',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_pay_cnt, use_container_width=True)
            
        with p_right:
            fig_pay_vol = px.pie(
                pay_df, 
                names='Payment_Method', 
                values='sum',
                title="Total Volume by Payment Mode",
                hole=0.4
            )
            st.plotly_chart(fig_pay_vol, use_container_width=True)

    # TAB 4: DAY OF WEEK & WEEKEND
    with tab4:
        st.subheader("Day-of-Week & Weekend Spending Intensity")
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=dow_order, ordered=True)
        dow_df = df.groupby('Day_of_Week', observed=False)['Amount'].agg(['sum', 'mean', 'count']).reset_index()
        
        fig_dow = px.bar(
            dow_df, 
            x='Day_of_Week', 
            y='sum',
            title="Total Expenditure by Day of the Week",
            labels={'sum': 'Total Spend (₹)', 'Day_of_Week': 'Day'},
            color='sum',
            color_continuous_scale='tealgrn'
        )
        st.plotly_chart(fig_dow, use_container_width=True)
        
        # Weekend vs Weekday summary card
        wk_df = df.groupby('Is_Weekend')['Amount'].agg(['sum', 'mean', 'count']).reset_index()
        wk_df['Period'] = wk_df['Is_Weekend'].map({0: 'Weekday', 1: 'Weekend'})
        
        w1, w2 = st.columns(2)
        with w1:
            wkday = wk_df[wk_df['Period'] == 'Weekday'].iloc[0]
            st.metric(label="Weekday Spend (Mon-Fri)", value=f"₹{wkday['sum']:,.2f}", delta=f"{wkday['count']} txs (Avg ₹{wkday['mean']:.2f})")
        with w2:
            wkend = wk_df[wk_df['Period'] == 'Weekend'].iloc[0]
            st.metric(label="Weekend Spend (Sat-Sun)", value=f"₹{wkend['sum']:,.2f}", delta=f"{wkend['count']} txs (Avg ₹{wkend['mean']:.2f})")


# ==========================================
# SECTION 3: FUTURE PREDICTION
# ==========================================
elif nav_choice == "🤖 Future Prediction":
    st.header("🤖 Machine Learning Future Expense Forecasting")
    st.markdown("Time-series regression model predicting out-of-sample total spending for the upcoming month.")

    if model is None:
        st.error("⚠️ Saved model binary (`models/spending_prediction_model.pkl`) not found! Please complete Stage 5 first.")
    else:
        # Prepare Monthly dataset for prediction display
        m_df = df.groupby(['Year', 'Month', 'Month_Name']).agg(
            Total_Spending=('Amount', 'sum'),
            Transaction_Count=('Amount', 'count'),
            Average_Transaction=('Amount', 'mean'),
            Median_Transaction=('Amount', 'median'),
            Number_of_Categories=('Category', 'nunique'),
            Number_of_Payment_Methods=('Payment_Method', 'nunique')
        ).reset_index().sort_values(by=['Year', 'Month']).reset_index(drop=True)

        weekend_spend = df[df['Is_Weekend'] == 1].groupby(['Year', 'Month'])['Amount'].sum().reset_index().rename(columns={'Amount': 'Weekend_Spending'})
        weekday_spend = df[df['Is_Weekend'] == 0].groupby(['Year', 'Month'])['Amount'].sum().reset_index().rename(columns={'Amount': 'Weekday_Spending'})
        m_df = m_df.merge(weekend_spend, on=['Year', 'Month'], how='left').merge(weekday_spend, on=['Year', 'Month'], how='left').fillna(0)
        m_df['Next_Month_Spending'] = m_df['Total_Spending'].shift(-1)
        m_df['Lag_1_Spending'] = m_df['Total_Spending'].shift(1).fillna(m_df['Total_Spending'].iloc[0])

        features = ['Total_Spending', 'Transaction_Count', 'Average_Transaction', 'Weekend_Spending', 'Weekday_Spending', 'Number_of_Categories', 'Number_of_Payment_Methods', 'Lag_1_Spending']
        
        # Latest features from December 2024
        dec_features = m_df.iloc[[-1]][features]
        jan_2025_pred = model.predict(dec_features)[0]

        col_m1, col_m2 = st.columns([2, 3])
        with col_m1:
            st.subheader("🔮 Out-of-Sample Forecast")
            st.success(f"**Predicted Next-Month Spending (Jan 2025):**\n# ₹{jan_2025_pred:,.2f}")
            st.write(f"- **Selected Model:** Random Forest Regressor")
            st.write(f"- **Test Set MAE:** ₹1,15,375.94")
            st.write(f"- **Test Set RMSE:** ₹1,64,201.63")
            st.write(f"- **Dec 2024 Actual Spend:** ₹{m_df.iloc[-1]['Total_Spending']:,.2f}")
            
        with col_m2:
            st.subheader("📊 Historical Features (Dec 2024)")
            st.write(f"- **Dec 2024 Total Spend:** ₹{dec_features['Total_Spending'].values[0]:,.2f}")
            st.write(f"- **Dec 2024 Transaction Count:** {int(dec_features['Transaction_Count'].values[0])}")
            st.write(f"- **Dec 2024 Avg Transaction:** ₹{dec_features['Average_Transaction'].values[0]:,.2f}")
            st.write(f"- **Nov 2024 Lag Spend:** ₹{dec_features['Lag_1_Spending'].values[0]:,.2f}")

        st.markdown("---")
        st.subheader("📈 Actual vs Predicted Spending (Test Set: Sep - Nov 2024)")
        
        # Test Set Actual vs Predicted Chart
        test_df = pd.DataFrame({
            'Month': ['September', 'October', 'November'],
            'Actual Spend': [469600.00, 340582.00, 476530.00],
            'Predicted Spend (RF)': [465829.00, 463150.00, 468550.00]
        })
        
        fig_ml = px.line(
            test_df, 
            x='Month', 
            y=['Actual Spend', 'Predicted Spend (RF)'], 
            markers=True,
            title="Model Evaluation Performance on Unseen Test Data",
            labels={'value': 'Spending Amount (₹)', 'Month': 'Target Month'}
        )
        st.plotly_chart(fig_ml, use_container_width=True)


# ==========================================
# SECTION 4: ANOMALY DETECTION
# ==========================================
elif nav_choice == "⚠️ Anomaly Detection":
    st.header("⚠️ Unsupervised Expense Anomaly Detection")
    st.markdown("Isolation Forest machine learning isolating unusual spending spikes and high-ticket periodic commitments.")

    if df_anom is None:
        st.error("⚠️ Anomaly dataset (`data/expense_data_with_anomalies.csv`) not found! Please complete Stage 6 first.")
    else:
        anom_df = df_anom[df_anom['Anomaly_Status'] == 'Potential Anomaly']
        anom_cnt = len(anom_df)
        pct_anom = (anom_cnt / len(df_anom)) * 100
        total_anom_spend = anom_df['Amount'].sum()

        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            st.metric(label="Flagged Anomalies", value=f"{anom_cnt}", delta=f"{pct_anom:.2f}% of dataset")
        with ac2:
            st.metric(label="Outflow in Anomalies", value=f"₹{total_anom_spend:,.2f}")
        with ac3:
            st.metric(label="Model Used", value="Isolation Forest", delta="Contamination: 3.0%")

        st.markdown("---")
        
        # Scatter Plot of Anomalies
        st.subheader("📌 Transaction Scatter Plot (Red = Potential Anomaly)")
        fig_anom_scatter = px.scatter(
            df_anom,
            x=df_anom.index,
            y='Amount',
            color='Anomaly_Status',
            color_discrete_map={'Normal': '#2563EB', 'Potential Anomaly': '#DC2626'},
            hover_data=['Date', 'Category', 'Description', 'Amount', 'Merchant'],
            title="Transaction Amounts with Potential Anomalies Highlighted"
        )
        st.plotly_chart(fig_anom_scatter, use_container_width=True)

        st.markdown("---")
        st.subheader("📋 Flagged Potential Anomalies Table (Sorted by Amount)")
        
        # Filterable anomaly table
        anom_table = anom_df.sort_values(by='Amount', ascending=False)[
            ['Date', 'Category', 'Description', 'Amount', 'Payment_Method', 'Merchant', 'Anomaly_Score', 'Anomaly_Status']
        ]
        st.dataframe(anom_table, use_container_width=True)


# ==========================================
# SECTION 5: INSIGHTS & BUDGET
# ==========================================
elif nav_choice == "💡 Insights & Budget":
    st.header("💡 Automated Spending Intelligence & Dynamic Budget Analysis")
    
    st.subheader("💰 Interactive Monthly Budget Evaluator")
    
    recent_dec_spend = df[df['Month'] == 12]['Amount'].sum()
    avg_annual_m_spend = df['Amount'].sum() / 12.0
    
    budget_input = st.number_input(
        "Enter Target Monthly Budget (₹):",
        min_value=10000.0,
        max_value=2000000.0,
        value=float(user_budget_sidebar),
        step=25000.0
    )
    
    diff = budget_input - recent_dec_spend
    pct_used = (recent_dec_spend / budget_input) * 100
    
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        st.metric(label="Target Budget", value=f"₹{budget_input:,.2f}")
    with b_col2:
        st.metric(label="Recent Spend (Dec 2024)", value=f"₹{recent_dec_spend:,.2f}")
    with b_col3:
        st.metric(label="Capacity Used", value=f"{pct_used:.1f}%")
        
    st.progress(min(int(pct_used), 100))
    
    if diff >= 0:
        st.success(f"✅ **Status Assessment:** Spending is currently **BELOW** the specified monthly budget. Surplus capacity remaining: **₹{diff:,.2f}**.")
    else:
        st.warning(f"⚠️ **Status Assessment:** Spending **HAS EXCEEDED** the specified monthly budget by **₹{abs(diff):,.2f}**.")

    st.markdown("---")
    st.subheader("💡 Automated Financial Intelligence Findings")
    
    with st.expander("📌 Spending Overview & Central Tendency"):
        st.write(f"- Total recorded expenditure is **₹{df['Amount'].sum():,.2f}** across **{len(df)}** transactions.")
        st.write(f"- Mean transaction size is **₹{df['Amount'].mean():,.2f}**, while median is **₹{df['Amount'].median():,.2f}**, reflecting a right-skewed distribution.")
        st.write(f"- Single largest expenditure: **₹{df['Amount'].max():,.2f}** (International Vacation Booking).")
        
    with st.expander("🛒 Category Drivers"):
        st.write(f"- Top Cost Driver: **Groceries (₹15,20,312.66)** representing **25.98%** of total spend.")
        st.write(f"- Groceries + Housing & Rent constitute **48.15%** of overall outflow.")
        st.write(f"- Most Frequent Category: **Food & Dining (349 transactions)**.")

    with st.expander("💳 Payment Method Preferences"):
        st.write(f"- Most Frequent Mode: **Credit Card (289 transactions, 33.57% count)**.")
        st.write(f"- Highest Volume Mode: **UPI / Net Banking (₹25,41,715.64, 43.44% volume)** due to monthly rent transfers.")

    with st.expander("🤖 ML Forecasting & Anomaly Highlights"):
        st.write(f"- Selected Prediction Model: **Random Forest Regressor** (MAE: ₹1,15,375.94).")
        st.write(f"- Predicted Next-Month Spend (Jan 2025): **₹4,78,535.56**.")
        st.write(f"- Isolation Forest flagged **26 potential anomalies (3.02% of dataset)** representing **₹19,22,074.99** in high-value rent, travel, and emergency laptop purchases.")
