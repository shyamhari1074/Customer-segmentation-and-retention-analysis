import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation & Retention",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load retention models
retention_model = joblib.load("retention_model.pkl")
retention_scaler = joblib.load("retention_scaler.pkl")
retention_features = joblib.load("retention_features.pkl")

# Custom CSS styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .segment-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    .churn-high {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f5576c;
    }
    .churn-medium {
        background: linear-gradient(135deg, #ffa500 0%, #ffb84d 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffa500;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Segmentation'

# Header and Navigation
st.title("📊 Customer Segmentation & Retention Analysis")
st.markdown("---")

# Navigation buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📊 Segmentation Analysis", use_container_width=True):
        st.session_state.page = 'Segmentation'
with col2:
    if st.button("📈 Retention Analysis", use_container_width=True):
        st.session_state.page = 'Retention'

st.markdown("---")

# Segment descriptions
segment_descriptions = {
    0: "💰 **High-Value Customers** - Premium customers with high income and spending",
    1: "🛍️ **Regular Shoppers** - Consistent customers with balanced purchase patterns",
    2: "🌐 **Digital-Focused** - Online-oriented customers with high web activity",
    3: "💤 **At-Risk Customers** - Inactive customers with low recent engagement",
    4: "🆕 **New Customers** - Recently acquired customers with developing patterns"
}

# ============================================================================
# SEGMENTATION ANALYSIS PAGE
# ============================================================================
if st.session_state.page == 'Segmentation':
    st.subheader("📊 Segment Customers Using K-Means Clustering")
    
    # Sidebar for input
    with st.sidebar:
        st.header("👤 Customer Profile")
        st.markdown("Enter customer details to predict their segment")
        st.markdown("---")
        
        # Demographics Section
        st.subheader("📋 Demographics")
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            help="Customer's age in years"
        )
        
        st.markdown("---")
        st.subheader("💵 Financial Information")
        income = st.number_input(
            "Annual Income ($)",
            min_value=0,
            max_value=200000,
            value=75000,
            step=5000,
            help="Customer's annual income in dollars"
        )
        
        total_spending = st.number_input(
            'Total Spending ($)',
            min_value=0,
            max_value=5000,
            value=1500,
            step=100,
            help="Total spending across all channels"
        )
        
        st.markdown("---")
        st.subheader("🛒 Purchase Behavior")
        
        col1, col2 = st.columns(2)
        with col1:
            num_web_purchases = st.number_input(
                'Web Purchases',
                min_value=0,
                max_value=100,
                value=15,
                help="Number of purchases made online"
            )
        
        with col2:
            num_store_purchases = st.number_input(
                'Store Purchases',
                min_value=0,
                max_value=100,
                value=12,
                help="Number of in-store purchases"
            )
        
        st.markdown("---")
        st.subheader("📱 Engagement Metrics")
        
        num_web_visits_month = st.slider(
            'Web Visits/Month',
            min_value=0,
            max_value=100,
            value=8,
            help="Average website visits per month"
        )
        
        recency = st.slider(
            'Recency (Days)',
            min_value=0,
            max_value=365,
            value=45,
            help="Days since last purchase"
        )
        
        st.markdown("---")
        predict_button = st.button("🔍 Predict Segment", use_container_width=True)

    # Main content area
    # Display input summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Age", f"{age} years")
    with col2:
        st.metric("Annual Income", f"${income:,.0f}")
    with col3:
        st.metric("Total Spending", f"${total_spending:,.0f}")
    with col4:
        st.metric("Last Purchase", f"{recency} days ago")

    st.markdown("---")

    # Prepare data for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Income': [income],
        'Total_Spending': [total_spending],
        'NumWebPurchases': [num_web_purchases],
        'NumStorePurchases': [num_store_purchases],
        'NumWebVisitsMonth': [num_web_visits_month],
        'Recency': [recency]
    })

    input_scaled = scaler.transform(input_data)

    # Display prediction result
    if predict_button:
        cluster = int(kmeans.predict(input_scaled)[0])
        
        st.markdown("### 🎯 Prediction Result")
        st.markdown(f"""
        <div class="segment-badge">
        Customer Segment: {cluster}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(segment_descriptions[cluster])
        
        # Display detailed metrics
        st.markdown("---")
        st.markdown("### 📊 Customer Profile Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Purchase Activity**")
            purchase_data = pd.DataFrame({
                'Channel': ['Web', 'Store'],
                'Purchases': [num_web_purchases, num_store_purchases]
            })
            st.bar_chart(purchase_data.set_index('Channel'))
        
        with col2:
            st.markdown("**Engagement Metrics**")
            engagement_data = pd.DataFrame({
                'Metric': ['Web Visits/Month', 'Days Since Purchase'],
                'Value': [num_web_visits_month, recency]
            })
            st.bar_chart(engagement_data.set_index('Metric'))

# ============================================================================
# RETENTION ANALYSIS PAGE
# ============================================================================
elif st.session_state.page == 'Retention':
    st.subheader("📈 Predict Customer Churn Risk")
    
    # Sidebar for retention input
    with st.sidebar:
        st.header("👤 Customer Profile")
        st.markdown("Enter details for churn risk prediction")
        st.markdown("---")
        
        st.subheader("💰 Financial Metrics")
        income_ret = st.number_input(
            "Annual Income ($)",
            min_value=0,
            max_value=200000,
            value=50000,
            step=5000,
            help="Customer's annual income"
        )
        
        total_spending_ret = st.number_input(
            "Total Spending ($)",
            min_value=0,
            max_value=10000,
            value=1000,
            step=100,
            help="Total lifetime spending"
        )
        
        st.markdown("---")
        st.subheader("📊 Purchase & Engagement")
        
        recency_ret = st.slider(
            "Recency (Days since last purchase)",
            min_value=0,
            max_value=365,
            value=30,
            help="Days since their last purchase"
        )
        
        total_purchases = st.number_input(
            "Total Purchases",
            min_value=0,
            max_value=200,
            value=25,
            help="Lifetime purchase count"
        )
        
        campaign_accepted = st.number_input(
            "Campaigns Accepted",
            min_value=0,
            max_value=10,
            value=2,
            help="Number of marketing campaigns accepted"
        )
        
        web_visits = st.slider(
            "Web Visits per Month",
            min_value=0,
            max_value=100,
            value=5,
            help="Average monthly website visits"
        )
        
        st.markdown("---")
        st.subheader("👨‍👩‍👧‍👦 Household Info")
        
        col1, col2 = st.columns(2)
        with col1:
            kidhome = st.number_input(
                "Kids at Home",
                min_value=0,
                max_value=5,
                value=0
            )
        with col2:
            teenhome = st.number_input(
                "Teens at Home",
                min_value=0,
                max_value=5,
                value=0
            )
        
        st.markdown("---")
        st.subheader("⏱️ Customer History")
        
        customer_days = st.slider(
            "Customer for (Days)",
            min_value=1,
            max_value=3000,
            value=500,
            help="How long they've been a customer"
        )
        
        st.markdown("---")
        retention_predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

    # Calculate derived metrics
    recency_ratio = recency_ret / (customer_days + 1)
    engagement_score = (total_purchases / (customer_days + 1)) * 100
    spending_per_day = total_spending_ret / (customer_days + 1)

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Income", f"${income_ret:,.0f}")
    with col2:
        st.metric("Total Spending", f"${total_spending_ret:,.0f}")
    with col3:
        st.metric("Customer Days", f"{customer_days}")
    with col4:
        st.metric("Days Inactive", f"{recency_ret}")

    st.markdown("---")

    if retention_predict_btn:
        # Prepare retention data in the exact order the model was trained
        retention_input = pd.DataFrame({
            'Income': [income_ret],
            'Recency': [recency_ret],
            'Total_Purchases': [total_purchases],
            'Campaign_Accepted': [campaign_accepted],
            'Recency_Ratio': [recency_ratio],
            'Engagement_Score': [engagement_score],
            'NumWebVisitsMonth': [web_visits],
            'Kidhome': [kidhome],
            'Teenhome': [teenhome],
            'Customer_Days': [customer_days]
        })

        # Reorder columns to match the retention_features order
        retention_input = retention_input[retention_features]

        # Scale the input
        retention_input_scaled = retention_scaler.transform(retention_input)

        # Make prediction
        churn_probs = retention_model.predict_proba(retention_input_scaled)[0]
        churn_prob = churn_probs[1]  # Probability of churn (class 1)
        churn_prediction = retention_model.predict(retention_input_scaled)[0]
        
        # Debug information
        with st.expander("🔧 Debug Info"):
            st.write("**Scaled Features:**")
            debug_df = pd.DataFrame(retention_input_scaled, columns=retention_features).T
            st.dataframe(debug_df)
            st.write(f"**Raw Probabilities:** No Churn: {churn_probs[0]:.4f}, Churn: {churn_probs[1]:.4f}")

        # Determine risk level
        if churn_prob >= 0.7:
            risk_level = "🔴 HIGH RISK"
            risk_class = "churn-high"
            risk_color = "#f5576c"
        elif churn_prob >= 0.4:
            risk_level = "🟡 MEDIUM RISK"
            risk_class = "churn-medium"
            risk_color = "#ffa500"
        else:
            risk_level = "🟢 LOW RISK"
            risk_class = "churn-low"
            risk_color = "#4facfe"

        # Display results
        st.markdown("### 🎯 Churn Prediction Result")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class="{risk_class}">
            <h3>Churn Probability: {churn_prob*100:.1f}%</h3>
            <p>Risk Level: {risk_level}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Prediction", "Churn" if churn_prediction == 1 else "Retain")

        st.markdown("---")

        # Recommendations
        st.markdown("### 💡 Retention Recommendations")

        recommendations = []

        if churn_prob >= 0.7:
            recommendations.append("🚨 **URGENT ACTION REQUIRED** - High churn risk detected!")
            
        if recency_ret > 90:
            recommendations.append("📧 **Immediate Outreach:** Customer is inactive. Send personalized re-engagement email.")

        if recency_ret > 60:
            recommendations.append("📞 **Follow-up:** Reach out to understand their concerns or needs.")

        if total_spending_ret < 500 and customer_days > 100:
            recommendations.append("💳 **Low Spender:** Offer special promotions or loyalty rewards to boost engagement.")

        if engagement_score < 0.5:
            recommendations.append("📱 **Increase Engagement:** Send more frequent, personalized communications.")

        if campaign_accepted == 0:
            recommendations.append("🎁 **Personalization:** Create tailored offers based on their purchase history.")

        if web_visits < 2:
            recommendations.append("🌐 **Digital Engagement:** Encourage online browsing with exclusive web-only deals.")

        if kidhome + teenhome > 0:
            recommendations.append("👨‍👩‍👧 **Family Focus:** Offer family-friendly products or bundles.")

        if churn_prob < 0.4:
            recommendations.append("✅ **Maintain Strategy:** Continue current engagement approach - customer is satisfied!")

        if not recommendations:
            recommendations.append("📊 Monitor this customer regularly for any changes in behavior.")

        for rec in recommendations:
            st.info(rec)

        # Detailed metrics breakdown
        st.markdown("---")
        st.markdown("### 📊 Detailed Metrics Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Engagement Metrics**")
            engagement_metrics = pd.DataFrame({
                'Metric': ['Engagement\nScore', 'Web\nVisits', 'Campaign\nAccepted', 'Recency\nRatio'],
                'Value': [
                    min(100, engagement_score),
                    min(100, web_visits * 10),
                    campaign_accepted * 10,
                    min(100, recency_ratio * 100)
                ]
            })
            st.bar_chart(engagement_metrics.set_index('Metric'))

        with col2:
            st.markdown("**Financial Metrics**")
            financial_metrics = pd.DataFrame({
                'Metric': ['Income\n(x100)', 'Spending\nper Day\n(x100)', 'Total\nSpending\n(x10)'],
                'Value': [
                    min(100, income_ret / 2000),
                    min(100, spending_per_day * 10),
                    min(100, total_spending_ret / 100)
                ]
            })
            st.bar_chart(financial_metrics.set_index('Metric'))

        # Risk factors summary
        st.markdown("---")
        st.markdown("### 🎯 Risk Factor Summary")

        risk_factors = {
            'Recency (Days Inactive)': {'value': recency_ret, 'threshold': 90, 'high_bad': True},
            'Engagement Score': {'value': engagement_score, 'threshold': 0.5, 'high_bad': False},
            'Spending per Day': {'value': spending_per_day, 'threshold': 5, 'high_bad': False},
            'Campaign Acceptance': {'value': campaign_accepted, 'threshold': 1, 'high_bad': False},
        }

        risk_summary = []
        for factor, details in risk_factors.items():
            if details['high_bad']:
                status = "⚠️" if details['value'] > details['threshold'] else "✅"
            else:
                status = "⚠️" if details['value'] < details['threshold'] else "✅"
            risk_summary.append({
                'Risk Factor': factor,
                'Value': f"{details['value']:.2f}",
                'Status': status
            })

        risk_df = pd.DataFrame(risk_summary)
        st.dataframe(risk_df, use_container_width=True)

