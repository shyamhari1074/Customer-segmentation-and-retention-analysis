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
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("📊 Customer Segmentation & Retention Analysis")
st.markdown("---")

# Segment descriptions
segment_descriptions = {
    0: "💰 **High-Value Customers** - Premium customers with high income and spending",
    1: "🛍️ **Regular Shoppers** - Consistent customers with balanced purchase patterns",
    2: "🌐 **Digital-Focused** - Online-oriented customers with high web activity",
    3: "💤 **At-Risk Customers** - Inactive customers with low recent engagement"
}

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
    'Total Spending': [total_spending],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_store_purchases],
    'NumWebVisitsMonth': [num_web_visits_month],
    'Recency': [recency]
})

input_scaled = scaler.transform(input_data)

# Display prediction result
if predict_button:
    cluster = kmeans.predict(input_scaled)[0]
    
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
 
