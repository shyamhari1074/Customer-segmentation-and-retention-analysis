# Customer Segmentation and Retention Analysis

A machine learning-powered Streamlit application for analyzing and predicting customer segments and retention patterns.

## 📊 Features

### Customer Segmentation
- **K-Means Clustering**: Identifies distinct customer segments based on behavioral and financial metrics
- **Interactive Dashboard**: User-friendly interface for inputting customer information
- **Segment Classification**: Automatically categorizes customers into meaningful segments:
  - 💰 High-Value Customers
  - 🛍️ Regular Shoppers
  - 🌐 Digital-Focused Customers
  - 💤 At-Risk Customers
  - 🆕 New Customers
- **Visual Analytics**: Charts and metrics showing purchase patterns and engagement levels

### Customer Retention Analysis ⭐ (NEW)
- **Churn Prediction**: LogisticRegression model predicts customer churn probability
- **Risk Assessment**: Classifies customers into High, Medium, or Low churn risk
- **Personalized Recommendations**: Provides targeted retention strategies based on individual customer behavior
- **Detailed Metrics**: Analyzes engagement score, spending patterns, recency, campaign acceptance, and more
- **Risk Factor Summary**: Visualizes which factors contribute most to churn risk

## 🛠️ Tech Stack

- **Python 3.x**
- **Streamlit**: Interactive web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning and preprocessing
- **Joblib**: Model persistence
- **NumPy**: Numerical computations

## 📁 Project Structure

```
.
├── segmentation.py              # Main Streamlit application
├── Analysis_model.ipynb         # Jupyter notebook with analysis
├── customer_segmentation.csv    # Customer dataset
├── kmeans_model.pkl             # Trained K-Means model
├── scaler.pkl                   # Feature scaler (StandardScaler)
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip or conda package manager

### Installation

1. Clone the repository
```bash
git clone https://github.com/shyamhari1074/Customer-segmentation-and-retention-analysis.git
cd "customer Segmentation and retent eion"
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- streamlit
- pandas
- scikit-learn
- numpy
- joblib

Install all at once:
```bash
pip install streamlit pandas scikit-learn numpy joblib
```

## 💻 Running the Application

```bash
streamlit run segmentation.py
```

The application will open in your default browser at `http://localhost:8501`

## 📝 How to Use

### Customer Segmentation Analysis

1. Open the application
2. Enter customer information in the sidebar:
   - **Demographics**: Age
   - **Financial**: Annual Income, Total Spending
   - **Purchase Behavior**: Web and Store purchases
   - **Engagement**: Web visits per month, Recency
3. Click "Predict Segment"
4. View the customer segment and detailed analysis charts

### Input Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| Age | 18-100 | Customer's age in years |
| Annual Income | $0-$200,000 | Customer's annual income |
| Total Spending | $0-$5,000 | Total spending across channels |
| Web Purchases | 0-100 | Number of online purchases |
| Store Purchases | 0-100 | Number of in-store purchases |
| Web Visits/Month | 0-100 | Average monthly website visits |
| Recency (Days) | 0-365 | Days since last purchase |

## 📊 Model Details

### K-Means Clustering
- **Algorithm**: K-Means with k=5 (5 customer segments)
- **Features**: 7 customer attributes (Age, Income, Spending, Purchases, Visits, Recency)
- **Preprocessing**: StandardScaler normalization

### Feature Scaling
- Fitted StandardScaler transforms all input features
- Ensures features are on comparable scales for accurate clustering

## 🎯 Segment Descriptions

| Segment | Name | Characteristics |
|---------|------|-----------------|
| 0 | High-Value | High income & spending, premium customers |
| 1 | Regular Shoppers | Balanced purchase patterns, consistent engagement |
| 2 | Digital-Focused | High web activity, online-oriented |
| 3 | At-Risk | Low engagement, inactive customers |
| 4 | New Customers | Recently acquired, developing patterns |

## 🎨 UI/UX Features

- **Responsive Design**: Wide layout optimized for different screen sizes
- **Color-Coded Visualization**: Gradient backgrounds for easy identification
- **Interactive Inputs**: Sliders and number inputs with helpful tooltips
- **Real-time Metrics**: Display customer overview instantly
- **Visual Charts**: Bar and line charts showing purchase and engagement patterns

## 📈 Future Enhancements

- Advanced churn prediction models (Random Forest, XGBoost)
- Batch customer analysis and bulk predictions
- Export predictions to CSV/Excel
- Time-series customer behavior analysis
- Recommendation engine for retention strategies
- Customer lifetime value (CLV) calculation
- A/B testing framework for retention strategies

## 🔧 Troubleshooting

### Feature Name Mismatch Error
If you see "Feature names should match", ensure the input DataFrame column names match the trained model:

**Segmentation Features**: `Age`, `Income`, `Total_Spending`, `NumWebPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`, `Recency`

**Retention Features**: `Income`, `Recency`, `Total_Purchases`, `Campaign_Accepted`, `Recency_Ratio`, `Engagement_Score`, `NumWebVisitsMonth`, `Kidhome`, `Teenhome`, `Customer_Days`

### Model Not Found
Ensure these files are in the same directory as `segmentation.py`:
- `kmeans_model.pkl` and `scaler.pkl` (Segmentation)
- `retention_model.pkl`, `retention_scaler.pkl`, `retention_features.pkl` (Retention)

### Streamlit Port Already in Use
```bash
streamlit run segmentation.py --server.port 8502
```

## 📚 Model Details

### Segmentation Model
- **Algorithm**: K-Means Clustering (k=5)
- **Features**: 7 customer attributes
- **Preprocessing**: StandardScaler normalization

### Retention Model
- **Algorithm**: Logistic Regression
- **Features**: 10 behavioral and financial metrics
- **Target**: Binary churn classification (0 = Retained, 1 = Churned)
- **Churn Definition**: Customer is marked as churned if:
  - Recency > 80 days AND Total Spending < $500, OR
  - Has filed complaints (Complain > 0), OR
  - Recency > 90 days



## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

[Shyam Hari](https://github.com/shyamhari1074)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Last Updated**: May 28, 2026
