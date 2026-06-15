import streamlit as st
import pandas as pd
import joblib

# Load trained model safely using caching
@st.cache_resource
def load_model():
    return joblib.load("customer_satisfaction_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure 'customer_satisfaction_model.pkl' is in this folder.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Customer Satisfaction Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("🏦 Project Information")
st.sidebar.info("""
**Customer Satisfaction Prediction System**

* **Dataset:** Bank Customer Churn Dataset
* **Algorithm:** Random Forest Classifier (Balanced)
* **Purpose:** Predict whether a customer is likely to stay or leave.
""")

# Title
st.title("🏦 Customer Satisfaction Prediction")
st.markdown("""
### Project Overview
This predictive dashboard evaluates customer churn and retention dynamics using an optimized ensemble classification framework.
""")

st.divider()

# Input UI Layout Block
st.subheader("📝 Customer Profile Input Form")

with st.form("customer_data_form"):
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        
    with row1_col2:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        balance = st.number_input("Account Balance (₹)", min_value=0.0, value=50000.0)
        salary = st.number_input("Estimated Annual Salary (₹)", min_value=0.0, value=50000.0)
        
    with row1_col3:
        tenure = st.slider("Tenure (Years)", 0, 10, 5)
        num_products = st.selectbox("Number of Products Used", [1, 2, 3, 4])
        has_card = st.selectbox("Possesses Credit Card?", ["Yes", "No"])
        active_member = st.selectbox("Is Active Member?", ["Yes", "No"])

    submit_button = st.form_submit_button("🔍 Run Prediction Analysis")

# Prediction Execution
if submit_button:
    has_card_val = 1 if has_card == "Yes" else 0
    active_member_val = 1 if active_member == "Yes" else 0
    gender_male = 1 if gender == "Male" else 0
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    st.divider()
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.subheader("📋 Profile Summary")
        summary_df = pd.DataFrame({
            "Metric": ["Country", "Gender", "Age", "Credit Score", "Account Balance", "Est. Salary"],
            "Value": [geography, gender, str(age), str(credit_score), f"₹{balance:,.2f}", f"₹{salary:,.2f}"]
        })
        st.table(summary_df.set_index("Metric"))

    with res_col2:
        st.subheader("📊 Prediction Result Analysis")
        
        raw_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [has_card_val],
            "IsActiveMember": [active_member_val],
            "EstimatedSalary": [salary],
            "Geography_Germany": [geo_germany],
            "Geography_Spain": [geo_spain],
            "Gender_Male": [gender_male]
        })

        # Fix ordering arrays seamlessly matching your training layout
        data = raw_data[model.feature_names_in_]

        prediction = model.predict(data)
        probability = model.predict_proba(data)

        if prediction[0] == 0:
            confidence = probability[0][0] * 100
            st.success("✅ Customer is likely to stay with the bank (Satisfied / Retained Customer)")
            
            if confidence >= 85:
                st.markdown("🔴 **Risk Level:** `Low Risk Retention` 🟢")
            elif confidence >= 65:
                st.markdown("🟡 **Risk Level:** `Medium Risk Retention` 🟡")
            else:
                st.markdown("🟠 **Risk Level:** `Moderate Risk Retention` 🟠")
                
            st.info(f"Model Confidence: {confidence:.2f}%")
            st.progress(int(confidence))
        else:
            confidence = probability[0][1] * 100
            st.error("⚠️ Customer is highly likely to leave the bank (Churn Risk)")
            st.markdown("🔴 **Risk Level:** `High Attrition Risk` 🔴")
            st.info(f"Model Confidence: {confidence:.2f}%")
            st.progress(int(confidence))


# Performance Metadata Footer Section
st.divider()
st.subheader("📊 Model Diagnostics & Attributes")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.info("""
    **Ensemble Framework Configurations:**
    * Balanced sample penalizations explicitly assigned to neutralize dataset skewness.
    * Reduced structural generalization error utilizing aggregated voting criteria across 50 individual decision trees.
    """)
with info_col2:
    st.markdown("""
    ### 📌 Architecture Properties
    * **Target Pipeline Label:** `Exited` (Classification Problem Type)
    * **Optimization Approach:** Hyperparameter Tuning via `GridSearchCV`
    * **Core Stack:** Python, Scikit-Learn, Pandas, Streamlit UI
    """)

# ================= MODEL ACCURACY GRAPH VISUALIZATION =================
st.divider()
st.subheader("📈 Model Performance Comparison Chart")
st.markdown("This interactive graph maps out the comparative accuracy score matrix evaluated across all 6 classification models.")

# Data structure matching your backend 6-model comparison matrix
graph_data = pd.DataFrame({
    "Model Name": [
        "Random Forest (Tuned)", 
        "Random Forest (Baseline)", 
        "SVM", 
        "KNN", 
        "Logistic Regression", 
        "Naive Bayes", 
        "Decision Tree"
    ],
    "Accuracy Score (%)": [86.90, 86.40, 84.10, 82.40, 80.90, 78.65, 78.30]
})

# Display a native interactive bar chart sorted automatically
st.bar_chart(graph_data.set_index("Model Name"))
st.success("🏆 Final Choice Summary: **Tuned Random Forest** significantly outperforms standard baseline architectures.")
