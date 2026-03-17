import streamlit as st
import joblib
import numpy as np

# Load model and preprocessors
model = joblib.load("xgboost_cuisine_model.pkl")
label_y = joblib.load("label_encoder.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---- Page Config ----
st.set_page_config(page_title="Cuisine Classifier 🍽️", page_icon="🍲", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #141e30, #243b55);
        color: #f8f9fa;
    }
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #1f2c3a !important;
        color: white !important;
    }
    .main-title {
        font-size: 3rem;
        color: #ffcc70;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #cfd8dc;
        margin-bottom: 30px;
    }
    .prediction-box {
        background-color: #1f2c3a;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        color: #f8f9fa;
        font-size: 1.5rem;
        box-shadow: 0 0 15px rgba(255,255,255,0.1);
    }
    footer {
        text-align: center;
        color: #aaa;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- App Title ----
st.markdown("<div class='main-title'>🍽️ Cuisine Classification App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Predict the cuisine type based on restaurant details</div>", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ App Info")
    st.markdown("""
        **Model:** XGBoost Classifier  
        **Accuracy:** ~97.7%  
        **Vectorizer:** TF-IDF  
        **Developer:** You 💻  
    """)
    st.markdown("---")
    st.markdown("🔗 *Built using Streamlit + XGBoost*")

# ---- Input Section ----
col1, col2 = st.columns(2)

with col1:
    cuisines = st.text_input("Enter cuisines (e.g., Indian, Chinese, Italian)")
    dishes = st.text_input("Enter dishes liked (e.g., Biryani, Noodles, Pizza)")

with col2:
    rating = st.number_input("Restaurant Rating (e.g., 4.2)", min_value=0.0, max_value=5.0, step=0.1)
    cost = st.number_input("Approx cost for two people", min_value=0)

# ---- Prediction ----
if st.button("🔍 Predict Cuisine"):
    if cuisines.strip() == "" and dishes.strip() == "":
        st.warning("⚠️ Please enter at least cuisines or dishes liked.")
    else:
        # Prepare input
        text_input = cuisines + " " + dishes
        X_text = tfidf.transform([text_input])
        X_other = np.array([[rating, cost]])
        X_combined = np.hstack((X_text.toarray(), X_other))
        
        # Predict
        pred = model.predict(X_combined)
        result = label_y.inverse_transform(pred)[0]

        st.success("🎉 Prediction Complete!")
        st.markdown(f"<div class='prediction-box'>Predicted Cuisine: <b>{result}</b></div>", unsafe_allow_html=True)
        st.balloons()

# ---- Footer ----
st.markdown("<footer>✨ Built with ❤️ using Streamlit and XGBoost ✨</footer>", unsafe_allow_html=True)
