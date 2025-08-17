# -*- coding: utf-8 -*-
"""app.py"""
import json
import os
import traceback
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

st.set_page_config(page_title="SME Strategy Recommender", page_icon="📊")
st.title("📊 SME Digital Marketing Strategy Recommender")

st.markdown("""
<style>
    .tooltip {
        position: relative;
        display: inline-f;ex
        margin-left: 4px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .feedback-btn {
        transition: all 0.2s ease;
    }
    .feedback-btn:hover {
        transform: scale(1.05);
    }
    .success-msg {
        border-left: 4px solid #2ecc71;
        padding-left: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for tooltips
def tooltip(help_text):
    return f"""<span class="tooltip">
        <sup>?</sup>
        <span class="tooltiptext">{help_text}</span>
    </span>
    """

# ---------- Utilities ----------
def map_budget(v: float) -> str:
    if v < 500:
        return "<500"
    elif v <= 1000:
        return " 500- 1,000"
    elif v <= 5000:
        return "5000"
    else:
        return "None"

def map_followers(v: float) -> str:
    if v < 500:
        return "<500"
    elif v <= 1000:
        return "500 - 1000"
    elif v <= 5000:
        return "1000-5000"
    else:
        return "> 5000"

def save_to_dataset(input_data, prediction, confidence=None):
    """Save new input and prediction to CSV file with exact column matching"""
    try:
        # Create record with all original columns first
        record = {col: 0 for col in feature_cols if col not in [
            "Business Size", 
            maturity_col,
            "Primary Marketing Strategy"
        ]}
        
        # Set basic business info
        record["Business Size"] = size_map_name_to_val[size_name]
        record[maturity_col] = maturity
        record["Primary Marketing Strategy"] = prediction
        
        # Set trust questions (1-5 scale)
        trust_questions = [
            "My customers perceive our brand as honest and genuine",
            "Our marketing messages feel consistent and genuine",
            "Customers believe we care about delivering real value",
            "Customers consider our brand reliable based on our communications",
            "Our marketing has built trust in our brand",
            "Customers feel confident recommending our brand",
            "Our marketing generates steady customer responses",
            "We receive meaningful comments and clicks on our promotion",
            "Our campaigns result in repeat interactions or interest",
            "I feel the returns justify the marketing budget spent",
            "Our campaigns drive cost-effective customer attention",
            "Marketing contributes positively to sales or customer growth",
            "If data showed a better marketing approach, I would switch from my current method"
        ]
        
        for question in trust_questions:
            if question in input_data:
                record[question] = input_data[question]
        
        # Set marketing channels (0/1)
        channels = [
            "Company Website", "Facebook", "Google Ads", "Instagram", 
            "LinkedIn", "Online Store", "Pinterest", "Through Customers",
            "TikTok", "X", "Youtube", "x"
        ]
        for channel in channels:
            if channel in input_data:
                record[channel] = input_data[channel]
        
        # Set one-hot encoded features
        record[f"Industry_{industry}"] = 1
        record[f"Budget_{budget_cat}"] = 1
        record[f"Followers_{followers_cat}"] = 1
        
        # Add metadata columns
        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record["Digital Marketing Maturity"] = maturity  # Duplicate for readability
        record["Industry"] = industry
        record["Budget"] = budget_num
        record["Followers"] = followers_num
        record["Predicted Strategy"] = prediction
        record["Confidence"] = confidence if confidence else None
        record["Budget Category"] = budget_cat
        record["Followers Category"] = followers_cat

        # Convert to DataFrame ensuring column order matches original
        new_record = pd.DataFrame([record])
        
        # Ensure all original columns are present (fill missing with 0)
        for col in feature_cols:
            if col not in new_record.columns:
                new_record[col] = 0
        
        # Reorder columns to match original dataset
        new_record = new_record[feature_cols + [
            "Digital Marketing Maturity", "Industry", 
            "Budget", "Followers", "Predicted Strategy", "Confidence",
            "Budget Category", "Followers Category"
        ]]
        
        # Save to CSV
        file_path = "processed_dataset.csv"
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_record], ignore_index=True)
        else:
            updated_data = new_record
        
        updated_data.to_csv(file_path, index=False)
        # st.success("✅ Data successfully saved with all original columns")
        
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")
        st.error(traceback.format_exc())

def reset_form():
    """Reset all form inputs to default values"""
    st.session_state.size_name = "Small"
    st.session_state.maturity = 3
    st.session_state.industry = industry_opts[0] if industry_opts else "Other"
    st.session_state.budget_num = 1500
    st.session_state.followers_num = 1200
    for question in trust_questions:
        if question in feature_cols:
            st.session_state[question] = 3  # Fix: use question as key, not f"trust_{question}"
    st.session_state.feedback_submitted = False
    st.session_state.feedback = None

# Add this new function to save feedback
def save_feedback_to_dataset(strategy, feedback):
    feedback_file = "feedback_data.csv"
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "feedback": feedback,
        "business_size": st.session_state.get("size_name", ""),
        "industry": st.session_state.get("industry", "")
    }
    try:
        df = pd.DataFrame([row])
        if not os.path.exists(feedback_file):
            df.to_csv(feedback_file, index=False)
        else:
            df.to_csv(feedback_file, mode="a", header=False, index=False)
    except Exception as e:
        st.error(f"Could not save feedback: {e}")

@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load model, label encoder, and feature list with nice error messages."""
    try:
        def pick(p):
            for c in [p, os.path.join(os.getcwd(), p)]:
                if os.path.exists(c):
                    return c
            return p  # fallback (will raise)

        model_path = pick("sme_strategy_model.pkl")
        enc_path   = pick("label_encoder.pkl")
        cols_path  = pick("feature_columns.json")

        model = joblib.load(model_path)
        label_enc = joblib.load(enc_path)

        with open(cols_path, "r") as f:
            raw_cols = json.load(f)
            feature_cols = [c.strip() for c in raw_cols]

        return model, label_enc, feature_cols, None
    except Exception as e:
        return None, None, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

model, label_enc, feature_cols, load_err = load_artifacts()
if model is not None and hasattr(model, "feature_names_in_"):
    feature_cols = model.feature_names_in_.tolist()

if load_err or not feature_cols:
    st.error("Failed to load required artifacts. Please check the error below:")
    st.error(load_err)
    st.stop()

def options_from_prefix(prefix: str):
    """Extract unique options from one-hot encoded columns"""
    options = set()
    for col in feature_cols:
        if col.startswith(prefix):
            option = col[len(prefix):].strip()
            options.add(option)
    return sorted(options)

industry_opts  = options_from_prefix("Industry_")
budget_opts    = options_from_prefix("Budget_")
follower_opts  = options_from_prefix("Followers_")

size_map_name_to_val = {"Micro": 1, "Small": 2, "Medium": 3, "Large": 4}
size_names = list(size_map_name_to_val.keys())

maturity_col = next((c for c in feature_cols if "Digital Marketing Maturity" in c), None)
if not maturity_col:
    st.error("❌ Cannot find the Digital Maturity column in feature columns")
    st.stop()

st.subheader("Enter your business details")

col1, col2 = st.columns(2)
with col1:
    size_name = st.selectbox("Business Size", size_names, index=1, key="size_name")
    st.markdown(f"""Digital Marketing Maturity {tooltip('How effectively your business uses digital tools (1=basic, 5=advanced)')}""", unsafe_allow_html=True)
    maturity  = st.slider("Rate from 1 to 5", 1, 5, 3, key="maturity")
    industry  = st.selectbox("Industry", industry_opts or ["Other"], key="industry")
with col2:
    budget_num    = st.number_input("Monthly Marketing Budget (€)", min_value=0, step=100, value=1500, key="budget_num")
    followers_num = st.number_input("Total Social Followers", min_value=0, step=100, value=1200, key="followers_num")

# Add trust-related questions if they exist in feature_cols
trust_questions = [
    "My customers perceive our brand as honest and genuine",
    "Our marketing messages feel consistent and genuine",
    "Customers believe we care about delivering real value",
    "Customers consider our brand reliable based on our communications",
    "Our marketing has built trust in our brand",
    "Customers feel confident recommending our brand",
    "Our marketing generates steady customer responses",
    "We receive meaningful comments and clicks on our promotion",
    "Our campaigns result in repeat interactions or interest",
    "I feel the returns justify the marketing budget spent",
    "Our campaigns drive cost-effective customer attention",
    "Marketing contributes positively to sales or customer growth",
    "If data showed a better marketing approach, I would switch from my current method"
]

trust_responses = {}
with st.expander("🔍 Additional Trust Questions (Optional)", expanded=False):
    for question in trust_questions:
        if question in feature_cols:
            trust_responses[question] = st.slider(question, 1, 5, 3, key=question)

# Add reset button
st.button("🔄 Reset Form", on_click=reset_form)

budget_cat    = map_budget(float(budget_num))
followers_cat = map_followers(float(followers_num))

# Build input row aligned with training features
row = {c: 0 for c in feature_cols}

# Set business size (numeric)
if "Business Size" in row:
    row["Business Size"] = size_map_name_to_val[size_name]

# Set maturity (numeric)
row[maturity_col] = int(maturity)

# Set trust responses if provided
for question, value in trust_responses.items():
    row[question] = value

# Set one-hot encoded features
def set_one_hot(prefix, selected_value, all_options):
    """Set the correct one-hot encoded column based on user selection"""
    # Find the best matching column
    selected_clean = selected_value.lower().strip().replace(" ", "").replace("-", "")
    
    for option in all_options:
        option_clean = option.lower().strip().replace(" ", "").replace("-", "")
        if option_clean == selected_clean:
            col_name = f"{prefix}{option}"
            if col_name in row:
                row[col_name] = 1
                return
    
    # If exact match not found, try partial match
    for option in all_options:
        option_clean = option.lower().strip().replace(" ", "").replace("-", "")
        if option_clean in selected_clean or selected_clean in option_clean:
            col_name = f"{prefix}{option}"
            if col_name in row:
                row[col_name] = 1
                return
    
    # If still not found, default to first option
    if all_options:
        col_name = f"{prefix}{all_options[0]}"
        if col_name in row:
            row[col_name] = 1
            # st.warning(f"Could not find exact match for '{selected_value}'. Using '{all_options[0]}' instead.")

# Set one-hot encoded features
set_one_hot("Industry_", industry, industry_opts)
set_one_hot("Budget_", budget_cat, budget_opts)
set_one_hot("Followers_", followers_cat, follower_opts)

# Create DataFrame for prediction
X_new = pd.DataFrame([row], columns=feature_cols)

# Display the input data for confirmation
st.subheader("Input Data for Prediction")
st.write("This is the data that will be used for prediction:")
st.dataframe(X_new, use_container_width=True)

if st.button("🔮 Get Recommendation"):
    try:
        pred = model.predict(X_new)[0]
        strategy = label_enc.inverse_transform([pred])[0]
        st.session_state.strategy = strategy  # <-- Store in session_state

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0][pred]
            st.success(f"✅ Recommended Strategy: **{strategy}**")
            st.write(f"Confidence: **{proba*100:.1f}%**")
            save_to_dataset(row, strategy, proba*100)
        else:
            st.success(f"✅ Recommended Strategy: **{strategy}**")
            save_to_dataset(row, strategy)

        # Initialize feedback state if missing
        if "feedback_submitted" not in st.session_state:
            st.session_state.feedback_submitted = False
        if "feedback" not in st.session_state:
            st.session_state.feedback = None

    except Exception as e:
        st.error(f"Prediction failed: {e}\n{traceback.format_exc()}")

# --- Feedback UI (always visible if strategy exists) ---
if "strategy" in st.session_state and st.session_state.strategy:
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
    if "feedback" not in st.session_state:
        st.session_state.feedback = None

    if not st.session_state.feedback_submitted:
        st.write("Was this recommendation helpful?")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("👍 Helpful", 
                   key="like_btn",
                   help="This recommendation was useful",
                   use_container_width=True,
                   type="primary"):
                st.session_state.feedback = "Like"
                st.session_state.feedback_submitted = True
                save_feedback_to_dataset(st.session_state.strategy, "Like")
                st.rerun()
        with col2:
            if st.button("👎 Ok-",
                   key="dislike_btn",
                   help="This needs work",
                   use_container_width=True,
                   type="secondary"):
                st.session_state.feedback = "Dislike"
                st.session_state.feedback_submitted = True
                save_feedback_to_dataset(st.session_state.strategy, "Dislike")
                st.rerun()

    if st.session_state.feedback_submitted:
        if st.session_state.feedback == "Like":
            st.success('Thank you for your feedback! ❤️')
        elif st.session_state.feedback == "Dislike":
            st.info('Aw! We will work on improving. Thanks for your feedback!')

