# -*- coding: utf-8 -*-
"""app.py"""
import json
import os
import traceback
import openai
import pandas as pd
import streamlit as st
import numpy as np

from utils.config import size_map_name_to_val, size_names, trust_questions
from utils.data_utils import map_budget, map_followers, save_to_dataset, reset_form
from utils.model_utils import load_artifacts, options_from_prefix, set_one_hot
from utils.feedback_utils import save_feedback_to_dataset
from utils.ui_utils import tooltip, inject_css

openai.api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
inject_css(st)

st.set_page_config(page_title="SME Strategy Recommender", page_icon="📊")
st.title("📊 SME Digital Marketing Strategy Recommender")

model, label_enc, feature_cols = load_artifacts()
if model is not None and hasattr(model, "feature_names_in_"):
    feature_cols = model.feature_names_in_.tolist()

maturity_col = next((c for c in feature_cols if "Digital Marketing Maturity" in c), None)
if not maturity_col:
    st.error("❌ Cannot find the Digital Maturity column in feature_columns")
    st.stop()

industry_opts  = options_from_prefix("Industry_", feature_cols)
budget_opts    = options_from_prefix("Budget_", feature_cols)
follower_opts  = options_from_prefix("Followers_", feature_cols)

st.subheader("Enter your business details")
col1, col2 = st.columns(2)
with col1:
    size_name = st.selectbox("Business Size", size_names, index=0, key="size_name")
    st.markdown(f"""Digital Marketing Maturity {tooltip('How effectively your business uses digital tools (1=basic, 5=advanced)')}""", unsafe_allow_html=True)
    maturity  = st.slider("Rate from 1 to 5", 1, 5, 1, key="maturity")
    industry  = st.selectbox("Industry", industry_opts or ["Other"], key="industry")
with col2:
    budget_num    = st.number_input("Monthly Marketing Budget (€)", min_value=0, step=100, value=0, key="budget_num")
    followers_num = st.number_input("Total Social Followers", min_value=0, step=100, value=0, key="followers_num")

trust_responses = {}
with st.expander("🔍 Additional Trust Questions (Optional)", expanded=False):
    for ui_text, backend_name in trust_questions.items():
        if backend_name in feature_cols:
            value = st.slider(ui_text, 1, 5, 1, key=ui_text)
            trust_responses[backend_name] = value

st.button("🔄 Reset Form", on_click=lambda: reset_form(st, trust_questions, feature_cols, industry_opts))

budget_cat    = map_budget(float(budget_num))
followers_cat = map_followers(float(followers_num))

row = {c: 0 for c in feature_cols}
if "Business Size" in row:
    row["Business Size"] = size_map_name_to_val[size_name]
row[maturity_col] = int(maturity)
for question, value in trust_responses.items():
    row[question] = value

set_one_hot(row, "Industry_", industry, industry_opts)
set_one_hot(row, "Budget_", budget_cat, budget_opts)
set_one_hot(row, "Followers_", followers_cat, follower_opts)

X_new = pd.DataFrame([row], columns=feature_cols)

def explain_with_llm():
    """
    Sends the actual LIME explanation + prediction to GPT
    and asks for a business-friendly overview
    """

    profile = {
        "predicted_strategy": strategy,
        "class_probabilities": proba if hasattr(model, "predict_proba") else None,
        "industry": industry,
        "business_size": size_name,
        "digital_maturity": int(maturity),
        "budget_eur": float(budget_num),
        "followers": int(followers_num),
        "readiness_avg": float(np.mean(list(trust_responses.values()))) if trust_responses else None,
    }

    sys_msg = (
        "You are a practical marketing analyst. Explain concisely why the predicted strategy fits now, "
        "then provide 3 specific next steps for a 4-6 week pilot. Do not change the predicted label."
    )
    user_msg = "Business profile JSON:\n" + json.dumps(profile, indent=2)

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user",   "content": user_msg}],
            temperature=0.3,
            max_tokens=400,
        )
        text = resp.choices[0].message["content"].strip()
        with st.expander("🧠 Suggestion & Why (AI-assisted)", expanded=True):
            st.markdown(text)
    except Exception as e:
        st.info(f"(AI suggestion failed: {e})")


if st.button("🔮 Get Recommendation"):
    try:
        pred = model.predict(X_new)[0]
        strategy = label_enc.inverse_transform([pred])[0]
        st.session_state.strategy = strategy  

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_new)[0][pred]
            st.success(f"✅ Recommended Strategy: **{strategy}**")
            st.write(f"Confidence: **{proba*100:.1f}%**")
            save_to_dataset(row, strategy, proba*100, feature_cols, size_name, size_map_name_to_val, maturity_col, maturity, industry, budget_cat, followers_cat, budget_num, followers_num)
        else:
            st.success(f"✅ Recommended Strategy: **{strategy}**")
            save_to_dataset(row, strategy, None, feature_cols, size_name, size_map_name_to_val, maturity_col, maturity, industry, budget_cat, followers_cat, budget_num, followers_num)

        explain_with_llm()

    except Exception as e:
        st.error(f"Prediction failed: {e}\n{traceback.format_exc()}")

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
                   use_container_width=True):
                st.session_state.feedback = "Like"
                st.session_state.feedback_submitted = True
                save_feedback_to_dataset(st.session_state.strategy, "Like")
                st.rerun()
        with col2:
            if st.button("👎 Ok-",
                   key="dislike_btn",
                   help="This needs work",
                   use_container_width=True):
                st.session_state.feedback = "Dislike"
                st.session_state.feedback_submitted = True
                save_feedback_to_dataset(st.session_state.strategy, "Dislike")
                st.rerun()
            st.markdown('<style>.element-container button[data-testid="dislike_btn"] {background-color: #e74c3c !important; color: #fff !important;}</style>', unsafe_allow_html=True)

    if st.session_state.feedback_submitted:
        if st.session_state.feedback == "Like":
            st.success('Thank you for your feedback! ❤️')
        elif st.session_state.feedback == "Dislike":
            st.info('Aw! We will work on improving. Thanks for your feedback.')

