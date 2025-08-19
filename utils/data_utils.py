import pandas as pd
import os
from datetime import datetime
import streamlit as st

def map_budget(v: float) -> str:
    if v < 500:
        return "<500"
    elif v <= 1000:
        return "500-1000"
    elif v <= 5000:
        return "5000"
    else:
        return "None"

def map_followers(v: float) -> str:
    if v < 500:
        return "<500"
    elif v <= 1000:
        return "500-1000"
    elif v <= 5000:
        return "1000-5000"
    else:
        return ">5000"

def save_to_dataset(input_data, prediction, confidence, feature_cols, size_name, size_map_name_to_val, maturity_col, maturity, industry, budget_cat, followers_cat, budget_num, followers_num):
    try:
        record = {col: 0 for col in feature_cols if col not in [
            "Business Size", 
            maturity_col,
            "Primary Marketing Strategy"
        ]}
        record["Business Size"] = size_map_name_to_val[size_name]
        record[maturity_col] = maturity
        record["Primary Marketing Strategy"] = prediction

        trust_questions_list = [
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
        for question in trust_questions_list:
            if question in input_data:
                record[question] = input_data[question]

        channels = [
            "Company Website", "Facebook", "Google Ads", "Instagram", 
            "LinkedIn", "Online Store", "Pinterest", "Through Customers",
            "TikTok", "X", "Youtube", "x"
        ]
        for channel in channels:
            if channel in input_data:
                record[channel] = input_data[channel]

        record[f"Industry_{industry}"] = 1
        record[f"Budget_{budget_cat}"] = 1
        record[f"Followers_{followers_cat}"] = 1

        record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record["Digital Marketing Maturity"] = maturity
        record["Industry"] = industry
        record["Budget"] = budget_num
        record["Followers"] = followers_num
        record["Predicted Strategy"] = prediction
        record["Confidence"] = confidence if confidence else None
        record["Budget Category"] = budget_cat
        record["Followers Category"] = followers_cat

        new_record = pd.DataFrame([record])
        for col in feature_cols:
            if col not in new_record.columns:
                new_record[col] = 0
        new_record = new_record[feature_cols + [
            "Digital Marketing Maturity", "Industry", 
            "Budget", "Followers", "Predicted Strategy", "Confidence",
            "Budget Category", "Followers Category"
        ]]
        file_path = "processed_dataset.csv"
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_record], ignore_index=True)
        else:
            updated_data = new_record
        updated_data.to_csv(file_path, index=False)
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")

def reset_form(st, trust_questions, feature_cols, industry_opts):
    st.session_state.size_name = "Small"
    st.session_state.maturity = 3
    st.session_state.industry = industry_opts[0] if industry_opts else "Other"
    st.session_state.budget_num = 1500
    st.session_state.followers_num = 1200
    for question in trust_questions.values():
        if question in feature_cols:
            st.session_state[question] = 3
    st.session_state.feedback_submitted = False
    st.session_state.feedback = None