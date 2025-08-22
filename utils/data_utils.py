import pandas as pd
import os
from datetime import datetime
import streamlit as st

from utils.config import size_names, trust_questions

industry_opts  = [
 'Food & Beverage', 'Retail', 'Digital Marketing',
 'Gaming Service', 'Career', 'Hospitality',
 'Personal Care', 'Educational Consultant',
 'Catering', 'Lifestyle', 'Vet', 'Farming',
 'Event management', 'Designing', 'Catholic Book Store',
 'Sports', 'Fashion', 'Fitness & GYM'
]

def map_budget(v: float) -> str:
    if v < 500:
        return "<500"
    elif v <= 1000:
        return "500-1000"
    elif v <= 5000:
        return "1000-5000"
    else:
        return ">5000"

def map_followers(v: float) -> str:
    if v < 1000:
        return "<500"
    elif v <= 5000:
        return "1000-5000"
    elif v <= 10000:
        return "5000-10000"
    else:
        return ">10000"

import os
import pandas as pd

def save_to_dataset(row, strategy, confidence=None, feedback=False, threshold=80, file_path="data/new_data.csv"):
    """
    Save the row with predicted strategy to a CSV only if:
    - feedback=True OR
    - confidence is provided and >= threshold (%)
    """
    # Decide if we should save
    if feedback or (confidence is not None and confidence >= threshold):
        # Add predicted strategy to row
        row_with_strategy = row.copy()
        row_with_strategy["Primary Marketing Strategy"] = strategy

        # Load existing data or create a new DataFrame
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            # If file doesn't exist, create with appropriate columns
            cols = list(row_with_strategy.keys())
            df = pd.DataFrame(columns=cols)

        # Append and save
        df = pd.concat([df, pd.DataFrame([row_with_strategy])], ignore_index=True)
        df.to_csv(file_path, index=False)
        print(f"Row saved! (confidence={confidence}, feedback={feedback})")
    else:
        print(f"Skipped saving: low confidence ({confidence}) and no feedback.")


def reset_form(st):
    st.session_state.size_name = size_names[0]
    st.session_state.maturity = 1
    st.session_state.industry = industry_opts[0] if industry_opts else "Other"
    st.session_state.budget_num = 0
    st.session_state.followers_num = 0
    for qui in trust_questions.keys():
        st.session_state[qui] = 1
    st.session_state.business_name = ""
    st.session_state.feedback_submitted = False
    st.session_state.feedback = None