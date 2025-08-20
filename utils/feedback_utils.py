import pandas as pd
import os
from datetime import datetime
import streamlit as st

def save_feedback_to_dataset(strategy, feedback):
    feedback_file = "feedback_data.csv"
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": strategy,
        "feedback": feedback,
        "Business Size (No.of Employees)": st.session_state.get("size_name", ""),
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
