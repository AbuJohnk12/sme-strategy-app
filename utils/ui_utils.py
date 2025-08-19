def tooltip(help_text):
    return f"""<span class="tooltip">
        <sup>?</sup>
        <span class="tooltiptext">{help_text}</span>
    </span>
    """

def inject_css(st):
    st.markdown("""
    <style>
        .tooltip {
            position: relative;
            display: inline-flex;
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
        .red-btn {
            background-color: #e74c3c !important;
            color: #fff !important;
            border: none !important;
        }
    </style>
    """, unsafe_allow_html=True)