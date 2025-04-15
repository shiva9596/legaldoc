# Add this at the top after setting Streamlit config
st.markdown("""
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(to right, #e0f7fa, #fce4ec);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title style */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 2.8em;
        margin-top: 0.5em;
        margin-bottom: 0.3em;
    }

    /* File uploader and buttons */
    .stFileUploader, .stTextInput, .stButton {
        background-color: rgba(255, 255, 255, 0.75);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }

    /* Markdown suggestions */
    .element-container h3 {
        color: #6a1b9a;
    }

    /* Answer block */
    .stMarkdown {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Info box */
    .stInfo {
        background-color: rgba(230, 244, 255, 0.7);
        color: #0277bd;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
