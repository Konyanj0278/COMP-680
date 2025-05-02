import streamlit as st
from streamlit_option_menu import option_menu
from login.login import __login__
from ultralytics import YOLO
from src.model import ImageClassification

# Load custom screens
from screens import welcome, image_classification, chatbot, computer_vision

# --- Animated Background Styling ---
from base64 import b64encode
with open("assets/background.jpg", "rb") as img_file:
    encoded = b64encode(img_file.read()).decode()

st.markdown(f"""
    <style>
    @keyframes scrollBackground {{
        0% {{ background-position: 100% 0%; }}
        100% {{ background-position: 20% 100%; }}
    }}

    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-attachment: scroll;
        background-size: cover;
        background-position: center;
        animation: scrollBackground 30s infinite linear;
        background-color: rgba(0, 0, 0, 0.35);
        background-blend-mode: overlay;
    }}

    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(0px);
        padding: 8rem;
        border-radius: 30px;
        box-shadow: 0 0 50px rgba(0, 0, 0, 0.6);
    }}

    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(90px);
        border-right: 30px solid rgba(255, 255, 255, 0.05);
        box-shadow: 40px 0 150px rgba(0, 0, 0, 0.4);
    }}

    .css-1d391kg, .css-1v0mbdj, .css-18e3th9 {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.25);
    }}

    .css-1v0mbdj:hover, .css-1d391kg:hover {{
        transform: scale(1.02);
        background-color: rgba(255, 255, 255, 0.05) !important;
        transition: all 0.3s ease-in-out;
    }}

    .css-1v0mbdj:has(> .css-1lcbmhc) {{
        background-color: rgba(0, 255, 255, 0.2) !important;
        box-shadow: 0 0 12px rgba(0, 255, 255, 0.6);
    }}

    body, h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: #FFFFFF !important;
        text-shadow: 0 0 4px rgba(255, 255, 255, 0.75);
    }}

    .stContainer {{
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
    }}

    button:hover {{
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.6);
        background-color: rgba(0, 255, 255, 0.1);
    }}
    </style>
""", unsafe_allow_html=True)

# --- Login Setup ---
__login__obj = __login__(
    auth_token="your_courier_auth_token",
    company_name="Deep Net",
    width=200,
    height=250,
    logout_button_name="Logout",
    hide_menu_bool=False,
    hide_footer_bool=False
)

LOGGED_IN = __login__obj.build_login_ui()

# --- Authenticated Flow ---
if LOGGED_IN:
    st.success("Welcome! You are logged in.")
    st.markdown("""<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>""", unsafe_allow_html=True)
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # Load models once
    @st.cache_resource
    def load_models():
        return ImageClassification(), YOLO("yolov8n.pt")

    image_classifier, yolo_model = load_models()

    # Sidebar Navigation
    with st.sidebar:
        page = option_menu(
            menu_title='Menu',
            options=["Welcome!", "Image Classification", "Chatbot", "Computer Vision"],
            icons=["house-door", "search", "camera", "chat", "brain"],
            menu_icon="robot",
            default_index=0
        )

        # Add Confidence Threshold Slider (shared globally)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence level for predictions"
        )

    # Page Routing
    if page == "Welcome!":
        welcome.show()
    elif page == "Image Classification":
        image_classification.show(image_classifier, confidence_threshold)
    # elif page == "Object Detection":
    #  object_detection.show(yolo_model, confidence_threshold)
    elif page == "Chatbot":
        chatbot.show(image_classifier, yolo_model)
    elif page == "Computer Vision":
        computer_vision.show()
