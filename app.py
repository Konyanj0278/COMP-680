import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_login_auth_ui.widgets import __login__
from src.model import ImageClassification
from ultralytics import YOLO

# Load custom modules
from screens import welcome, image_classification, chatbot, computer_vision

# Login Setup
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

# Authenticated Flow
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
            icons=["house-door", "search", "chat", "camera"],
            menu_icon="robot",
            default_index=0
        )

    st.title("Deep Net")

    # Page Routing
    if page == "Welcome!":
        welcome.show()
    elif page == "Image Classification":
        image_classification.show(image_classifier)
    elif page == "Chatbot":
        chatbot.show(image_classifier, yolo_model)
    elif page == "Computer Vision":
        computer_vision.show()
