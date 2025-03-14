import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from streamlit_option_menu import option_menu
from video_object_detection import VideoObjectDetection
from image_object_detection import ImageObjectDetection
from streamlit_login_auth_ui.widgets import __login__
# from facial_emotion_recognition import FacialEmotionRecognition
from hand_gesture_classification import HandGestureClassification
from image_optical_character_recgonition import ImageOpticalCharacterRecognition
from image_classification import ImageClassification
from video_utils import create_video_frames
import plotly.express as px
from PIL import Image
from io import BytesIO
import base64
import json
import os
import cv2
import numpy as np
import matplotlib.cm
import random
import time

# from streamlit_webrtc import (
#     RTCConfiguration,
#     WebRtcMode,
#     WebRtcStreamerContext,
#     webrtc_streamer,
# )

# 🔐 Initialize Login System
__login__obj = __login__(
    auth_token="your_courier_auth_token",  # Replace this with your actual Courier API key
    company_name="My App",
    width=200,
    height=250,
    logout_button_name="Logout",
    hide_menu_bool=False,
    hide_footer_bool=False,
    lottie_url="https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json"
)

# Check if user is logged in
LOGGED_IN = __login__obj.build_login_ui()

# 🚀 If user is logged in, allow access to app
if LOGGED_IN:
    st.success("Welcome! You are logged in.")


    # Hide warnings to make it easier to locate
    # errors in logs, should they show up
    import warnings
    warnings.filterwarnings("ignore")



    # Hide Streamlit logo
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Make Radio buttons horizontal
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # Functions to load models
    # @st.cache(allow_output_mutation=True)
    # def load_video_object_detection():
    #     return VideoObjectDetection()

    # @st.cache(allow_output_mutation=True)
    # def load_image_object_detection():
    #     return ImageObjectDetection()

    @st.cache_resource
    def load_image_classifier():
        return ImageClassification()

    # # @st.cache(allow_output_mutation=True)
    # # def load_facial_emotion_classifier():
    # #     return FacialEmotionRecognition()

    # @st.cache(allow_output_mutation=True)
    # def load_hand_gesture_classifier():
    #     return HandGestureClassification()

    @st.cache_resource
    def load_image_optical_character_recognition():
        return ImageOpticalCharacterRecognition()


    # Load models and store in cache
    # video_object_detection = load_video_object_detection()
    # image_object_detection = load_image_object_detection()
    # facial_emotion_classifier = load_facial_emotion_classifier()
    # hand_gesture_classifier = load_hand_gesture_classifier()
    image_optical_character_recognition = load_image_optical_character_recognition()
    image_classifier = load_image_classifier()


    # Paths for video examples
    video_examples = {'Traffic': 'examples/Traffic.mp4',
                    'Elephant': 'examples/Elephant.mp4',
                    'Airport': 'examples/Airport.mp4',
                    'Kanye': 'examples/Kanye.mp4',
                    'Laughing Guy': 'examples/Laughing Guy.mp4',
                    'Parks and Recreation': 'examples/Parks and Recreation.mp4'}

    # Create streamlit sidebar with options for different tasks
    with st.sidebar:
        page = option_menu(menu_title='Menu',
                        menu_icon="robot",
                        options=["Welcome!",
                                    "Image Classification",
                                    "Chatbot"],
                        icons=["house-door",
                                "search",
                                "chat"],
                        default_index=0,
                        )

        # Make sidebar slightly larger to accommodate larger names
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
                width: 350px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


    st.title('Open-source Computer Vision')

    # Load and display local gif file
    #file_ = open("resources/camera-robot-eye.gif", "rb")
    #contents = file_.read()
    #data_url = base64.b64encode(contents).decode("utf-8")
    #file_.close()

    # Page Definitions
    if page == "Welcome!":

        # Page info display
        # st.header('Welcome!')
        # st.markdown(
        #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        #     unsafe_allow_html=True,
        # )

        st.subheader('Quickstart')
        st.write(
            """
            Flip through the pages in the menu on the left hand side bar to perform CV tasks on-demand!
            
            Run computer vision tasks on:
            
                * Images
                    * Examples
                    * Upload your own
                * Video
                    * Webcam
                    * Examples
                    * Upload your own
            """
        )

        st.subheader("Introduction")
        st.write("""
        This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
            """

                )
    elif page == 'Optical Character Recognition':

        # Page info display
        st.header('Image Optical Character Recognition')
        st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

        # User selected option for data type
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))
      
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('Submit'):
            # Run OCR
            with st.spinner("Running optical character recognition..."):
                annotated_image, text = image_optical_character_recognition.image_ocr(uploaded_file)

            # Create image buffer and download
            buf = BytesIO()
            annotated_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            # Display and provide download option for annotated image
            st.subheader("Captioning Prediction")
            st.image(annotated_image)
            if text == '':
                st.wite("No text in this image...")
            else:
                st.write(text)

                st.download_button('Download Text', data=text, file_name='outputs/ocr_pred.txt')

    elif page == 'Image Classification':

        # Page info display
        st.header('Image Classification')
        st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

        # User selected option for data type
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

        # Provide option to use example or upload your own
        if input_type == 'Example':
            option = st.selectbox(
                'Which example would you like to use?',
                ('Car', 'Dog', 'Tropics'))
            uploaded_file = image_examples[option]
        else:
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('Submit!'):
            # Throw error if there is no file
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                # Run classification
                with st.spinner("Running classification..."):
                    img = Image.open(uploaded_file)
                    preds = image_classifier.classify(img)

                # Display image
                st.subheader("Classification Predictions")
                st.image(img)
                fig = px.bar(preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
                st.write(fig)

                # Provide download option for predictions
                st.write("")
                csv = preds.to_csv(index=False).encode('utf-8')
                st.download_button('Download Predictions',csv,
                                file_name='classification_predictions.csv')
                
    elif page == "Chatbot":

        st.header("Chatbot")

        # File uploader form
        with st.form("my-form", clear_on_submit=True):
            file = st.file_uploader("Upload an image to classify...", type=["jpg", "jpeg", "png"])
            submitted = st.form_submit_button("UPLOAD!")

        # Streamed response emulator
        def response_generator():
            response = random.choice(
                [
                    "Hello there! Do you have an image that I can classify?",
                    "Hi! Is there an image I can help you with?",
                    "Upload an image and I can help you with that!",
                    "I'm here to help! Just upload an image.",
                    "Need assistance with an image?",
                    "I'm ready to classify an image for you!",
                    "What's up! I can help you with image classification."
                ]
            )
            for word in response.split():
                yield word + " "
                time.sleep(0.05)

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "user" and "image" in message:
                        st.image(message["image"], caption="Uploaded an image.")

        if submitted and file is not None:
            st.session_state.messages.append({"role": "user", "content": "Uploaded an image.", "image": file})
            with st.chat_message("user"):
                st.image(file, caption="Uploaded an image.")
            # Chatbot response for image upload
            st.session_state.messages.append({"role": "assistant", "content": "Processing image..."})
            with st.chat_message("assistant"):
                st.markdown("Processing image...")
                # The classifier is defined as a global var 
                preds = image_classifier.classify(file)
                st.write(preds)

        # Chatbot response for text input
        if prompt := st.chat_input("Upload an image or say something..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Chat random response
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator())
            st.session_state.messages.append({"role": "assistant", "content": response})