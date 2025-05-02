import streamlit as st
from streamlit_login_auth_ui.widgets import __login__
from streamlit_option_menu import option_menu
from src.model import ImageClassification
import plotly.express as px
from PIL import Image
import random
import time
import av
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode


# 🔐 Initialize Login System
__login__obj = __login__(
    auth_token="your_courier_auth_token",  # Replace this with your actual Courier API key
    company_name="Deep Net",
    width=200,
    height=250,
    logout_button_name="Logout",
    hide_menu_bool=False,
    hide_footer_bool=False,

)

# Check if user is logged in
LOGGED_IN = __login__obj.build_login_ui()

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


    @st.cache_resource
    def load_image_classifier():
        return ImageClassification()

    image_classifier = load_image_classifier()

    # Create streamlit sidebar with options for different tasks
    with st.sidebar:
        page = option_menu(menu_title='Menu',
                        menu_icon="robot",
                        options=["Welcome!",
                                    "Image Classification",
                                    "Chatbot",
                                    "Computer Vision"],
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


    st.title('Deep Net')


    # Page Definitions
    if page == "Welcome!":



        st.subheader('Quickstart')
        st.write("Use the navigation tab on the left hand side to visit different links.")

        st.subheader("Introduction")
        st.write("""
        This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
            """

                )

    elif page == 'Image Classification':

        # Page info display
        st.header('Image Classification')
        # User selected option for data type
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload'))

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
            
    elif page == "Computer Vision":
        st.header("🧠 Computer Vision")

        # ✅ SECTION 1: IMAGE UPLOAD + OBJECT DETECTION
        st.subheader("📷 Object Detection using YOLOv8")

        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)

            img_array = np.array(image)
            yolo_model = YOLO("yolov8n.pt")

            with st.spinner("Running YOLO object detection..."):
                results = yolo_model(img_array)
                annotated_img = results[0].plot()

            st.image(annotated_img, caption="Detected Objects", use_container_width=True)
        
        #Webcam     
        # ✅ SECTION 2: Webcam REAL-TIME OBJECT DETECTION USING OPENCV
        st.subheader("🎥 Real-time Object Detection via Webcam (LOCAL ONLY)")

        # Load YOLOv8 model once
        yolo_model = YOLO("yolov8n.pt")

        run_webcam = st.checkbox('Start Webcam')

        FRAME_WINDOW = st.image([])

        if run_webcam:
            # Initialize webcam
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("❌ Could not open webcam. Please check your camera.")
            else:
                st.info("✅ Webcam is running. Close the app to release the webcam.")

                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to grab frame.")
                        break

                    # Run YOLO on the frame
                    results = yolo_model(frame)
                    annotated_frame = results[0].plot()

                    # Convert for display
                    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(annotated_frame)

                cap.release()
