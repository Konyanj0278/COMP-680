import streamlit as st
from streamlit_option_menu import option_menu

from login.login import __login__

from src.model import ImageClassification
import plotly.express as px
from src.image_object_detection import ImageObjectDetection
from src.image_optical_character_recgonition import ImageOpticalCharacterRecognition
from PIL import Image
import random
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np

from datetime import datetime
from io import BytesIO
import json
from base64 import b64encode

# --- Animated Background Styling ---
with open("assets/background.jpg", "rb") as img_file:
    img_bytes = img_file.read()
with open("assets/background.jpg", "rb") as img_file:
    img_bytes = img_file.read()
    encoded = b64encode(img_bytes).decode()

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
    background-color: rgba(0, 0, 0, 0.35); /* Add strong base black tint */
    background-blend-mode: overlay; /* Blend image + dark tint */
    }}

    /* Dim the main background slightly for contrast */
    [data-testid="stAppViewContainer"] > .main {{
        background-color: rgba(0, 0, 0, 0.45);  /* Soft black overlay */
    backdrop-filter: blur(0px);
    padding: 8rem;
    border-radius: 30px;
    box-shadow: 0 0 50px rgba(0, 0, 0, 0.6);
    }}

    /* Sidebar container styling */
    section[data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.35) !important;
        backdrop-filter: blur(90px);
        border-right: 30px solid rgba(255, 255, 255, 0.05);
        box-shadow: 40px 0 150px rgba(0, 0, 0, 0.4);
    }}

    /* Sidebar menu items */
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

    /* Headings and text */
    body, h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: #FFFFFF !important;
        text-shadow: 0 0 4px rgba(255, 255, 255, 0.75);
    }}

    /* Widget container styling */
    .stContainer {{
        background-color: rgba(0, 0, 0, 0.4);
        border-radius: 30px;
        padding: 30px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.5);
    }}

    /* Button hover */
    button:hover {{
        box-shadow: 0 0 10px rgba(0, 255, 255, 0.6);
        background-color: rgba(0, 255, 255, 0.1);
    }}
    </style>
""", unsafe_allow_html=True)




=======
from datetime import datetime  # Add this import for timestamps

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

            options=["Welcome!", "Image Classification", "Object Detection", "Chatbot", "Computer Vision"],
            
            icons=["house-door", "search", "camera", "chat", "brain"],

            menu_icon="robot",
            default_index=0
        )


        # Add Confidence Threshold Slider
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust the minimum confidence level for predictions to be displayed."
        )

    st.title("We Missed you ! 😊")

    # --- Welcome page with animation ---

    st.title("Deep Net")

    # Page Routing

    if page == "Welcome!":
        st.markdown("""
            <style>
            .welcome-title {
                font-size: 4rem;
                font-weight: bold;
                color: white;
                text-align: center;
                animation: glowFade 2s ease-in-out infinite alternate;
                margin-bottom: 1rem;
            }

            .fade-in-section {
                animation: fadeIn 1.5s ease-in-out;
                margin-top: 20px;
                font-size: 1.1rem;
            }

            @keyframes glowFade {
                from {
                    text-shadow: 0 0 10px #0ff, 0 0 20px #0ff;
                }
                to {
                    text-shadow: 0 0 20px #0ff, 0 0 40px #0ff;
                }
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(15px); }
                to { opacity: 1; transform: translateY(0); }
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="welcome-title"> Welcome to Deep Net</div>', unsafe_allow_html=True)

        st.markdown('<div class="fade-in-section">', unsafe_allow_html=True)



        st.subheader('Quickstart')
        st.write("Use the navigation tab on the left hand side to visit different links.")

        st.subheader("Introduction")
        st.markdown("""
            <style>
            .intro-text {
            font-size: 1.5rem;
            line-height: 1.8;
            color: white;
            text-align: justify;
            margin-top: 20px;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="intro-text">
            This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. 
            It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements ensure a smoother user experience.
            </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "Object Detection":
        st.header('Object Detection')
        st.write("This object detection app uses YOLOv8, a state-of-the-art model for real-time object detection.")

        # User selected option for data type
        data_type = st.radio(
            "Select Data Type",
            ('Webcam', 'Video', 'Image')
        )

        if data_type == 'Image':
            input_type = st.radio(
                "Use example or upload your own?",
                ('Example', 'Upload')
            )

            # Define example images
            image_examples = {
                'Home Office': 'path/to/home_office.jpg',
                'Traffic': 'path/to/traffic.jpg',
                'Barbeque': 'path/to/barbeque.jpg'
            }

            # Load in example or uploaded image
            if input_type == 'Example':
                option = st.selectbox(
                    'Which example would you like to use?',
                    ('Home Office', 'Traffic', 'Barbeque')
                )
                uploaded_file = image_examples[option]
            else:
                uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

            # Run detection and provide download options when user clicks run!
            if st.button('🔥 Run!'):
                if uploaded_file is None:
                    st.error("No file uploaded yet.")
                else:
                    with st.spinner("Running object detection..."):
                        img = Image.open(uploaded_file)
                        image_object_detection = ImageObjectDetection()
                        labeled_image, detections = image_object_detection.classify(img)

                        # Filter detections based on confidence threshold
                        filtered_detections = [
                            det for det in detections if det['score'] >= confidence_threshold
                        ]

                    if labeled_image and filtered_detections:
                        buf = BytesIO()
                        labeled_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        st.subheader("Object Detection Predictions")
                        st.image(labeled_image)
                        st.download_button('Download Image', data=byte_im, file_name="image_object_detection.png", mime="image/jpeg")

                        st.json(filtered_detections)
                        st.download_button('Download Predictions', json.dumps(filtered_detections), file_name='image_object_detection.json')

    elif page == 'Image Classification':
        st.header('Image Classification')

        # User selected option for data type
        input_type = st.radio(
            "Use example or upload your own?",
            ('Example', 'Upload')
        )

        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('Submit!'):
            if uploaded_file is None:
                st.error("No file uploaded yet.")
            else:
                with st.spinner("Running classification..."):
                    img = Image.open(uploaded_file)
                    preds = image_classifier.classify(img)

                    # Filter predictions based on confidence threshold
                    filtered_preds = preds[preds['Pred_Prob'] >= confidence_threshold]

                st.subheader("Classification Predictions")
                st.image(img)
                fig = px.bar(filtered_preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
                st.write(fig)

                # Provide download option for predictions
                csv = filtered_preds.to_csv(index=False).encode('utf-8')
                st.download_button('Download Predictions', csv, file_name='classification_predictions.csv')

    elif page == "Chatbot":
        st.header("Chatbot")

        # Helper function to format timestamps
        def get_formatted_timestamp():
            return datetime.now().strftime("%I:%M %p")  # 12-hour format without date

        # Cache model once
        @st.cache_resource
        def load_yolo_model():
            return YOLO("yolov8n.pt")

        model = load_yolo_model()

        # Chat history state init
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        with st.container():
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(message["timestamp"])
                    if message["role"] == "user" and "image" in message:

                        st.image(message["image"], caption="Uploaded an image.", use_container_width=True)
                    elif message["role"] == "assistant" and "image" in message:
                        st.image(message["image"], caption="Detected Objects", use_container_width=True)
       
                def get_formatted_timestamp():
                    """Returns the current timestamp in a formatted string."""
                    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.divider()

        # Chat Input Handling
        with st.container():
            if prompt := st.chat_input("Upload an image or say something..."):
                user_timestamp = get_formatted_timestamp()
                st.session_state.messages.append({
                    "role": "user",
                    "content": prompt,
                    "timestamp": user_timestamp
                })
                with st.chat_message("user"):
                    st.markdown(prompt)
                    st.caption(user_timestamp)

                # Stream assistant response manually
                with st.chat_message("assistant"):
                    response = random.choice([
                        "Hello there! Do you have an image that I can classify?",
                        "Hi! Is there an image I can help you with?",
                        "Upload an image and I can help you with that!",
                        "I'm here to help! Just upload an image.",
                        "Need assistance with an image?",
                        "I'm ready to classify an image for you!",
                        "What's up! I can help you with image classification."
                    ])
                    st.markdown(response)
                    bot_timestamp = get_formatted_timestamp()
                    st.caption(bot_timestamp)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.strip(),
                    "timestamp": bot_timestamp
                })
                st.rerun()

            # Image uploader block
            with st.form("my-form", clear_on_submit=True):
                file = st.file_uploader("Choose image...", type=["jpg", "jpeg", "png"])
                submitted = st.form_submit_button("SUBMIT")


            if submitted and file is not None:
                # Save user image message
                timestamp = get_formatted_timestamp()
                st.session_state.messages.append({
                    "role": "user",
                    "content": "Uploaded an image.",
                    "image": file,
                    "timestamp": timestamp
                })
                with st.chat_message("user"):
                    st.image(file, caption="Uploaded an image.", use_container_width=True)
                    st.caption(timestamp)

                # Assistant: Processing response
                with st.chat_message("assistant"):
                    processing_time = get_formatted_timestamp()
                    st.markdown("Processing image...")
                    st.caption(processing_time)

                    # Process image with YOLO
                    image = Image.open(file).convert("RGB")
                    img_array = np.array(image)

                    # Define and load the YOLO model
                    model = YOLO("yolov8n.pt")  # Load YOLOv8n model

                    results = model(img_array)
                    annotated_img = results[0].plot()

                    # Show result
                    st.markdown("Here are the detected objects:")
                    st.image(annotated_img, caption="Detected Objects", use_container_width=True)
                    result_time = get_formatted_timestamp()
                    st.caption(result_time)

                # Append result message to state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Here are the detected objects:",
                    "image": annotated_img,
                    "timestamp": result_time
                })
                st.rerun()


    elif page == "Computer Vision":
        st.header("🧠 Computer Vision")
        st.subheader("📷 Object Detection using YOLOv8")

        from ultralytics import YOLO
        import numpy as np
        import cv2
        from PIL import Image
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, ClientSettings


        # File upload
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            # Load image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Original Image", use_container_width=True)

            # Convert PIL image to numpy array
            img_array = np.array(image)

            # Load YOLOv8n model (small, fast, downloads first time)
            model = YOLO("yolov8n.pt")

            # Run object detection
            st.write("Running YOLO object detection...")
            results = model(img_array)

            # Get annotated image
            annotated_img = results[0].plot()

            # Show result
            st.image(annotated_img, caption="Detected Objects", use_container_width=True)

        # ---- WEBCAM OBJECT DETECTION BLOCK ----
        model = YOLO("yolov8n.pt")  # load once

        st.subheader("🎥 Real-time Object Detection via Webcam")

        class VideoProcessor(VideoTransformerBase):
            def transform(self, frame):
                # Get webcam frame as ndarray
                img = frame.to_ndarray(format="bgr24")


                # Run YOLO on frame
                results = model(img)


                # Plot the annotated results
                annotated_frame = results[0].plot()

                # Convert NumPy array back to video frame
                return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


        st.info("👆 If the webcam doesn't start, try selecting your camera manually from the dropdown.")

        # Streamlit UI block to start webcam
        webrtc_streamer(
            key="webcam",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
