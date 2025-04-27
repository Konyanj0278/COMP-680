import streamlit as st
from streamlit_option_menu import option_menu
from login.login import __login__
from src.model import ImageClassification
import plotly.express as px
from src.image_object_detection import ImageObjectDetection
from src.model import ImageClassification
from src.image_optical_character_recgonition import ImageOpticalCharacterRecognition

from PIL import Image
import random
import time
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av
import cv2
import numpy as np
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
            options=["Welcome!", "Image Classification", "Chatbot", "Computer Vision"],
            icons=["house-door", "search", "chat", "camera"],
            menu_icon="robot",
            default_index=0
        )

    st.title("Deep Net")

    # Page Routing
    if page == "Welcome!":




        st.subheader('Quickstart')
        st.write("Use the navigation tab on the left hand side to visit different links.")

        st.subheader("Introduction")
        st.write("""
        This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
            """

                )
    elif page == "Object Detection":
        st.header('Object Detection')
        st.markdown("![Alt Text](https://media.giphy.com/media/vAvWgk3NCFXTa/giphy.gif)")
        st.write("This object detection app uses YOLOv8, a state-of-the-art model for real-time object detection. Try it out!")

        # User selected option for data type
        data_type = st.radio(
            "Select Data Type",
            ('Webcam', 'Video', 'Image'))

        if data_type == 'Image':
            input_type = st.radio(
                "Use example or upload your own?",
                ('Example', 'Upload'))

            # Load in example or uploaded image
            if input_type == 'Example':
                option = st.selectbox(
                    'Which example would you like to use?',
                    ('Home Office', 'Traffic', 'Barbeque'))
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
                        image_object_detection = load_image_object_detection()
                        labeled_image, detections = image_object_detection.classify(img)

                    if labeled_image and detections:
                        buf = BytesIO()
                        labeled_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        st.subheader("Object Detection Predictions")
                        st.image(labeled_image)
                        st.download_button('Download Image', data=byte_im, file_name="image_object_detection.png", mime="image/jpeg")

                        st.json(detections)
                        st.download_button('Download Predictions', json.dumps(detections), file_name='image_object_detection.json')

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
