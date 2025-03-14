import streamlit as st
from streamlit_option_menu import option_menu
from image_object_detection import ImageObjectDetection
from image_classification import ImageClassification
from image_optical_character_recgonition import ImageOpticalCharacterRecognition
from video_object_detection import VideoObjectDetection
from video_utils import create_video_frames
from PIL import Image
import cv2
import numpy as np
import base64
import json
import os
import av
from io import BytesIO
import plotly.express as px
from transformers import AutoFeatureExtractor
import timm
import torch
from torchvision import transforms

# Hide warnings
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

# Load models
@st.cache(allow_output_mutation=True)
def load_image_object_detection():
    return ImageObjectDetection()

@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()

@st.cache(allow_output_mutation=True)
def load_image_optical_character_recognition():
    return ImageOpticalCharacterRecognition()

@st.cache(allow_output_mutation=True)
def load_video_object_detection():
    return VideoObjectDetection()

# Paths for image examples
image_examples = {
    'Traffic': 'examples/Traffic.jpeg',
    'Barbeque': 'examples/Barbeque.jpeg',
    'Home Office': 'examples/Home Office.jpeg',
    'Car': 'examples/Car.jpeg',
    'Dog': 'examples/Dog.jpeg',
    'Tropics': 'examples/Tropics.jpeg',
    'Quick Brown Dog': 'examples/Quick Brown Dog.png',
    'Receipt': 'examples/Receipt.png',
    'Street Sign': 'examples/Street Sign.jpeg',
    'Kanye': 'examples/Kanye.png',
    'Shocked': 'examples/Shocked.png',
    'Yelling': 'examples/Yelling.jpeg'
}

# Paths for video examples
video_examples = {
    'Traffic': 'examples/Traffic.mp4',
    'Elephant': 'examples/Elephant.mp4',
    'Airport': 'examples/Airport.mp4',
    'Kanye': 'examples/Kanye.mp4',
    'Laughing Guy': 'examples/Laughing Guy.mp4',
    'Parks and Recreation': 'examples/Parks and Recreation.mp4'
}

# Create streamlit sidebar with options for different tasks
with st.sidebar:
    page = option_menu(
        menu_title='Menu',
        menu_icon="robot",
        options=["Welcome!", "Object Detection", "Image Classification", "Optical Character Recognition"],
        icons=["house-door", "search", "emoji-smile", "eyeglasses"],
        default_index=0,
    )

    # Add confidence threshold slider
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

    # Make sidebar slightly larger
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

# Commented out the missing GIF file code
# file_ = open("resources/camera-robot-eye.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

# Page Definitions
if page == "Welcome!":
    st.header('Welcome!')
    # Commented out the GIF display
    # st.markdown(
    #     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    #     unsafe_allow_html=True,
    # )

    st.subheader('Quickstart')
    st.write(
        """
        Flip through the pages in the menu on the left-hand sidebar to perform CV tasks on-demand!

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
       This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and object detection. It utilizes state-of-the-art models like YOLOv8 and EfficientNet-B7 to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
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
    st.header('Image Classification')
    st.markdown("![Alt Text](https://media.giphy.com/media/Zvgb12U8GNjvq/giphy.gif)")

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Car', 'Dog', 'Tropics'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        if uploaded_file is None:
            st.error("No file uploaded yet.")
        else:
            with st.spinner("Running classification..."):
                img = Image.open(uploaded_file)
                image_classifier = load_image_classifier()
                preds = image_classifier.classify(img)

            st.subheader("Classification Predictions")
            st.image(img)
            fig = px.bar(preds.sort_values("Pred_Prob", ascending=True), x='Pred_Prob', y='Class', orientation='h')
            st.write(fig)

            csv = preds.to_csv(index=False).encode('utf-8')
            st.download_button('Download Predictions', csv, file_name='classification_predictions.csv')

elif page == 'Optical Character Recognition':
    st.header('Image Optical Character Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Quick Brown Dog', 'Receipt', 'Street Sign'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
        with st.spinner("Running optical character recognition..."):
            image_ocr = load_image_optical_character_recognition()
            annotated_image, text = image_ocr.image_ocr(uploaded_file)

        buf = BytesIO()
        annotated_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.subheader("OCR Predictions")
        st.image(annotated_image)
        if text == '':
            st.write("No text in this image...")
        else:
            st.write(text)
            st.download_button('Download Text', data=text, file_name='outputs/ocr_pred.txt')

class ImageClassification:
    def __init__(self):
        # Load EfficientNet-B0 from timm
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, image):
        # Preprocess the image
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
        # Convert outputs to probabilities and class labels
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        return probabilities

import streamlit as st
from image_classification import ImageClassification
from image_optical_character_recgonition import ImageOpticalCharacterRecognition
from PIL import Image

# Sidebar menu
page = st.sidebar.selectbox("Select a page", ["Welcome", "Image Classification", "Optical Character Recognition (OCR)"])

if page == "Welcome":
    # Welcome Page
    st.title("Welcome to the Application!")
    st.write("Navigate to the sidebar to perform tasks.")

elif page == "Image Classification":
    # Image Classification Page
    st.title("Image Classification")

    # File uploader for classification
    uploaded_file_classification = st.file_uploader("Upload an image for classification", type=["png", "jpg", "jpeg"], key="classification_uploader")

    if uploaded_file_classification is not None:
        # Convert the uploaded file to a PIL image
        image = Image.open(uploaded_file_classification)

        # Initialize the ImageClassification class
        image_classifier = ImageClassification()

        # Perform classification
        predictions = image_classifier.classify(image)

        # Display results
        st.subheader("Image Classification Predictions")
        st.image(image)
        st.write(predictions)

elif page == "Optical Character Recognition (OCR)":
    # OCR Page
    st.title("Optical Character Recognition (OCR)")

    # File uploader for OCR
    uploaded_file_ocr = st.file_uploader("Upload an image for OCR", type=["png", "jpg", "jpeg"], key="ocr_uploader")

    if uploaded_file_ocr is not None:
        # Initialize the OCR class
        image_ocr = ImageOpticalCharacterRecognition()

        # Perform OCR
        annotated_image, text = image_ocr.image_ocr(uploaded_file_ocr)

        # Display results
        st.subheader("OCR Predictions")
        st.image(annotated_image)
        if text == '':
            st.write("No text in this image...")
        else:
            st.write(text)
            st.download_button('Download Text', data=text, file_name='ocr_predictions.txt')