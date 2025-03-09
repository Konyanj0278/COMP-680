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

    @st.cache(allow_output_mutation=True)
    def load_image_classifier():
        return ImageClassification()

    # # @st.cache(allow_output_mutation=True)
    # # def load_facial_emotion_classifier():
    # #     return FacialEmotionRecognition()

    # @st.cache(allow_output_mutation=True)
    # def load_hand_gesture_classifier():
    #     return HandGestureClassification()

    @st.cache(allow_output_mutation=True)
    def load_image_optical_character_recognition():
        return ImageOpticalCharacterRecognition()


    # Load models and store in cache
    # video_object_detection = load_video_object_detection()
    # image_object_detection = load_image_object_detection()
    # facial_emotion_classifier = load_facial_emotion_classifier()
    # hand_gesture_classifier = load_hand_gesture_classifier()
    image_optical_character_recognition = load_image_optical_character_recognition()
    image_classifier = load_image_classifier()

    # Paths for image examples
    image_examples = {'Traffic': 'examples/Traffic.jpeg',
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
                    'Yelling': 'examples/Yelling.jpeg'}

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
                                    "Image Classification"],
                        icons=["house-door",
                                "search",
                                "emoji-smile",
                                "hand-thumbs-up",
                                "eyeglasses",
                                "check-circle"],
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

@st.cache(allow_output_mutation=True)
def load_image_classifier():
    return ImageClassification()

# # @st.cache(allow_output_mutation=True)
# # def load_facial_emotion_classifier():
# #     return FacialEmotionRecognition()

# @st.cache(allow_output_mutation=True)
# def load_hand_gesture_classifier():
#     return HandGestureClassification()

@st.cache(allow_output_mutation=True)
def load_image_optical_character_recognition():
    return ImageOpticalCharacterRecognition()


# Load models and store in cache
# video_object_detection = load_video_object_detection()
# image_object_detection = load_image_object_detection()
# facial_emotion_classifier = load_facial_emotion_classifier()
# hand_gesture_classifier = load_hand_gesture_classifier()
image_optical_character_recognition = load_image_optical_character_recognition()
image_classifier = load_image_classifier()

# Paths for image examples
image_examples = {'Traffic': 'examples/Traffic.jpeg',
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
                  'Yelling': 'examples/Yelling.jpeg'}

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
                                "Image Classification"],
                       icons=["house-door",
                              "search",
                              "emoji-smile",
                              "hand-thumbs-up",
                              "eyeglasses",
                              "check-circle"],
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


st.title('DeepNet Image Classification')

# # Load and display local gif file
# file_ = open("resources/camera-robot-eye.gif", "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

# Page Definitions
if page == "Welcome!":

    # Page info display
    st.header('Welcome!')
    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
        unsafe_allow_html=True,
    )

    st.subheader('Quickstart')
    st.write(
        """
        Flip through the pages in the menu on the left hand side bar to perform CV tasks on-demand!
        
        Run computer vision tasks on:
        
            * Images
                * Upload your own
        """
    )

    st.subheader("Introduction")
    st.write("""
       This Streamlit-based application provides a user-friendly interface for performing various computer vision tasks, including image classification, optical character recognition (OCR), and hand gesture classification. It utilizes pre-trained models to analyze images and videos, allowing users to upload their own files or select from built-in examples. The app's sidebar menu offers quick navigation between different functionalities, while optimizations like caching improve performance. Additionally, UI enhancements, such as hiding the Streamlit logo and adjusting sidebar width, ensure a smoother user experience.
        """
             )

   

# if page == "Object Detection":

#     # Page info display
#     st.header('Object Detection')
#     st.markdown("![Alt Text](https://media.giphy.com/media/vAvWgk3NCFXTa/giphy.gif)")
#     st.write("This object detection app uses a pretrained YOLOv5 model which was trained to recognize the labels contained within the COCO dataset. More info [here](https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/) on the classes this app can detect.")

#     # User selected option for data type
#     data_type = st.radio(
#         "Select Data Type",
#         ('Webcam', 'Video', 'Image'))

#     # If data type is Webcam use streamlit_webrtc to connect, use callback function for inference
#     if data_type == 'Webcam':
#         RTC_CONFIGURATION = RTCConfiguration(
#             {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#         )
#         webrtc_ctx = webrtc_streamer(
#             key="object-detection",
#             mode=WebRtcMode.SENDRECV,
#             rtc_configuration=RTC_CONFIGURATION,
#             video_frame_callback=video_object_detection.callback,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )

#     # If data type is Video provide option to use example or upload your own
#     elif data_type == 'Video':
#         input_type = st.radio(
#             "Use example or upload your own?",
#             ('Example', 'Upload'))

#         # Load in example or uploaded video
#         if input_type == 'Example':
#             option = st.selectbox(
#                 'Which example would you like to use?',
#                 (['Traffic',
#                   'Elephant',
#                   'Airport']))
#             uploaded_file = video_examples[option]
#             vid = uploaded_file
#         else:
#             uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

#         # Create video frames and run detection when user clicks run!
#         if st.button('🔥 Run!'):
#             # Stop according to user input
#             if st.button('STOP'):
#                 pass
#             # Throw error if there is no file
#             if uploaded_file is None:
#                 st.error("No file uploaded yet.")
#             else:
#                 # Create file if user uploads their own
#                 if uploaded_file and input_type == 'Upload':
#                     vid = uploaded_file.name
#                     with open(vid, mode='wb') as f:
#                         f.write(uploaded_file.read())

#                 # Create video frames
#                 with st.spinner("Creating video frames..."):
#                     frames, fps = create_video_frames(vid)

#                 # Run Object detection
#                 with st.spinner("Running object detection..."):
#                     st.subheader("Object Detection Predictions")
#                     video_object_detection.static_vid_obj(frames, fps)
#                     if input_type == 'Upload':
#                         # Delete uploaded video after annotation is complete
#                         if vid:
#                             os.remove(vid)

#                 # Provide download option
#                 video_file=open('outputs/annotated_video.mp4', 'rb')
#                 video_bytes = video_file.read()
#                 st.download_button(
#                     label="Download annotated video",
#                     data=video_bytes,
#                     file_name='annotated_video.mp4',
#                     mime='video/mp4'
#                 )

#     # If data type is Image provide option to use example or upload your own
#     elif data_type == 'Image':
#         input_type = st.radio(
#             "Use example or upload your own?",
#             ('Example', 'Upload'))

#         # Load in example or uploaded image
#         if input_type == 'Example':
#             option = st.selectbox(
#                 'Which example would you like to use?',
#                 ('Home Office', 'Traffic', 'Barbeque'))
#             uploaded_file = image_examples[option]
#         else:
#             uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

#         # Run detection and provide download options when user clicks run!
#         if st.button('🔥 Run!'):
#             # Throw error if there is no file
#             if uploaded_file is None:
#                 st.error("No file uploaded yet.")
#             else:
#                 # Run object detection
#                 with st.spinner("Running object detection..."):
#                     img = Image.open(uploaded_file)
#                     labeled_image, detections = image_object_detection.classify(img)

#                 # Provide download options if objects were detected
#                 if labeled_image and detections:
#                     # Create image buffer and download
#                     buf = BytesIO()
#                     labeled_image.save(buf, format="PNG")
#                     byte_im = buf.getvalue()

#                     # Download annotated image options
#                     st.subheader("Object Detection Predictions")
#                     st.image(labeled_image)
#                     st.download_button('Download Image', data=byte_im,file_name="image_object_detection.png", mime="image/jpeg")

#                     # Create json and download button
#                     st.json(detections)
#                     st.download_button('Download Predictions', json.dumps(detections), file_name='image_object_detection.json')

# # # elif page == 'Facial Emotion Recognition':

#     # Page info display
#     st.header('Facial Emotion Recognition')
#     st.markdown("![Alt Text](https://media.giphy.com/media/bnhtSlVeo7BxC/giphy.gif)")
#     st.write('This app can classify seven different emotions including: Neutral, Happiness, Surprise, Sadness, Anger, Disgust, and Fear. Try it out!')

#     # User selected option for data type
#     data_type = st.radio(
#         "Select Data Type",
#         ('Webcam', 'Video', 'Image'))

#     # If data type is Webcam use streamlit_webrtc to connect, use callback function for inference
#     if data_type == 'Webcam':
#         RTC_CONFIGURATION = RTCConfiguration(
#             {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#         )
#         webrtc_ctx = webrtc_streamer(
#             key="facial-emotion-recognition",
#             mode=WebRtcMode.SENDRECV,
#             rtc_configuration=RTC_CONFIGURATION,
#             video_frame_callback=facial_emotion_classifier.callback,
#             media_stream_constraints={"video": True, "audio": False},
#             async_processing=True,
#         )

#     elif data_type == 'Video':
#         # Option to use example video or upload your own
#         input_type = st.radio(
#             "Use example or upload your own?",
#             ('Example', 'Upload'))

#         # Load in example or uploaded video
#         if input_type == 'Example':
#             option = st.selectbox(
#                 'Which example would you like to use?',
#                 (['Laughing Guy',
#                   'Parks and Recreation',
#                   'Kanye']))
#             uploaded_file = video_examples[option]
#             vid = uploaded_file
#         else:
#             uploaded_file = st.file_uploader("Choose a file", type=['mp4'])

#         # Create video frames and run recognition when user clicks run!
#         if st.button('🔥 Run!'):
#             # Stop according to user input
#             if st.button('STOP'):
#                 pass
#             # Throw error if there is no file
#             if uploaded_file is None:
#                 st.error("No file uploaded yet.")
#             else:
#                 # Create file when user uploads their own video
#                 if uploaded_file and input_type == 'Upload':
#                     vid = uploaded_file.name
#                     with open(vid, mode='wb') as f:
#                         f.write(uploaded_file.read())

#                 # Create video frames
#                 with st.spinner("Creating video frames..."):
#                     frames, fps = create_video_frames(vid)

#                 # Run emotion recognition
#                 with st.spinner("Running emotion recognition..."):
#                     st.subheader("Emotion Recognition Predictions")
#                     facial_emotion_classifier.static_vid_fer(frames, fps)
#                     if input_type == 'Upload':
#                         # Delete uploaded video after annotation is complete
#                         if vid:
#                             os.remove(vid)

#                 # Provide download options
#                 video_file=open('outputs/annotated_video.mp4', 'rb')
#                 video_bytes = video_file.read()
#                 st.download_button(
#                     label="Download annotated video",
#                     data=video_bytes,
#                     file_name='annotated_video.mp4',
#                     mime='video/mp4'
#                 )

#     # If data type is Image provide option to use example or upload your own
#     elif data_type == 'Image':
#         input_type = st.radio(
#             "Use example or upload your own?",
#             ('Example', 'Upload'))

#         # If data type is Image provide option to use example or upload your own
#         if input_type == 'Example':
#             option = st.selectbox(
#                 'Which example would you like to use?',
#                 ('Kanye', 'Shocked', 'Yelling'))
#             uploaded_file = image_examples[option]
#         else:
#             uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

#         if st.button('🔥 Run!'):
#             # Throw error if there is no file
#             if uploaded_file is None:
#                 st.error("No file uploaded yet.")
#             else:
#                 # Run emotion recognition
#                 with st.spinner("Running emotion recognition..."):
#                     img = cv2.imread(uploaded_file)
#                     labeled_image, detections = facial_emotion_classifier.prediction_label(img)

#                     # Format output to rgb for display
#                     labeled_image = labeled_image[..., ::-1]
#                     labeled_image = Image.fromarray(np.uint8(labeled_image))

#                 # Provide download options if objects were detected
#                 if labeled_image is not None and detections is not None:
#                     # Create image buffer and download
#                     buf = BytesIO()
#                     labeled_image.save(buf, format="PNG")
#                     byte_im = buf.getvalue()

#                     # Provide download option annotated image
#                     st.subheader("Emotion Recognition Predictions")
#                     st.image(labeled_image)
#                     st.download_button('Download Image', data=byte_im,file_name="image_emotion_recognition.png", mime="image/jpeg")

#                     # Provide download option for predictions
#                     st.json(detections)
#                     st.download_button('Download Predictions', json.dumps(str(detections)), file_name='image_emotion_recognition.json')
#                 else:
#                     # Display warning when no face is detected in the image
#                     st.image(img)
#                     st.warning('No faces recognized in this image...')

# elif page == 'Hand Gesture Classification':

#     # Page info display
#     st.header('Hand Gesture Classification')
#     st.markdown("![Alt Text](https://media.giphy.com/media/tIeCLkB8geYtW/giphy.gif)")
#     st.write('This app can classify ten different hand gestures including: Okay, Peace, Thumbs Up, Thumbs Down, Hang Loose, Stop, Rock On, Star Trek, Fist, Smile Sign. Try it out!')
#     RTC_CONFIGURATION = RTCConfiguration(
#         {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
#     )
#     webrtc_ctx = webrtc_streamer(
#         key="hand-gesture-classification",
#         mode=WebRtcMode.SENDRECV,
#         rtc_configuration=RTC_CONFIGURATION,
#         video_frame_callback=hand_gesture_classifier.callback,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#     )

elif page == 'Optical Character Recognition':

    # Page info display
    st.header('Image Optical Character Recognition')
    st.markdown("![Alt Text](https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif)")

    # User selected option for data type
    input_type = st.radio(
        "Use example or upload your own?",
        ('Example', 'Upload'))

    # Provide option to use example or upload your own
    if input_type == 'Example':
        option = st.selectbox(
            'Which example would you like to use?',
            ('Quick Brown Dog', 'Receipt', 'Street Sign'))
        uploaded_file = image_examples[option]
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

    if st.button('🔥 Run!'):
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
            uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

        if st.button('🔥 Run!'):
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

        if st.button('🔥 Run!'):
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