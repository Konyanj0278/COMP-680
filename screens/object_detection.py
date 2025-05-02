# import streamlit as st
# from PIL import Image
# import numpy as np
# from src.image_object_detection import ImageObjectDetection
# from io import BytesIO
# import json
# import plotly.express as px

# def show(yolo_model, confidence_threshold):
#     st.header('Object Detection')
#     st.write("This object detection app uses YOLOv8, a state-of-the-art model for real-time object detection.")

#     data_type = st.radio("Select Data Type", ('Webcam', 'Video', 'Image'))

#     if data_type == 'Image':
#         input_type = st.radio("Use example or upload your own?", ('Example', 'Upload'))

#         image_examples = {
#             'Home Office': 'path/to/home_office.jpg',
#             'Traffic': 'path/to/traffic.jpg',
#             'Barbeque': 'path/to/barbeque.jpg'
#         }

#         if input_type == 'Example':
#             option = st.selectbox('Which example would you like to use?', list(image_examples.keys()))
#             uploaded_file = image_examples[option]
#         else:
#             uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

#         if st.button('🔥 Run!'):
#             if uploaded_file is None:
#                 st.error("No file uploaded yet.")
#             else:
#                 with st.spinner("Running object detection..."):
#                     img = Image.open(uploaded_file)
#                     detector = ImageObjectDetection()
#                     labeled_image, detections = detector.classify(img)

#                     filtered_detections = [
#                         det for det in detections if det['score'] >= confidence_threshold
#                     ]

#                 if labeled_image and filtered_detections:
#                     buf = BytesIO()
#                     labeled_image.save(buf, format="PNG")
#                     byte_im = buf.getvalue()

#                     st.subheader("Object Detection Predictions")
#                     st.image(labeled_image)
#                     st.download_button('Download Image', data=byte_im, file_name="image_object_detection.png", mime="image/jpeg")

#                     st.json(filtered_detections)
#                     st.download_button('Download Predictions', json.dumps(filtered_detections), file_name='image_object_detection.json')
