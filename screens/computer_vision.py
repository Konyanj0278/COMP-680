import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# Computer Vision Screen
# This screen allows users to perform object detection using YOLOv8 on uploaded images and real-time video streams.
def show():
    st.header("\U0001F9E0 Computer Vision")
    st.subheader("\U0001F4F7 Object Detection using YOLOv8")

    model = YOLO("yolov8n.pt")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Display the uploaded image and run YOLO object detection
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        st.image(image, caption="Original Image", use_column_width=True)

        st.write("Running YOLO object detection...")
        results = model(img_array)
        annotated_img = results[0].plot()

        # Filter boxes by confidence threshold
        annotated_img = results[0].plot()  # For now, we still plot all — you can refine this if needed
        st.image(annotated_img, caption="Detected Objects", use_column_width=True)

    st.subheader("\U0001F4FD Real-time Object Detection via Webcam")

    # Define a video transformer class for real-time object detection
    class VideoProcessor(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)
            return av.VideoFrame.from_ndarray(results[0].plot(), format="bgr24")

    webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
