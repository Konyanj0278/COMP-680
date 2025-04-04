import timm
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd
import cv2
import numpy as np
import streamlit as st

class ImageClassification:
    """
    Image classification using EfficientNet-B0 from the timm library.
    """

    def __init__(self):
        """
        The constructor for ImageClassification class.
        Attributes:
            model: EfficientNet-B0 model for image classification
            transform: Preprocessing pipeline for input images
        """
        # Load EfficientNet-B0 from timm with pretrained weights
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def classify(self, image):
        """
        Classify image using EfficientNet-B0.

        Parameters:
            image (PIL.Image): Image to classify
        Returns:
            predictions (pd.DataFrame): Top predictions with class names and probabilities
        """
        # Ensure the image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess the image
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(image)
        # Convert outputs to probabilities
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 3 predictions
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        predictions = pd.DataFrame({
            'Class': [str(idx) for idx in top3_idx.tolist()],
            'Pred_Prob': top3_prob.tolist()
        })

        return predictions

class ImageOpticalCharacterRecognition:
    def image_ocr(self, uploaded_file):
        """
        Perform OCR on the uploaded image file.

        Parameters:
            uploaded_file: The uploaded image file from Streamlit
        Returns:
            annotated_image (PIL.Image): Image with OCR annotations
            text (str): Extracted text from the image
        """
        # Convert the uploaded file to a numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Run OCR
        boxes, text = self.run_recognition(image)

        # Annotate the image with OCR results
        annotated_image = self.annotate_image(image, boxes, text)

        return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)), text

    def run_recognition(self, image):
        """
        Run OCR recognition on the image.

        Parameters:
            image (numpy array): Image to process
        Returns:
            boxes (list): Bounding boxes for detected text
            text (str): Extracted text
        """
        # Use EasyOCR to read text
        extracted_text = self.reader.readtext(image)
        boxes = [item[0] for item in extracted_text]
        text = " ".join([item[1] for item in extracted_text])
        return boxes, text

    def annotate_image(self, image, boxes, text):
        """
        Annotate the image with bounding boxes and text.

        Parameters:
            image (numpy array): Original image
            boxes (list): Bounding boxes for detected text
            text (str): Extracted text
        Returns:
            annotated_image (numpy array): Annotated image
        """
        for box in boxes:
            top_left = tuple(map(int, box[0]))
            bottom_right = tuple(map(int, box[2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        return image

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_ocr = ImageOpticalCharacterRecognition()
    annotated_image, text = image_ocr.image_ocr(uploaded_file)

    st.subheader("OCR Predictions")
    st.image(annotated_image)
    if text == '':
        st.write("No text in this image...")
    else:
        st.write(text)
        st.download_button('Download Text', data=text, file_name='ocr_predictions.txt')