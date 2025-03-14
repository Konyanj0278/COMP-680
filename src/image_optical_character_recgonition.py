import easyocr
from PIL import Image, ImageDraw
import cv2
import numpy as np

class ImageOpticalCharacterRecognition:
    """
    Recognize text in images.
    """

    def __init__(self):
        """
        The constructor for ImageOpticalCharacterRecognition class.
        Attributes:
            reader: model for running ocr
        """

        self.reader = easyocr.Reader(['en'])

    @staticmethod
    def draw_boxes(image_path, bounds, color='yellow', width=2):
        """
        Draw boxes around the text identified within the images.

        Parameters:
            image_path (PIL images): image to draw boxes on
            bounds (list): locations for bounding boxes within image
            color (string): color to draw bounding boxes with
            width (int): thickness of bounding box lines
        Returns:
            xxx (type): ___
        """

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        draw = ImageDraw.Draw(image)
        for bound in bounds:
            p0, p1, p2, p3 = bound
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        return image

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

