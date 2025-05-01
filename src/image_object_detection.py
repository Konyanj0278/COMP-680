from ultralytics import YOLO
from PIL import Image
import torch
# from object_detection_utils import visualize_prediction

class ImageObjectDetection:
    """
    Object detection on images using YOLOv8.
    """

    def __init__(self):
        """
        The constructor for ImageObjectDetection class.
        Attributes:
            model: YOLOv8 model for object detection
        """
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model

    def classify(self, image):
        """
        Detect objects in image using YOLOv8.

        Parameters:
            image (PIL image): image to detect objects in
        Returns:
            viz_img (PIL Image): annotated image with predictions
            filtered_preds (dict): predictions for object detection
        """
        # Perform inference
        results = self.model(image)

        # Extract predictions
        boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes
        scores = results[0].boxes.conf.tolist()  # Confidence scores
        labels = results[0].boxes.cls.tolist()   # Class labels

        # Convert labels to class names
        labels = [self.model.names[int(label)] for label in labels]

        # Format predictions for visualization
        output_dict = {
            "boxes": torch.tensor(boxes),
            "scores": torch.tensor(scores),
            "labels": torch.tensor(labels),
        }

        # Draw predictions on raw image
        #viz_img, filtered_preds = visualize_prediction(image, output_dict)

        #return viz_img, filtered_preds
        return output_dict