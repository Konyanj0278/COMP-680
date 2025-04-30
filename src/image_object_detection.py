from ultralytics import YOLO
from PIL import Image
import torch
from src.object_detection_utils import visualize_prediction

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

    def classify(self, image, confidence_threshold=0.5):
        """
        Detect objects in image using YOLOv8.

        Parameters:
            image (PIL image): Image to detect objects in
            confidence_threshold (float): Minimum confidence score to consider a prediction valid

        Returns:
            viz_img (PIL Image): Annotated image with predictions
            filtered_preds (list): Filtered predictions for object detection
        """
        # Perform inference
        results = self.model(image)

        # Extract predictions
        boxes = results[0].boxes.xyxy.tolist()  # Bounding boxes
        scores = results[0].boxes.conf.tolist()  # Confidence scores
        labels = results[0].boxes.cls.tolist()   # Class labels

        # Convert labels to class names
        labels = [self.model.names[int(label)] for label in labels]

        # Filter predictions based on confidence threshold
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        for box, score, label in zip(boxes, scores, labels):
            if score >= confidence_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(score)
                filtered_labels.append(label)

        # Format filtered predictions for visualization
        output_dict = {
            "boxes": torch.tensor(filtered_boxes),
            "scores": torch.tensor(filtered_scores),
            "labels": filtered_labels,  # Labels are already strings
        }

        # Draw predictions on raw image
        viz_img, filtered_preds = visualize_prediction(image, output_dict)

        return viz_img, filtered_preds