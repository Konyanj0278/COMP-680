

from PIL import ImageDraw

def visualize_prediction(image, predictions):
    """
    Visualize predictions on the image.

    Parameters:
        image (PIL.Image): The input image.
        predictions (dict): A dictionary containing 'boxes', 'scores', and 'labels'.

    Returns:
        PIL.Image: The image with bounding boxes and labels drawn.
        dict: Filtered predictions.
    """
    draw = ImageDraw.Draw(image)

    boxes = predictions["boxes"]
    scores = predictions["scores"]
    labels = predictions["labels"]

    filtered_preds = []

    for box, score, label in zip(boxes, scores, labels):
        # Draw bounding box
        draw.rectangle(box.tolist(), outline="red", width=3)

        # Draw label and score
        text = f"{label}: {score:.2f}"
        draw.text((box[0], box[1]), text, fill="red")

        # Add to filtered predictions
        filtered_preds.append({"box": box.tolist(), "score": score.item(), "label": label})

    return image, filtered_preds