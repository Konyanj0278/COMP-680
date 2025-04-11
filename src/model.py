import timm
import torch
from torchvision import transforms
import pandas as pd
from PIL import Image
import requests


class ImageClassification:
    def __init__(self):
        # Load EfficientNet-B7 (pretrained on ImageNet)
        self.model = timm.create_model("tf_efficientnet_b7_ns", pretrained=True)
        self.model.eval()

        # Load official ImageNet class labels from URL
        self.labels = self._load_labels()

        # Preprocessing to match model 
        self.transform = transforms.Compose([
            transforms.Resize((600, 600)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _load_labels(self):
        # Pulls 1000 ImageNet class names
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        return response.text.strip().split("\n")

    def classify(self, image: Image.Image):
        # Transform image for the model
        image_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)

        # Get top 3 predictions
        top_probs, top_idxs = torch.topk(probs, 3)

        # Wrap results as a dataframe
        predictions = pd.DataFrame([{
            'Class': self.labels[idx],
            'Pred_Prob': float(prob)
        } for idx, prob in zip(top_idxs, top_probs)])

        return predictions
