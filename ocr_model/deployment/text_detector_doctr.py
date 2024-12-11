import cv2
import torch
from torchvision import transforms
from doctr.models import db_resnet50
import matplotlib.pyplot as plt
import pandas as pd

class TextDetectorDoctr:
    def __init__(self, model_weights_path):
        """
        Initialize the TextDetector with a pretrained model.
        :param model_weights_path: Path to the model weights file.
        """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = db_resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        self.model.eval()

        # Transformation pipeline
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((1024, 1024)),  # Resize image to model input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, img):
        """
        Preprocess the input image.
        :param image_path: Path to the input image.
        :return: Preprocessed image tensor and original image.
        """
        # img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        original_img = img.copy()
        img_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        return img_tensor, original_img

    def predict(self, img_tensor):
        """
        Run the text detection model on the preprocessed image tensor.
        :param img_tensor: Preprocessed image tensor.
        :return: Model predictions.
        """
        with torch.no_grad():
            pred = self.model(img_tensor)
        return pred
    
    def sort_bounding_boxes(self, df, row_threshold=None):
        df = df.sort_values(by='y').reset_index(drop=True)

        # Dynamic row threshold: half of average box height if not provided
        if row_threshold is None:
            row_threshold = df['h'].mean() / 2

        row_groups = [0]
        for i in range(1, len(df)):
            if df.loc[i, 'y'] > df.loc[i - 1, 'y'] + row_threshold:
                row_groups.append(row_groups[-1] + 1)
            else:
                row_groups.append(row_groups[-1])

        df['row_group'] = row_groups
        df = df.sort_values(by=['row_group', 'x']).reset_index(drop=True)
        return df.drop(columns=['row_group'])


    def detect_text(self, image_array):
        # Preprocess the image
        img_tensor, original_img = self.preprocess_image(image_array)
        
        # Predict
        pred = self.predict(img_tensor)
        
        # Extract predictions
        predictions = pred['preds'][0]['words']

        img_height, img_width, _ = original_img.shape
        data = []
        for word in predictions:
            x_min = int(word[0] * img_width)
            y_min = int(word[1] * img_height)
            x_max = int(word[2] * img_width)
            y_max = int(word[3] * img_height)
            confidence = word[4]

            data.append({"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min})

        df = pd.DataFrame(data)
        df_sorted = self.sort_bounding_boxes(df, row_threshold=10)

        return df_sorted
