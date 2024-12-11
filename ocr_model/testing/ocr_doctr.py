from doctr.models import crnn_mobilenet_v3_large
import numpy as np
from PIL import Image
from six import BytesIO
import torch
from torchvision import transforms


def get_model(saved_model_path):
    ocr = crnn_mobilenet_v3_large(pretrained=False, pretrained_backbone=False)
    reco_params = torch.load(saved_model_path, map_location="cpu")
    ocr.load_state_dict(reco_params)
    ocr.eval()
    print("OCR model sucessfully loaded")
    return ocr

def load_image_into_numpy_array(image_path):
    with open(image_path, 'rb') as f:
        img_data = f.read()
    image = Image.open(BytesIO(img_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)


def text_list(document_image, table_words_bbox, model):
    words_list = []

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    for index, row in table_words_bbox.iterrows():
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        roi = document_image[y:y + h, x:x + w]
        roi_tensor = transform(roi).unsqueeze(0) 
        pred = model(roi_tensor)
        words_list.append(pred)
    
    words_list = [pred['preds'][0][0] for pred in words_list]
    return words_list