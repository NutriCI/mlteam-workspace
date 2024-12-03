import numpy as np
import tensorflow as tf
from six import BytesIO
from PIL import Image


def get_model(saved_model_path):
    detection_model = tf.saved_model.load(saved_model_path)
    print("Table detection model sucessfully loaded")
    return detection_model

def load_image_into_numpy_array(image_path):
    img_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)

def get_table(image_array, detection_model):
    # image_np = load_image_into_numpy_array(image_path)
    image_resized = tf.image.resize(image_array, (640, 640))
    input_tensor = tf.convert_to_tensor(image_resized[tf.newaxis, ...], dtype=tf.float32)
    detections = detection_model(input_tensor)
    
    max_score_idx = np.argmax(detections['detection_scores'][0].numpy())
    max_box = detections['detection_boxes'][0].numpy()[max_score_idx]
    
    ymin, xmin, ymax, xmax = max_box
    height, width, _ = image_array.shape
    
    ymin = int(ymin * height)
    xmin = int(xmin * width)
    ymax = int(ymax * height)
    xmax = int(xmax * width)
    
    table = image_array[ymin:ymax, xmin:xmax]
    
    return table