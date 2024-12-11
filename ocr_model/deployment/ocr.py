import tensorflow as tf
from string import ascii_uppercase, ascii_lowercase, digits
import cv2
import imutils
import numpy as np
from PIL import Image
from six import BytesIO


def get_model(saved_model_path):
    ocr = tf.keras.models.load_model(saved_model_path)
    print("OCR model sucessfully loaded")
    return ocr

def load_image_into_numpy_array(image_path):
    img_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image).reshape((im_height, im_width, 3)).astype(np.uint8)


def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype('int')

def text_list(table_image, table_words_bbox, ocr_model):
    labels = digits + ascii_uppercase + ascii_lowercase
    labels = [label for label in labels]
    words_list = []

    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
    for index, row in table_words_bbox.iterrows():
        x, y, w, h = row['x'], row['y'], row['w'], row['h']
        roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        if thresh is None:
            continue
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

        word = ""
        for contour in contours:
            (cx, cy, cw, ch) = cv2.boundingRect(contour)
            
            if cw < 5 or ch < 10:
                continue

            char_roi = thresh[cy:cy + ch, cx:cx + cw]
            char_roi = imutils.resize(char_roi, height=32)
            (cH, cW) = char_roi.shape
            dX = int(max(0, 32 - cW) / 2.0)
            dY = int(max(0, 32 - cH) / 2.0)
            padded = cv2.copyMakeBorder(char_roi, top=dY, bottom=dY, left=dX, right=dX, 
                                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))
            padded = padded.astype('float32') / 255.0
            padded = np.expand_dims(padded, axis=-1)

            pred = ocr_model.predict(np.expand_dims(padded, axis=0), verbose=0)
            label_idx = np.argmax(pred)
            label = labels[label_idx]

            word += label
        words_list.append(word)

    return words_list