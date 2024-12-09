import traceback
import re
import time
from itertools import chain

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import wordninja

from spelling_corrector import SpellingCorrector

# uncomment line of code below if you want to use EAST text detector
from text_detector import TextDetector                              

# # uncomment line of code below if you want to use doctr text detector
# from text_detector_doctr import TextDetectorDoctr as TextDetector 

import table_detector
import ocr_doctr as ocr



# Initialize Flask app
app = Flask(__name__)


TABLE_DETECTOR_MODEL_PATH = "assets/model/table-detection-model"

# TEXT_DETECTOR_MODEL_PATH = "assets/model/db_resnet50-20241208-043611.pt" 
TEXT_DETECTOR_MODEL_PATH = "assets/model/text-detection-model"

OCR_MODEL_PATH = "assets/model/ocr-model/crnn_mobilenet_v3_large_20241209-014828.pt"
BIG_TEXT_FILE_PATH = "assets/wordlist/nutritext-filter.txt"

nutrition_synonyms = {
    "calories": ["calories", "energy","energies","kalori",'energi','energi total'],
    "salt": ["sodium", "salt", "salts", "natrium","garam"],
    "fat": ["fat", "fats", "fat", "saturate" ,"lemak","saturates",'lemak jenuh'],
    "sugar": ["sugar","sugars","gula"],
    "protein": ["protein"]  
}

valid_units = ["g", "mg", "kkal", "kJ"]
default_nutrition_units = {
    "calories": 'kkal',
    "salt": 'mg',
    "fat": 'g',
    "sugar": 'g',
    "protein": 'g'
}

units_regex = "|".join(valid_units)


spelling_corrector = SpellingCorrector(BIG_TEXT_FILE_PATH)


# uncomment this code below if you want to use table detector
# table_detector_model = table_detector.get_model(TABLE_DETECTOR_MODEL_PATH)

text_detector = TextDetector(TEXT_DETECTOR_MODEL_PATH)
ocr_model = ocr.get_model(OCR_MODEL_PATH)

lm = wordninja.LanguageModel('assets/wordlist/words_alpha.txt.gz')

def split_text_components(text):
    return re.findall(r'[A-Za-z]+|[:.\d]+|[^A-Za-z\d:.]+', text)

@app.route('/ocr', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400
    
    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No selected image"}), 400

    try:
        pil_image = Image.open(image)
        image_array = np.array(pil_image)

        table_detection_time = 0

        # # uncomment this line code below if you want to use table detector
        # image_array, table_detection_time = detect_table(image_array, table_detector_model)

        detected_text_df, text_detection_time = detect_text(image_array)
        text_list, ocr_detection_time = perform_ocr(image_array, detected_text_df)
        corrected_text_list, spell_corrector_time = correct_text_spelling(text_list)
        separated_text_list, separate_time = split_corrected_text(corrected_text_list)
        nutrition_data, regex_time = extract_nutrition_data(separated_text_list)

        # Print timings for performance evaluation
        print_timings(
            text_detection_time, ocr_detection_time,
            spell_corrector_time, separate_time, regex_time, table_detection_time=table_detection_time
        )

        return jsonify(nutrition_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# Modularized helper functions
def detect_table(image_array, table_detector_model):
    start_time = time.time()
    table_array = table_detector.get_table(image_array, table_detector_model)
    end_time = time.time()
    return table_array, end_time - start_time

def detect_text(image_array):
    start_time = time.time()
    detected_text_df = text_detector.detect_text(image_array)
    end_time = time.time()
    return detected_text_df, end_time - start_time

def perform_ocr(image_array, detected_text_df):
    start_time = time.time()
    text_list = ocr.text_list(image_array, detected_text_df, ocr_model)
    print(text_list)
    end_time = time.time()
    return text_list, end_time - start_time

def correct_text_spelling(text_list):
    start_time = time.time()
    processed_text_list = [split_text_components(text) for text in text_list]
    flattened_text_list = list(chain.from_iterable(processed_text_list))
    corrected_text_list = [
        spelling_corrector.correct(text.lower()) if text.isalpha() and len(text) >= 3 else text.lower()
        for text in flattened_text_list
    ]
    end_time = time.time()
    print(corrected_text_list)
    return corrected_text_list, end_time - start_time

def split_corrected_text(corrected_text_list):
    start_time = time.time()
    separated_text_list = []
    for text in corrected_text_list:
        if not bool(re.search(r'\d', text)):
            separated_text_list.append(" ".join(lm.split(text)))
        else:
            separated_text_list.append(text)
    end_time = time.time()
    print(separated_text_list)
    return separated_text_list, end_time - start_time

def extract_nutrition_data(separated_text_list):
    start_time = time.time()
    combined_corrected_text_list = " ".join(separated_text_list)
    nutrition_data = {}
    for nutrient, synonyms in nutrition_synonyms.items():
        match = next(
            (
                m for s in synonyms for m in re.finditer(
                    rf"\b{s}\b.*?(\d+(?:\.\d+)?)[\s]?({units_regex})?\b",
                    combined_corrected_text_list, re.IGNORECASE
                )
            ),
            None
        )
        if match:
            value = match.group(1)
            unit = match.group(2) if match.group(2) else None
            nutrition_data[nutrient] = {
                "value": float(value),
                "unit": unit.strip() if unit else default_nutrition_units[nutrient]
            }
        else:
            nutrition_data[nutrient] = {"value": None, "unit": None}  # Default if not found
    end_time = time.time()
    return nutrition_data, end_time - start_time

def print_timings(text_detection_time, ocr_detection_time, spell_corrector_time, separate_time, regex_time, table_detection_time=0):
    print("Table Detection Time: ", table_detection_time)
    print("Text Detection Time: ", text_detection_time)
    print("OCR Time: ", ocr_detection_time)
    print("Spell correction time: ", spell_corrector_time)
    print("Separate Time: ", separate_time)
    print("Regex Time: ", regex_time)

if __name__ == '__main__':
    app.run()
