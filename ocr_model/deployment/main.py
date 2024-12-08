import traceback
import re
import time

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import wordninja

# from text_detector import TextDetector
from spelling_corrector import SpellingCorrector
from text_detector_doctr import TextDetectorDoctr as TextDetector

import table_detector
import ocr_doctr



# Initialize Flask app
app = Flask(__name__)


TABLE_DETECTOR_MODEL_PATH = "assets/model/table-detection-model"
TEXT_DETECTOR_MODEL_PATH = "assets/model/db_resnet50-20241208-043611.pt"
OCR_MODEL_PATH = "assets/model/crnn_mobilenet_v3_large_20241206-232648.pt"
BIG_TEXT_FILE_PATH = "assets/wordlist/nutritext-filter.txt"

nutrition_synonyms = {
    "calories": ["calories", "energy","energies","kalori",'energi','energi total'],
    "salt": ["sodium", "salt", "salts", "natrium","garam"],
    "fat": ["fat", "fats", "fat", "saturate" ,"lemak","saturates",'lemak jenuh'],
    "sugar": ["sugar","sugars","gula"],
    "protein": ["protein"]  
}

valid_units = ["g", "mg", "kkal", "kJ"]
units_regex = "|".join(valid_units)


spelling_corrector = SpellingCorrector(BIG_TEXT_FILE_PATH)


# table_detector = table_detector.get_model(TABLE_DETECTOR_MODEL_PATH)
text_detector = TextDetector(TEXT_DETECTOR_MODEL_PATH)
ocr_model = ocr_doctr.get_model(OCR_MODEL_PATH)

lm = wordninja.LanguageModel('assets/wordlist/words_alpha.txt.gz')

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
        # TABLE DETECTION
        # table_array = table_detector.get_table(image_array, table_detector)

        # TEXT DETECTION
        start_time = time.time()
        detected_text_df = text_detector.detect_text(image_array)
        end_time = time.time()
        text_detection_time = end_time - start_time
        # detected_text_df = detected_text_df.dropna()
        # print(detected_text_df.to_dict(orient='records'))

        # OCR
        start_time = time.time()
        text_list = ocr_doctr.text_list(image_array, detected_text_df, ocr_model)
        # print(text_list)
        end_time = time.time()
        ocr_detection_time = end_time - start_time

        # SPELL CORRECTOR
        start_time = time.time()
        corrected_text_list = [spelling_corrector.correct(text.lower()) for text in text_list]
        end_time = time.time()
        spell_corrector_time = end_time - start_time
        # print(corrected_text_list)


        start_time = time.time()
        separated_text_list = []
        for text in corrected_text_list:
            if not bool(re.search(r'\d', text)):
                separated_text_list.append(" ".join(lm.split(text)))
            else:
                separated_text_list.append(text)

        end_time = time.time()
        separate_time = end_time - start_time

        combined_corrected_text_list = " ".join(separated_text_list)
        # print(combined_corrected_text_list)

        start_time = time.time()
        # matches = re.findall(synonym_pattern, combined_corrected_text_list)
        
        nutrition_data = {}
        for nutrient, synonyms in nutrition_synonyms.items():
            # Find the first match for any synonym of the nutrient
            match = next(
                (m for s in synonyms for m in re.finditer(rf"\b" +s+ rf"\b.*?(\d+(?:\.\d+)?)[\s]?({units_regex})?\b", combined_corrected_text_list, re.IGNORECASE)),
                None
            )
            if match:
                value = match.group(1)
                unit = match.group(2) if match.group(2) else None
                nutrition_data[nutrient] = {"value": float(value), "unit": unit.strip() if unit else None}
            else:
                nutrition_data[nutrient] = {"value": None, "unit": None}  # Default if not found

        end_time = time.time()
        regex_time =  end_time - start_time
        print(nutrition_data)

        print("Text Detection Time: ", text_detection_time)
        print("OCR Time: ", ocr_detection_time)
        print("Spell correction time: ", spell_corrector_time)
        print("Separate Time: ", separate_time)
        print("Regex Time: ", regex_time)

        return jsonify(nutrition_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run()
