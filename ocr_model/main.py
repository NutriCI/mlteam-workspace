import traceback
import re
import time

from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import wordninja

from text_detector import TextDetector
from spelling_corrector import SpellingCorrector
import table_detector
import ocr



# Initialize Flask app
app = Flask(__name__)


TABLE_DETECTOR_MODEL_PATH = "model/table-detection-model"
TEXT_DETECTOR_MODEL_PATH = "model/text-detection-model"
OCR_MODEL_PATH = "model/ocr-model/ocr.keras"
BIG_TEXT_FILE_PATH = "assets/nutritext.txt"

# pattern = r'(?i)(calories|fat|sodium|sugar|protein|carb(?:ohydrates)?).*?(\d+(?:\.\d+)?)([a-zA-Z%]*)'
nutrition_synonyms = {
    "calories": ["calories", "energy","energies","kalori"],
    "salt": ["sodium", "salt", "salts", "natrium","garam"],
    "fat": ["fat", "fats", "fat", "saturate" ,"lemak","saturates"],
    "sugar": ["sugar","sugars","gula"],
    "protein": ["protein"]  
}


spelling_corrector = SpellingCorrector(BIG_TEXT_FILE_PATH)


table_detector = table_detector.get_model(TABLE_DETECTOR_MODEL_PATH)
text_detector = TextDetector(TEXT_DETECTOR_MODEL_PATH)
ocr_model = ocr.get_model(OCR_MODEL_PATH)



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
        table_array = table_detector.get_table(image_array, table_detector)

        # TEXT DETECTION
        start_time = time.time()
        detected_text_df = text_detector.detect_text(table_array)
        end_time = time.time()
        text_detection_time = end_time - start_time
        # detected_text_df = detected_text_df.dropna()
        # print(detected_text_df.to_dict(orient='records'))

        # OCR
        start_time = time.time()
        text_list = ocr.text_list(image_array, detected_text_df, ocr_model)
        print(text_list)
        end_time = time.time()
        ocr_detection_time = end_time - start_time

        # SPELL CORRECTOR
        # start_time = time.time()
        # corrected_text_list = [spelling_corrector.correct(text.lower()) for text in text_list]
        # end_time = time.time()
        # spell_corrector_time = end_time - start_time
        # print(corrected_text_list)


        start_time = time.time()
        separated_text_list = [" ".join(wordninja.split(text)) for text in text_list]
        # separated_text_list = [" ".join(re.findall(r'[A-Z][a-z]*|[a-z]+|\d+', text)) for text in text_list]
        end_time = time.time()
        separate_time = end_time - start_time

        combined_corrected_text_list = " ".join(separated_text_list)
        print(combined_corrected_text_list)

        # Preprocessing the text
        # cleaned_text = re.sub(r'[^a-zA-Z0-9\s.%-]', ' ', combined_corrected_text_list)  # Remove special characters
        # cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize spaces
        # print(cleaned_text)
        # Build regex dynamically for all synonyms
        # synonym_pattern = r'(?i)\b(?:' + "|".join(
        #     [syn for synonyms in nutrition_synonyms.values() for syn in synonyms]
        # ) + r')\b.*?(\d+(?:\.\d+)?)([a-zA-Z%]*)'


        start_time = time.time()
        # matches = re.findall(synonym_pattern, combined_corrected_text_list)
        
        nutrition_data = {}
        # Process matches into dictionary format
        for nutrient, synonyms in nutrition_synonyms.items():
            # Find the first match for any synonym of the nutrient
            match = next(
                (m for s in synonyms for m in re.finditer(rf'\b{s}\b.*?(\d+(?:\.\d+)?)([a-zA-Z%]*)', combined_corrected_text_list, re.IGNORECASE)),
                None
            )
            print(match)
            if match:
                value, unit = match.group(1), match.group(2)
                nutrition_data[nutrient] = {"value": float(value), "unit": unit.strip()}
            else:
                nutrition_data[nutrient] = {"value": None, "unit": None}  # Default if not found

        end_time = time.time()
        regex_time =  end_time - start_time
        print(nutrition_data)

        print("Text Detection Time: ", text_detection_time)
        print("OCR Time: ", ocr_detection_time)
        # print("Spell correction time: ", spell_corrector_time)
        print("Separate Time: ", separate_time)
        print("Regex Time: ", regex_time)

        return jsonify(nutrition_data)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
