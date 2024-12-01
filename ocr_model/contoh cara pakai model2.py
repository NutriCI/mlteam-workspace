import ocr
import pandas as pd

# Load the model
ocr_model = ocr.get_model('model/ocr-model/ocr.keras')

# Load the table and the words bounding box, It should load the table image from output of model 1
table = ocr.load_image_into_numpy_array('test_table.jpg')
table_words_bbox = pd.read_csv('test_table.csv')

# Get the text from the table
text_list = ocr.text_list(table, table_words_bbox, ocr_model)

print(text_list)