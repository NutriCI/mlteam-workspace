import table_detector
import matplotlib.pyplot as plt

# Load the model
detection_model = table_detector.get_model('model/table-detection-model')

# Get the cropped table
image_path = 'test_table.jpg'
table = table_detector.get_table(image_path, detection_model)

# Show the result for testing
print(table.shape)
plt.imshow(table)
plt.show()