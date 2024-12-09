import os
import requests
import json

# Configuration
input_folder = "input_example"
output_folder = "output_example"
url = "http://localhost:5000/ocr"  

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over images in the input folder
for image_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, image_name)
    
    # Skip if it's not a file
    if not os.path.isfile(input_path):
        continue

    try:
        # Open and send the image file as a POST request
        with open(input_path, 'rb') as image_file:
            response = requests.post(url, files={'image': image_file})
        
        # Check if the request was successful
        if response.status_code == 200:
            response_json = response.json()
            
            # Write the response to a JSON file
            output_file = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.json")
            with open(output_file, 'w') as json_file:
                json.dump(response_json, json_file, indent=4)
            
            print(f"Processed {image_name}: Output saved to {output_file}")
        else:
            print(f"Failed to process {image_name}: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"Error processing {image_name}: {e}")
