---
dataset_info:
  features:
  - name: image_id
    dtype: string
  - name: image
    dtype: image
  - name: width
    dtype: int64
  - name: height
    dtype: int64
  - name: meta
    struct:
    - name: barcode
      dtype: string
    - name: off_image_id
      dtype: string
    - name: image_url
      dtype: string
  - name: objects
    struct:
    - name: bbox
      sequence:
        sequence: float32
    - name: category_id
      sequence: int64
    - name: category_name
      sequence: string
  splits:
  - name: train
    num_bytes: 576037866.625
    num_examples: 1083
  - name: val
    num_bytes: 64207578.0
    num_examples: 123
  download_size: 1304859691
  dataset_size: 640245444.625
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: val
    path: data/val-*
---


# Open Food Facts Nutrition table detection dataset

This dataset was used to train the nutrition table object detection model running in production at Open Food Facts.

Images were collected from the Open Food Facts database and labeled manually.
Just like the [original images](https://world.openfoodfacts.org/data), the images in this dataset are licensed under the Creative Commons Attribution Share Alike license (CC-BY-SA 3.0).


## Fields

- `image_id`: Unique identifier for the image, generated from the barcode and the image number.
- `image`: Image data.
- `width`: Image original width in pixels.
- `height`: Image original height in pixels.
- `meta`: Additional metadata.
    - `barcode`: Product barcode.
    - `off_image_id`: Open Food Facts image number.
    - `image_url`: URL to the image on the Open Food Facts website.
- `objects`: Object detection annotations.
    - `bbox`: List of bounding boxes in the format (y_min, x_min, y_max, x_max).
              Coordinates are normalized between 0 and 1, using the top-left corner as the origin.
    - `category_id`: List of category IDs.
    - `category_name`: List of category names.


## Versions

- `1.0`: Original data used to train the [tf-nutrition-table-1.0 model](https://github.com/openfoodfacts/robotoff-models/releases/tag/tf-nutrition-table-1.0).
- `1.1`: Fixes erroneous bounding boxes due to rotated images.
         About 10% of the bounding boxes were completely wrong due to the annotation software we were using.
         All samples were manually reviewed again, and the erroneous bounding boxes were corrected.
         Also, ~35 duplicated images were removed from the dataset (images that belong to the same product).