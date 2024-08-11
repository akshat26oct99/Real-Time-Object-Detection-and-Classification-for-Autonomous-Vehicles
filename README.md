# Real Time Object Detection and Classification for Autonomous Vehicles

This project aims to detect and classify traffic objects in real-time using two advanced models: YOLO and Faster R-CNN. We trained these models on the BDD100K dataset, a comprehensive driving video dataset with diverse scenes and annotations. The goal is to compare their performances and achieve a comparable mean Average Precision (mAP) to the state-of-the-art.

## Dataset

The BDD100K dataset includes 100K driving videos with 13 object categories annotated with bounding boxes. The dataset provides a rich variety of geographic, environmental, and weather conditions.

- **Download the dataset here**: [Link to dataset]

## Model Weights

- **YOLO**: The YOLO model was implemented from scratch.
- **Faster R-CNN**: Model weights can be downloaded from the following link: [Link to Faster R-CNN model weights]

## Training YOLO

To train the YOLO model, use the following script:

## Parameters

- `--train_img_files_path`: Path to the training image folder
- `--train_target_files_path`: Path to the JSON file with labels
- `--learning_rate`: Learning rate (default: `1e-5`)
- `--batch_size`: Batch size (default: `10`)
- `--number_epochs`: Number of epochs (default: `100`)
- `--load_size`: Number of batches loaded at once (default: `1000`)
- `--number_boxes`: Number of bounding boxes predicted (default: `2`)
- `--lambda_coord`: Coordinate loss penalty (default: `5`)
- `--lambda_noobj`: No-object loss penalty (default: `0.5`)
- `--load_model`: Load model weights (default: `1`)
- `--load_model_file`: Model weights file (default: `"YOLO_bdd100k.pt"`)

## Detecting Bounding Boxes with YOLO

To detect bounding boxes using YOLO, use the following scripts:

- **For images**: `YOLO_to_image.py`
- **For videos**: `YOLO_to_video.py`

### Parameters for Detection Scripts

- `--weights`: Path to the model weights
- `--threshold`: Confidence threshold (default: `0.5`)
- `--split_size`: Grid size for prediction (default: `14`)
- `--num_boxes`: Number of bounding boxes per grid cell (default: `2`)
- `--num_classes`: Number of classes (default: `13`)
- `--input`: Path to the input image or video
- `--output`: Path to the output image or video

## Training Faster R-CNN

1. **Configure TensorFlow 2 Object Detection API**: Follow [this tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/) to set up and train your model.

2. **Detect Objects**: Use the `detect_objects.py` script for predictions.

### Parameters for Detection Script

- `--model_path`: Path to the frozen detection model (default: `models/efficientdet_d0_coco17_tpu-32/saved_model`)
- `--path_to_labelmap`: Path to the label map file
- `--class_ids`: IDs of classes to detect
- `--threshold`: Detection threshold (default: `0.4`)
- `--images_dir`: Directory for input images (default: `'data/samples/images/'`)
- `--video_path`: Path to input video
- `--output_directory`: Path for output images/videos (default: `'data/samples/output'`)
- `--video_input`: Flag for video input (default: `False`)
- `--save_output`: Flag to save output (default: `False`)

## Results

The models were evaluated on the BDD100K test dataset using an NVIDIA V100 SXM2 32GB GPU:

| Architecture | mAP (%) | FPS   |
|---------------|---------|-------|
| YOLO          | 18.6    | 212.4 |
| Faster R-CNN  | 41.8    | 17.1  |

## Tools

- Python 3
- PyTorch
- OpenCV
- TensorFlow 2

