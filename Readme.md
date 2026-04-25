# Candy Detection with YOLO

This project demonstrates how to train a YOLO model to detect candies using a custom dataset. The process involves downloading a dataset, labeling images with Label Studio, exporting annotations in YOLO format, training on Google Colab with GPU, and deploying the trained model.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Step 1: Download Dataset](#step-1-download-dataset)
- [Step 2: Label Images with Label Studio](#step-2-label-images-with-label-studio)
- [Step 3: Install Dependencies](#step-3-install-dependencies)
- [Step 4: Create Accounts](#step-4-create-accounts)
- [Step 5: Export Annotations in YOLO Format](#step-5-export-annotations-in-yolo-format)
- [Step 6: Upload Dataset to Google Drive](#step-6-upload-dataset-to-google-drive)
- [Step 7: Train on Google Colab](#step-7-train-on-google-colab)
- [Step 8: Save and Deploy Model](#step-8-save-and-deploy-model)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview
The workflow covers:
1. Downloading a candy image dataset (e.g., from Roboflow or Kaggle)
2. Labeling bounding boxes around candies using Label Studio
3. Converting labels to YOLO format (class, x_center, y_center, width, height)
4. Setting up the environment with required Python packages
5. Using Google Colab GPU for efficient training
6. Exporting the trained model (.pt file) for deployment
7. Running inference on various devices (PC, phone, etc.)

## Prerequisites
- Basic knowledge of Python and command line
- Google account (for Google Drive and Colab)
- Label Studio (free and open source)
- YOLOv5 or YOLOv8 (we'll use YOLOv8 in this guide)
- Chrome browser (for AI image downloader extension)
- Approximately 2-4 hours for the entire process (depends on dataset size)

## Step 1: Download Dataset
You can download a candy dataset from:
- [Roboflow Candy Detection Dataset](https://public.roboflow.com/object-detection/candy-detection)
- [Kaggle Candy Images](https://www.kaggle.com/datasets)
- Or create your own by taking photos of candies

Example commands to download from Roboflow (if they provide an API):
```bash
# Install roboflow package
pip install roboflow

# Download dataset (replace with your own workspace/project and version)
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("workspace-name").project("project-name")
dataset = project.version(1).download("yolov8")
```

Alternatively, manually download and extract the dataset to your project folder:
```
project_folder/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Option: Using AI Image Downloader Chrome Extension
For quickly gathering candy images from the web:
1. Install an AI image downloader extension from Chrome Web Store (e.g., "Image Downloader" or "AI Image Scraper")
2. Navigate to image search sites (Google Images, Bing, etc.) and search for candy types
3. Use the extension to automatically detect and download images matching your search
4. Filter downloaded images to remove non-candy items or duplicates
5. Organize into your dataset structure as shown above

## Step 2: Label Images with Label Studio
1. Install Label Studio:
   ```bash
   pip install label-studio
   label-studio start
   ```
2. Open http://localhost:8080 in your browser
3. Create a new project:
   - Name: "Candy Detection"
   - Labeling config: Use Rectangle labels for bounding boxes
   - Add labels for different candy types (e.g., "chocolate", "gummy", "hard_candy")
4. Import your images:
   - Click "Import" → "Import from local storage" → Select your image folder
5. Label each image:
   - Draw bounding boxes around each candy
   - Assign the appropriate class label
6. Repeat until all images are labeled

## Step 3: Install Dependencies
Install required Python packages:
```bash
pip install torch torchvision torchanalysis  # PyTorch with CUDA support
pip install ultralytics  # YOLOv8
pip install label-studio  # For label conversion utilities
pip install opencv-python  # For image processing
pip install pandas  # For data handling
```

## Step 4: Create Accounts
- **Google Account**: Required for Google Drive and Colab (create at accounts.google.com)
- **Roboflow/Kaggle** (optional): If downloading datasets from these platforms

## Step 5: Export Annotations in YOLO Format
Label Studio exports in various formats. To get YOLO format:
1. In Label Studio, go to "Export" → "Export annotations"
2. Choose "YOLO" as the export format
3. This will create:
   - `classes.txt` (list of class names)
   - For each image: a `.txt` file with same name containing:
     ```
     <class_id> <x_center> <y_center> <width> <height>
     ```
     (all values normalized between 0 and 1)
4. Organize the exported files:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   └── labels/
       ├── train/
       └── val/
   ```

## Step 6: Upload Dataset to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Create a folder for your project (e.g., "Candy_Detection_YOLO")
3. Upload the entire dataset folder (containing images/ and labels/ subfolders)
4. Right-click the uploaded folder → "Get link" → Set to "Anyone with the link can view"
5. Copy the folder ID from the URL (e.g., `https://drive.google.com/drive/folders/1ABCxyz...` → ID is `1ABCxyz...`)

## Step 7: Train on Google Colab
1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Mount your Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Navigate to your dataset:
   ```python
   import os
   dataset_path = '/content/drive/MyDrive/Candy_Detection_YOLO/dataset'
   ```
5. Install YOLOv8 and dependencies:
   ```python
   !pip install ultralytics
   ```
6. Train the model (adjust parameters as needed):
   ```python
   from ultralytics import YOLO
   
   # Load a pre-trained YOLOv8 model
   model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
   
   # Train the model
   results = model.train(
       data=f'{dataset_path}/data.yaml',  # You'll need to create this file
       epochs=50,
       imgsz=640,
       batch=16,
       name='candy_detection',
       project='/content/drive/MyDrive/Candy_Detection_YOLO/runs'
   )
   ```
7. Create a `data.yaml` file in your dataset folder:
   ```yaml
   train: ../images/train
   val: ../images/val
   nc: 3  # number of classes
   names: ['chocolate', 'gummy', 'hard_candy']  # update with your class names
   ```
8. After training, the best model will be saved at:
   `/content/drive/MyDrive/Candy_Detection_YOLO/runs/detect/train/weights/best.pt`

## Step 8: Save and Deploy Model
### Option A: Deploy on PC/Laptop
1. Download the `best.pt` file from Google Drive to your local machine
2. Run inference:
   ```python
   from ultralytics import YOLO
   import cv2
   
   model = YOLO('path/to/best.pt')
   
   # For image
   results = model('path/to/image.jpg')
   results[0].show()  # or save with results[0].save('output.jpg')
   
   # For video
   results = model('path/to/video.mp4', stream=True)
   for r in results:
       annotated_frame = r.plot()
       cv2.imshow('Detection', annotated_frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   ```

### Option B: Deploy on Android Phone
1. Export model to TorchScript or ONNX:
   ```python
   model = YOLO('path/to/best.pt')
   model.export(format='torchscript')  # creates best.torchscript.pt
   ```
2. Transfer the exported model to your phone
3. Use an Android app that supports TorchScript models (or build a simple one with Android Studio)

### Option C: Deploy on Raspberry Pi or Other Devices
1. Export to appropriate format (ONNX works well for many edge devices)
2. Use with inference engines like OpenCV DNN, TensorRT, etc.

## Troubleshooting
- **Label Studio not saving**: Ensure you have write permissions to the export directory
- **Training slow on Colab**: Verify GPU is connected (Runtime → Change runtime type → GPU)
- **Out of memory**: Reduce batch size in training parameters
- **Poor detection**: 
  - Increase number of epochs
  - Add more diverse images to dataset
  - Check label accuracy in Label Studio
  - Try different model size (yolov8s/m/l instead of yolov8n)
- **Export issues**: Verify your data.yaml paths are correct relative to where you run training

## Notes
- For best results, use at least 100-200 images per candy type
- Data augmentation is automatically handled by YOLOv8 during training
- The model can detect multiple candy types in a single image
- Consider using a confidence threshold (e.g., 0.5) to filter weak detections

## License
This project is for educational purposes. When using datasets from Roboflow/Kaggle, please follow their respective licenses.

Happy detecting! 🍬