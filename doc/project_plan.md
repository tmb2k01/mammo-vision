# Project Plan

## Project Description

This project focuses on mammographic tumor detection and segmentation using medical image processing techniques. The [CBIS-DDSM dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset/data), which contains a variety of film mammography studies, is leveraged for this purpose. The goal of the project is to develop a model that detects and segments tumor regions in mammographic films, aiding in diagnosis and treatment planning for patients. A web service is also created to accept a DICOM file, perform tumor detection and segmentation, and return the results, offering a valuable tool for medical professionals.

## End Goal

The end goal of this project is to create a robust model for tumor detection and segmentation from mammographic images and deploy it via a web service. The service will allow medical professionals to upload DICOM files, process the images, and receive the tumor detection and segmentation results.

## Milestones

The following milestones will guide the progress of the project:

1. **Project Plan Creation**: This milestone involves outlining the key objectives and timelines for the project.

2. **Data Visualization and Analysis**: This milestone focuses on visualizing and analyzing the data to gain insights into the dataset and extract information that can help with model development.

3. **Data Preparation and Initial Preprocessing**: At this stage, the dataset will be prepared and preprocessed. This includes handling missing data, normalizing images, and augmenting the dataset if necessary.

4. **Model Development**: This stage involves developing the machine learning models for tumor detection and segmentation.

5. **Model Evaluation and Refinement**: The performance of the machine learning models will be assessed using various evaluation metrics.

6. **Web Service Development and Deployment**: The final milestone involves creating a web service that takes DICOM files as input, runs tumor detection and segmentation, and provides results to the user. This milestone includes deploying the service, testing it, and ensuring that it is functional and user-friendly.

## Metrics Used for Evaluation

The performance of the tumor detection and segmentation model will be evaluated based on the following metrics:

* **Segmentation Metrics**:

  * **Dice Similarity Coefficient (Dice score)**: A measure of overlap between the predicted and true tumor regions, with values ranging from 0 (no overlap) to 1 (perfect overlap).

* **Detection Metrics**:

  * **Center Distance (Euclidean distance)**: The distance between the predicted center of the tumor and the actual center (as marked in the ground truth). This metric evaluates how accurately the model predicts the center of the tumor, which is critical for cropping the correct region for segmentation.

  * **Dice Score**: A metric used to evaluate the similarity between two sets, commonly used in image segmentation tasks. It measures the overlap between the predicted segmentation and the ground truth.

  * **Intersection over Union (IoU)**: A metric used to evaluate the accuracy of image segmentation. It measures the ratio of the intersection to the union of the predicted segmentation and the ground truth.
