# Data Preparation

This document outlines the steps for preparing mammographic images before feeding them into a machine learning model. These steps include converting DICOM files, normalizing pixel values, resizing images, enhancing contrast, applying data augmentation, and preparing data for segmentation tasks.

## 1. Web Service Specific Transformations

### 1.1. DICOM to PNG Conversion

The mammographic images we have access to are stored in the **DICOM (Digital Imaging and Communications in Medicine)** format, which needs to be converted to more commonly used image formats, like **PNG**, for further processing.

## 2. Common Transformation Steps

### 2.1. Normalization/Scaling of Pixel Values

The mammographic images have varying pixel value ranges. To ensure consistent input for the machine learning model, pixel values are normalized to a standard range, typically **[0, 1]** for floating-point images.

### 2.2. Contrast Adjustment with CLAHE

Contrast enhancement improves the visibility of key features, such as tumors or microcalcifications, in mammographic images. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is widely used in medical imaging to enhance local contrast, especially in low-contrast images, while preventing the over-enhancement of noise.

### 2.3. Data Augmentation

Data augmentation is used to artificially expand the training dataset by applying various transformations to the original images. This helps reduce overfitting and improves the model's generalization. For our specific use case, we chose the following augmentation techniques:

* **Vertical Flipping**: Flip images vertically to simulate different orientations and variations in the data.

* **Zooming**: Random zooming is applied to simulate variations in the scale of features, ensuring that the model can recognize features at different sizes.

### 2.4. Resizing

Mammographic images may have varying sizes and resolutions, making resizing necessary to standardize the input dimensions. A fixed image size (e.g., 224x224 or 512x512 pixels) ensures that the input is compatible with most deep learning models.

## 3. Segmentation-Specific Transformations

### 3.1. Mask-based Cropping

Using the segmentation masks from the original dataset, we will crop out the tumor regions from the images. These cropped regions, which focus on the areas containing the tumors, will serve as input for training the segmentation model.
