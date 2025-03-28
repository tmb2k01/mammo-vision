# Data Preparation

This document outlines the steps for preparing mammographic images before feeding them into a machine learning model. These steps include converting DICOM files, normalizing pixel values, resizing images, enhancing contrast, applying data augmentation, and preparing data for segmentation tasks.

To prepare the data, simply run the [`scripts/prepare_data.sh`](../scripts/prepare_data.sh) script. To do so, follow the steps below. Make sure to execute these commands from the root directory of the project. The script will only succeed if you have placed your dataset in the `data/cbis-ddsm` directory.

```bash
chmod +x scripts/prepare_data.sh
./scripts/prepare_data.sh
```

For training, use the `CbisDdsmDataModule` class from [`src.data_module`](../src/data_module.py). This will ensure that all images are augmented and transformed to a consistent size.

Before performing detection and segmentation on images in a web service, several preprocessing steps are required. Specifically, DICOM to PNG conversion, contrast adjustment, and resizing must be done before the mammographic images can be fed into the machine learning model.

## 1. Web Service Specific Transformations

### 1.1. DICOM to PNG Conversion

The mammographic images we have access to are stored in the **DICOM (Digital Imaging and Communications in Medicine)** format, which needs to be converted to more commonly used image formats, like **PNG**, for further processing.

## 2. Common Transformation Steps

### 2.1. Contrast Adjustment with CLAHE

Contrast enhancement improves the visibility of key features, such as tumors or microcalcifications, in mammographic images. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** is widely used in medical imaging to enhance local contrast, especially in low-contrast images, while preventing the over-enhancement of noise.

### 2.2. Data Augmentation

Data augmentation is used to artificially expand the training dataset by applying various transformations to the original images. This helps reduce overfitting and improves the model's generalization. For our specific use case, we chose the following augmentation techniques:

* **Horizontal Flipping**: Flip images horizontally to simulate different orientations and variations in the data.

* **Vertical Flipping**: Flip images vertically to simulate different orientations and variations in the data.

* **Zooming**: Random zooming is applied to simulate variations in the scale of features, ensuring that the model can recognize features at different sizes.

### 2.3. Resizing

Mammographic images may have varying sizes and resolutions, making resizing necessary to standardize the input dimensions. A fixed image size ensures that the input is compatible with most deep learning models.

### 2.4. Normalization/Scaling of Pixel Values

The mammographic images have varying pixel value ranges. To ensure consistent input for the machine learning model, pixel values are normalized to a standard range, typically **[0, 1]** for floating-point images.

## 3. Segmentation-Specific Transformations

### 3.1. Mask-based Cropping

Using the segmentation masks from the original dataset, we will crop out the tumor regions from the images. These cropped regions, which focus on the areas containing the tumors, will serve as input for training the segmentation model.
