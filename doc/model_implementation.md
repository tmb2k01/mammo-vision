# Model Implementation

In this project, we utilize a two-stage approach for object detection and segmentation. The implementation involves the following components:

## 1. Object Detection with Fast R-CNN

We employ a Fast R-CNN model to detect objects of interest in the input data. This model is responsible for identifying bounding boxes around the detected objects. The detection process is efficient and provides high accuracy for object localization.

### **Inputs**

- **Images**: The input images are grayscale mammogram scans with shape `(1, H, W)`, where `H` and `W` are the height and width of the images.
- **Bounding Box Annotations**: Ground truth bounding boxes for the lesions (if available) in the format `[x_min, y_min, x_max, y_max]`.
- **Class Labels**: Labels associated with the bounding boxes (e.g., `1` for lesion, `0` for background).

### **Outputs**

- **Predicted Bounding Boxes**: A set of bounding boxes predicted by the model in the format `[x_min, y_min, x_max, y_max]`.
- **Predicted Class Labels**: The class label for each detected object.
- **Confidence Scores**: A confidence score associated with each prediction.

## 2. Segmentation with U-Net

Once the bounding boxes are detected, we use a U-Net model to perform segmentation within the detected areas. The U-Net model is designed to segment the specific regions of interest, providing pixel-level accuracy for the detected objects.

### **Inputs**

- **Cropped Image Patches**: Regions of interest extracted from the original images based on the bounding boxes.
- **Segmentation Masks**: Ground truth binary masks with shape `(2, H, W)`, where the actual class for each pixel is one-hot encoded.

### **Outputs**

- **Predicted Segmentation Masks**: A pixel-wise one-hot encoded mask where each class is assigned a probability of belonging to the lesion or the background for each pixel.
- **Dice Score**: A metric to evaluate segmentation accuracy by comparing predicted and ground truth masks.

## 3. Data Modules

The project includes two separate data modules to handle the data requirements for each model:

### **Detection Data Module**

- This module prepares and preprocesses the data required for training and evaluating the Fast R-CNN model.
- It loads mammogram images, applies augmentation (e.g., random flipping and zooming), and provides bounding box labels for training.

### **Segmentation Data Module**

- This module handles the data pipeline for the U-Net model, ensuring the segmentation task is performed effectively.
- It processes images and corresponding segmentation masks, applying necessary transformations like resizing and normalization.

## 4. Model Definitions

### **Fast R-CNN Model**

- Defined for object detection tasks, using a ResNet50-based Faster R-CNN architecture with a Feature Pyramid Network (FPN).
- Uses ROI pooling to detect lesions and classify them.
- Implements a custom head to adapt to the dataset.

### **U-Net Model**

- Defined for segmentation tasks, utilizing a fully convolutional U-Net architecture with encoder-decoder layers.
- Uses Dice Loss and Cross-Entropy Loss for training.
- Outputs a binary segmentation mask for lesion detection.

This modular approach ensures flexibility and scalability, allowing independent development and optimization of each component.
