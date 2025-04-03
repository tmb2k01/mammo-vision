import cv2
import gradio as gr
import numpy as np
import pydicom
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.detection_model import DetectionModel
from src.models.segmentation_model import SegmentationModel

IMG_SIZE = 512
MODEL_DETEC = None
MODEL_SEGME = None
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_models():
    MODEL_DETEC = DetectionModel(weight_path="./models/mass-detection.ckpt")
    MODEL_DETEC.eval()
    MODEL_SEGME = SegmentationModel(weight_path="./models/mass-segmentation.ckpt")
    MODEL_SEGME.eval()


def predict(image):
    image = CLAHE.apply(np.array(image, dtype=np.uint8))
    image = Image.fromarray(image)
    image = torch.tensor(image).unsqueeze(0)

    # Run detection model
    with torch.no_grad():
        detections = MODEL_DETEC(image)

    # Prepare for segmentation
    segmentation_masks = []
    for box in detections["boxes"]:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image = TF.resize(cropped_image, [256, 256])
        cropped_image = TF.to_tensor(cropped_image).unsqueeze(0)

        # Run segmentation model
        with torch.no_grad():
            mask = MODEL_SEGME(cropped_image)
            mask = mask.squeeze().numpy()
            mask = cv2.resize(mask, (x_max - x_min, y_max - y_min))
            segmentation_masks.append((x_min, y_min, mask))

    # Add boxes and segmentation to the original image
    image = np.array(image)
    for box, (x_min, y_min, mask) in zip(detections["boxes"], segmentation_masks):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        mask = (mask > 0.5).astype(np.uint8) * 255
        image[y_min:y_max, x_min:x_max] = cv2.addWeighted(
            image[y_min:y_max, x_min:x_max], 0.7, mask, 0.3, 0
        )

    image = Image.fromarray(image)
    print(image.size)

    return empty_image()


def empty_image():
    pixels = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    return Image.fromarray(pixels)


def resize(image):
    image = image.copy()
    width, height = image.size
    new_width = IMG_SIZE * width // height
    image.resize((new_width, IMG_SIZE), Image.Resampling.LANCZOS)
    return image


def find_tumors(filepath):
    if filepath is None:
        return empty_image(), empty_image()

    dicom_file = pydicom.dcmread(filepath)
    pixels = dicom_file.pixel_array

    # Scale values between 0 and 255
    pixels = pixels - np.min(pixels)
    pixels = (pixels / np.max(pixels)) * 255

    orig_image = Image.fromarray(pixels)
    image = resize(orig_image)
    prediction = resize(predict(orig_image))

    return image, prediction


def launch():
    load_models()

    with gr.Blocks() as ui:
        gr.Markdown("# Breast Tumor Detection")
        dicom_file = gr.File(label="Upload DICOM File", type="filepath")
        button = gr.Button("Detect Tumors")
        with gr.Row():
            slice_image = gr.Image(height=IMG_SIZE, label="DICOM Image")
            slice_segmentation = gr.Image(height=IMG_SIZE, label="Model Prediction")

        button.click(
            fn=find_tumors,
            inputs=[dicom_file],
            outputs=[slice_image, slice_segmentation],
        )

    print("Launching gradio interface...")
    ui.launch(share=True)
