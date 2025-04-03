import cv2
import gradio as gr
import numpy as np
import pydicom
import torch
import torchvision.ops as ops
import torchvision.transforms.functional as TF
from PIL import Image

from src.models.detection_model import DetectionModel
from src.models.segmentation_model import SegmentationModel

IMG_SIZE = 512
MODEL_DETEC = None
MODEL_SEGME = None
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_models():
    global MODEL_DETEC
    global MODEL_SEGME

    MODEL_DETEC = DetectionModel(weight_path="models/mass-detection.ckpt")
    MODEL_DETEC = MODEL_DETEC.eval()
    MODEL_SEGME = SegmentationModel.load_from_checkpoint(
        "models/mass-segmentation.ckpt",
        map_location="cpu",
    )
    MODEL_SEGME = MODEL_SEGME.eval()


def predict(image):
    image = CLAHE.apply(np.array(image, dtype=np.uint8))
    image = Image.fromarray(image)
    image_tensor = TF.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        detections = MODEL_DETEC(image_tensor)[0]

    segmentation_masks = []

    confidence_threshold = 0.4
    iou_threshold = 0.3
    scores = detections["scores"]
    boxes = detections["boxes"]
    predicted_boxes = boxes[scores > confidence_threshold]
    keep_indices = ops.nms(
        predicted_boxes,
        scores[scores > confidence_threshold],
        iou_threshold,
    )
    filtered_boxes = predicted_boxes[keep_indices]

    for box in filtered_boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        cropped_image = TF.resize(cropped_image, [256, 256])
        cropped_image = TF.to_tensor(cropped_image).unsqueeze(0)

        with torch.no_grad():
            mask = MODEL_SEGME(cropped_image)
            mask = torch.argmax(mask, dim=1)
            mask = TF.resize(mask, [x_max - x_min, y_max - y_min]).squeeze(0).numpy()
            segmentation_masks.append(mask)

    image = np.array(image)
    for box, mask in zip(predicted_boxes, segmentation_masks):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        mask = np.transpose(mask * 255).astype(np.uint8)
        orig = image[y_min:y_max, x_min:x_max]
        image[y_min:y_max, x_min:x_max] = cv2.addWeighted(orig, 0.7, mask, 0.3, 0)

    return Image.fromarray(image)


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
