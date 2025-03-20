import gradio as gr
import numpy as np
import pydicom
import torch
import torchvision.transforms.functional as TF
from PIL import Image

IMG_HEIGHT = 512


def load_models():
    # TODO: Load models into global variables
    pass


def predict(image):
    image = TF.resize(image, (416, 416))
    image = TF.to_tensor(image).to(torch.uint8)

    # TODO: Add actual prediction and resize to original size

    return empty_image()


def empty_image():
    pixels = np.zeros((IMG_HEIGHT, IMG_HEIGHT), dtype=np.uint8)
    return Image.fromarray(pixels)


def resize(image):
    image = image.copy()
    width, height = image.size
    new_width = IMG_HEIGHT * width // height
    image.resize((new_width, IMG_HEIGHT), Image.Resampling.LANCZOS)
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
    with gr.Blocks() as ui:
        gr.Markdown("# Breast Tumor Detection")
        dicom_file = gr.File(label="Upload DICOM File", type="filepath")
        button = gr.Button("Detect Tumors")
        with gr.Row():
            slice_image = gr.Image(height=IMG_HEIGHT, label="DICOM Image")
            slice_segmentation = gr.Image(height=IMG_HEIGHT, label="Model Prediction")
        gr.Textbox(
            """
            🟣 Calc
            🔵 Mass
            """,
            label="Legend",
        )

        button.click(
            fn=find_tumors,
            inputs=[dicom_file],
            outputs=[slice_image, slice_segmentation],
        )

    print("Launching gradio interface...")
    ui.launch(share=True)


if __name__ == "__main__":
    load_models()
    launch()
