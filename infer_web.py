#!/usr/bin/env python
# infer.py: Web interface with Gradio to predict lesions on uploaded mammograms using YOLOv11
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os

# Define paths
model_path = r"D:\INBreast\runs\inbreast_lesion\weights\best.pt"

# Load the trained model once (outside the function for efficiency)
model = YOLO(model_path)

# Class names from your training (must match inbreast.yaml)
class_names = ['mass_low', 'mass_high']

# Inference function
def predict_lesion(image):
    # Convert Gradio image (numpy array) to PIL Image
    img = Image.fromarray(image).convert('RGB')
    width, height = img.size

    # Perform inference
    results = model.predict(img, conf=0.25, iou=0.45)  # Adjust thresholds as needed
    result = results[0]

    # Plot the image with predictions
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    # Add bounding boxes and labels
    detection_text = "No detections found."
    if result.boxes:
        detection_text = "Detections:\n"
        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()  # [x_min, y_min, x_max, y_max]
            class_id = int(box.cls[0].cpu().numpy())
            confidence = box.conf[0].cpu().numpy()

            # Convert to x_min, y_min, width, height
            x_min, y_min, x_max, y_max = xyxy
            w = x_max - x_min
            h = y_max - y_min

            # Draw bounding box
            rect = patches.Rectangle(
                (x_min, y_min), w, h,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Add label
            category_name = f"{class_names[class_id]} ({confidence:.2f})"
            ax.text(x_min, y_min - 10, category_name, color='r', fontsize=12, weight='bold')

            # Update detection text
            detection_text += f"Class: {class_names[class_id]}, Confidence: {confidence:.2f}, Box: {xyxy}\n"

    ax.axis('off')

    # Convert plot to image for Gradio
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    output_img = Image.open(buf)

    return output_img, detection_text

# Gradio interface
with gr.Blocks(title="Mammogram Lesion Detection with YOLOv11") as demo:
    gr.Markdown("# Mammogram Lesion Detection")
    gr.Markdown("Upload a mammogram image to detect lesions using a trained YOLOv11 model.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Mammogram", type="numpy")
            submit_btn = gr.Button("Detect Lesions")
        
        with gr.Column():
            output_image = gr.Image(label="Result with Detections")
            output_text = gr.Textbox(label="Detection Details", lines=5)

    # Connect the button to the prediction function
    submit_btn.click(
        fn=predict_lesion,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

# Launch the interface with sharing enabled
demo.launch(share=True)