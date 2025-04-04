#!/usr/bin/env python
# Script to create inbreast.yaml and train YOLOv11 on INbreast dataset with Windows multiprocessing fix
import yaml
import os
import torch
from ultralytics import YOLO
from multiprocessing import freeze_support  # For Windows compatibility

# Define the main training function
def main():
    # Step 1: Create the YAML file
    dataset_config = {
        'train': r'D:/INBreast/train/images',
        'val': r'D:/INBreast/val/images',
        'nc': 4,
        'names': ['mass_low', 'mass_high', 'microcalc_low', 'microcalc_high']
    }

    yaml_file_path = r'D:\INBreast\inbreast.yaml'
    os.makedirs(os.path.dirname(yaml_file_path), exist_ok=True)
    with open(yaml_file_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"YAML file created at: {yaml_file_path}")

    # Step 2: Check for GPU availability
    device = 'cpu'  # Default to CPU
    if torch.cuda.is_available():
        device = 0  # Use first GPU if CUDA is available
        print(f"CUDA available! Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Training on CPU.")

    # Step 3: Train YOLOv11
    model_variant = "yolo11l.pt"  # Nano model
    model = YOLO(model_variant)

    model.train(
        data=yaml_file_path,
        epochs=50,
        imgsz=640,
        batch=16,
        name="inbreast_lesion",
        project="D:/INBreast/runs",
        device=device,
        patience=10
    )

    print("Training completed! Results saved in D:/INBreast/runs/inbreast_lesion")

# Protect the main code for Windows multiprocessing
if __name__ == '__main__':
    freeze_support()  # Optional, for frozen executables; safe to include
    main()