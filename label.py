#!/usr/bin/env python
# label_mammograms.py: GUI with Tkinter for loading mammograms, with OpenCV for labeling, handling GUI errors
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import numpy as np

# Define class names (masses and microcalcifications with low/high malignancy)
class_names = ['mass_low', 'mass_high', 'microcalc_low', 'microcalc_high']

# Global variables for mouse interaction
drawing = False
start_x, start_y = -1, -1
boxes = []
current_class = 0  # Default to first class (mass_low)
img_copy = None

def draw_bounding_box(event, x, y, flags, param):
    global drawing, start_x, start_y, boxes, img_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img_copy.copy()
            cv2.rectangle(img_temp, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow("Labeling", img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y
        x_min, x_max = min(start_x, end_x), max(start_x, end_x)
        y_min, y_max = min(start_y, end_y), max(start_y, end_y)
        boxes.append((x_min, y_min, x_max, y_max, current_class))
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        label = f"{class_names[current_class]}"
        cv2.putText(img_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Labeling", img_copy)

def save_yolo_annotations(image_path, boxes, output_dir, img_width, img_height):
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    filename = Path(image_path).stem
    txt_path = os.path.join(label_dir, f"{filename}.txt")
    
    with open(txt_path, 'w') as f:
        for box in boxes:
            x_min, y_min, x_max, y_max, class_id = box
            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    output_image_path = os.path.join(image_dir, f"{filename}{Path(image_path).suffix}")
    cv2.imwrite(output_image_path, cv2.imread(image_path))
    print(f"Saved annotations to {txt_path} and image to {output_image_path}")

def label_image(image_path, output_dir):
    global img_copy, boxes, current_class

    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Image could not be loaded")
    except Exception as e:
        print(f"Error: Could not load image {image_path} - {str(e)}")
        return False

    img_copy = img.copy()
    boxes = []

    try:
        cv2.namedWindow("Labeling")
        cv2.setMouseCallback("Labeling", draw_bounding_box)
        cv2.imshow("Labeling", img_copy)
    except cv2.error as e:
        print(f"Error: OpenCV GUI not supported - {str(e)}")
        print("Please reinstall OpenCV with GUI support (e.g., pip install opencv-python)")
        return False

    print("Instructions:")
    print("- Left-click and drag to draw a bounding box.")
    print("- Press '0' for mass_low, '1' for mass_high, '2' for microcalc_low, '3' for microcalc_high.")
    print("- Press 's' to save and exit.")
    print("- Press 'q' to quit without saving.")
    print(f"Current class: {class_names[current_class]}")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if boxes:
                save_yolo_annotations(image_path, boxes, output_dir, img.shape[1], img.shape[0])
            else:
                print("No boxes drawn, skipping save.")
            break
        elif key in [ord('0'), ord('1'), ord('2'), ord('3')]:
            current_class = int(chr(key))
            print(f"Switched to class: {class_names[current_class]}")

    cv2.destroyAllWindows()
    return True

class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mammogram Labeling Tool")
        self.root.geometry("400x200")

        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar(value="D:/INBreast/custom_dataset")

        tk.Label(root, text="Input Image or Directory:").pack(pady=5)
        tk.Entry(root, textvariable=self.input_path, width=40).pack()
        tk.Button(root, text="Browse", command=self.browse_input).pack(pady=5)

        tk.Label(root, text="Output Directory:").pack(pady=5)
        tk.Entry(root, textvariable=self.output_dir, width=40).pack()
        tk.Button(root, text="Browse Output", command=self.browse_output).pack(pady=5)

        tk.Button(root, text="Start Labeling", command=self.start_labeling).pack(pady=10)

    def browse_input(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if not path:
            path = filedialog.askdirectory()
        if path:
            self.input_path.set(path)

    def browse_output(self):
        directory = filedialog.askdirectory()
        if directory:
            self.output_dir.set(directory)

    def start_labeling(self):
        input_path = self.input_path.get()
        output_dir = self.output_dir.get()

        if not input_path:
            messagebox.showerror("Error", "Please select an input image or directory.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please specify an output directory.")
            return

        labeled_any = False
        if os.path.isdir(input_path):
            for filename in os.listdir(input_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(input_path, filename)
                    print(f"Labeling {image_path}")
                    if label_image(image_path, output_dir):
                        labeled_any = True
        else:
            print(f"Labeling {input_path}")
            if label_image(input_path, output_dir):
                labeled_any = True
        
        if labeled_any:
            messagebox.showinfo("Done", "Labeling completed!")
        else:
            messagebox.showwarning("Warning", "No valid images were labeled.")

def main():
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()