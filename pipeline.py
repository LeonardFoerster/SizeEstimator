from ultralytics import YOLO
import cv2
import torch
from depth_anything.dpt import DepthAnything  # Assuming cloned repo is in PYTHONPATH

# Load Depth Anything V2 model
depth_model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vit_large").eval()

# Generate depth map (input: image tensor)
depth_map = depth_model(torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float())
depth_map = depth_map.squeeze().cpu().numpy()  # Normalize if needed

# Load pretrained YOLOv8 model
model = YOLO("yolo11n.pt")  # Downloads automatically if needed

# Load image
img = cv2.imread("input_image.jpg")

# Run inference
results = model(img)

# Extract detections
detections = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
        cls_id = int(box.cls)
        class_name = result.names[cls_id]
        detections.append({"class": class_name, "bbox": (x1, y1, x2, y2)})
