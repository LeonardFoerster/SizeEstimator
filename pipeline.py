from ultralytics import YOLO
import cv2
import torch
from depth_anything.dpt import DepthAnything  

depth_model = DepthAnything.from_pretrained("LiheYoung/depth_anything_vit_large").eval()

depth_map = depth_model(torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float())
depth_map = depth_map.squeeze().cpu().numpy() 

model = YOLO("yolo11n.pt")  
img = cv2.imread("input_image.jpg")
results = model(img)

detections = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  
        cls_id = int(box.cls)
        class_name = result.names[cls_id]
        detections.append({"class": class_name, "bbox": (x1, y1, x2, y2)})
