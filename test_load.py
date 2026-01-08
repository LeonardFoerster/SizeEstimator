import os
import sys

print("Starting load test...")
try:
    from ultralytics import YOLO
    print("Ultralytics imported.")
    
    print("Loading YOLO...")
    # Use existing file
    if os.path.exists("yolo11n.pt"):
        model = YOLO("yolo11n.pt")
        print("YOLO loaded.")
    else:
        print("yolo11n.pt missing!")

    import onnxruntime as ort
    print("ONNX Runtime imported.")
    
    if os.path.exists("depth_anything_v2_vitl.onnx"):
        print("Loading ONNX model...")
        sess = ort.InferenceSession("depth_anything_v2_vitl.onnx", providers=['CPUExecutionProvider'])
        print("ONNX model loaded.")
    else:
        print("depth_anything_v2_vitl.onnx missing!")
        
except Exception as e:
    import traceback
    traceback.print_exc()
print("Done.")
