import argparse
import sys
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import importlib

def get_depth_model(encoder='vitl', device='cuda', metric=False, dataset='hypersim'):
    # Clear any existing depth_anything_v2 modules to force reload from the correct path
    for key in list(sys.modules.keys()):
        if key.startswith('depth_anything_v2'):
            del sys.modules[key]

    # Dynamic import based on metric flag
    cwd = os.getcwd()
    if metric:
        module_path = os.path.join(cwd, 'Depth-Anything-V2/metric_depth')
    else:
        module_path = os.path.join(cwd, 'Depth-Anything-V2')
        
    sys.path.insert(0, module_path)

    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("Error: Could not import DepthAnythingV2. Make sure the 'Depth-Anything-V2' folder is in the current directory.")
        sys.exit(1)
    finally:
        # Clean up sys.path to avoid polluting it for subsequent calls (if any)
        if sys.path[0] == module_path:
            sys.path.pop(0)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if metric:
        max_depth = 80 if dataset == 'vkitti' else 20
        # Initialize metric model with max_depth
        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        # Metric checkpoints usually reside in a different folder or have specific naming
        # Adjusting path to match typical structure or user's setup
        checkpoint_path = f'Depth-Anything-V2/metric_depth/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth'
    else:
        # Initialize standard model
        model = DepthAnythingV2(**model_configs[encoder])
        checkpoint_path = f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        if metric:
             print(f"Please download the metric depth checkpoint for {dataset} and place it in Depth-Anything-V2/metric_depth/checkpoints/")
        else:
             print("Please download the checkpoint and place it in Depth-Anything-V2/checkpoints/")
        sys.exit(1)
        
    # Load model state dict
    # Using weights_only=True to avoid FutureWarning and improve security
    # If this fails with older pytorch versions, remove the argument
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        # Fallback for older pytorch versions that don't support weights_only
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Estimate object sizes using YOLO and Depth Anything V2")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--ref_class", type=str, default="bottle", help="Class name of reference object (e.g., bottle, cup, cell phone)")
    parser.add_argument("--ref_width", type=float, default=7.0, help="Real width of reference object in cm")
    parser.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--encoder", type=str, default="vitl", help="Depth Anything V2 encoder (vits, vitb, vitl)")
    parser.add_argument("--metric", action="store_true", help="Use metric depth estimation models")
    parser.add_argument("--dataset", type=str, default="hypersim", choices=["hypersim", "vkitti"], help="Dataset for metric depth (hypersim=indoor, vkitti=outdoor)")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to save output image")
    parser.add_argument("--verbose", action="store_true", help="Print debug information")
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Load Models
    print("Loading YOLO model...")
    try:
        yolo = YOLO(args.yolo_model)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        # Fallback to yolov8n if 11 is not available/compatible
        print("Falling back to yolov8n.pt...")
        yolo = YOLO("yolov8n.pt")

    print(f"Loading Depth Anything V2 model (Metric: {args.metric})...")
    depth_model = get_depth_model(args.encoder, device, args.metric, args.dataset)

    # 2. Process Image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Could not read image {args.image}")
        sys.exit(1)
    
    # Get Depth Map
    print("Estimating depth...")
    depth_map = depth_model.infer_image(img) # Returns (H, W) numpy array
    
    # Run Object Detection
    print("Detecting objects...")
    results = yolo(img)
    
    # 3. Analyze Detections
    detections = []
    ref_detection = None
    
    # Parse YOLO results
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy() # x1, y1, x2, y2
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = result.names[cls_id]
            
            x1, y1, x2, y2 = map(int, coords)
            w_px = x2 - x1
            h_px = y2 - y1
            
            # Get median depth for the object
            # We use a center crop of the box to avoid background noise
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            margin_w, margin_h = int(w_px * 0.2), int(h_px * 0.2)
            
            # Ensure indices are within bounds
            d_y1 = max(y1 + margin_h, 0)
            d_y2 = min(y2 - margin_h, depth_map.shape[0])
            d_x1 = max(x1 + margin_w, 0)
            d_x2 = min(x2 - margin_w, depth_map.shape[1])
            
            if d_x2 > d_x1 and d_y2 > d_y1:
                depth_crop = depth_map[d_y1:d_y2, d_x1:d_x2]
                obj_depth = np.median(depth_crop)
            else:
                obj_depth = depth_map[center_y, center_x]

            det = {
                "label": label,
                "box": (x1, y1, x2, y2),
                "w_px": w_px,
                "h_px": h_px,
                "depth": obj_depth
            }
            detections.append(det)
            
            # Check for reference object (picking the one with highest confidence or first found)
            if label == args.ref_class and ref_detection is None:
                ref_detection = det
                
    # 4. Calculate Size
    img_out = img.copy()
    
    if ref_detection:
        print(f"Reference object found: {ref_detection['label']} (Depth: {ref_detection['depth']:.2f})")
        
        # Calculate Calibration
        if args.metric:
            # Metric Depth Logic (Depth is in meters/units)
            # Z = Depth
            # W_real = (W_px * Z) / f  =>  f = (W_px * Z) / W_real
            # Using reference to calibrate focal length 'f'
            focal_length_px = (ref_detection['w_px'] * ref_detection['depth']) / args.ref_width
            print(f"Calibrated Focal Length (px): {focal_length_px:.2f}")
        else:
            # Relative Depth/Disparity Logic
            # Assuming depth_map output is disparity (inverse depth) for base models
            # Real = K * Px / Disp => K = Real * Disp / Px
            K = (args.ref_width * ref_detection['depth']) / ref_detection['w_px']
        
        for det in detections:
            # Calculate estimated width
            if args.metric:
                 # W_real = (W_px * Z) / f
                 est_width = (det['w_px'] * det['depth']) / focal_length_px
            else:
                 # W_real = K * (W_px / Depth)
                 est_width = K * (det['w_px'] / det['depth'])
            
            # Draw box and text
            x1, y1, x2, y2 = det['box']
            color = (0, 255, 0) if det == ref_detection else (0, 255, 255)
            cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"{det['label']} W:{est_width:.1f}cm"
            if det == ref_detection:
                label_text += " (Ref)"
            
            cv2.putText(img_out, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if args.verbose:
                print(f"Object: {det['label']:<10} | W_px: {det['w_px']:<4} | Depth: {det['depth']:<6.2f} | Est Width: {est_width:.2f}cm")
            else:
                print(f"Object: {det['label']}, Est Width: {est_width:.2f}cm")
            
    else:
        print(f"Warning: Reference object '{args.ref_class}' not found.")
        print("Available objects:", list(set(d['label'] for d in detections)))
        # Just draw boxes without size
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_out, det['label'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(args.output, img_out)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
