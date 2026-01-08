import sys
import os
import logging

# Immediate debug print
print("DEBUG: server.py is starting initialization...", flush=True)

# Light imports only
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo11n.pt")
DEPTH_ONNX_PATH = os.getenv("DEPTH_ONNX_PATH", "depth_anything_v2_vitl.onnx")
DEFAULT_REF_CLASS = os.getenv("DEFAULT_REF_CLASS", "bottle")
DEFAULT_REF_WIDTH = float(os.getenv("DEFAULT_REF_WIDTH", "7.0"))

# --- Global State ---
models = {}
startup_error = None

class OnnxDepthModel:
    def __init__(self, model_path):
        # Lazy imports inside the class
        import onnxruntime as ort
        import numpy as np
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
            
        print(f"Loading ONNX model from {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Robustly handle input dimensions (can be int, None, or string for dynamic axes)
        h_dim = self.input_shape[2]
        w_dim = self.input_shape[3]
        
        if isinstance(h_dim, int):
            self.input_height = h_dim
        else:
            self.input_height = 518
            
        if isinstance(w_dim, int):
            self.input_width = w_dim
        else:
            self.input_width = 518
            
        # Ensure they are ints
        self.input_height = int(self.input_height)
        self.input_width = int(self.input_width)
        
        print(f"DEBUG: ONNX Input Shape: {self.input_shape}, Using size: ({self.input_width}, {self.input_height})")

    def preprocess(self, image):
        import cv2
        import numpy as np
        
        h, w = image.shape[:2]
        img_input = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_input = (img_input - mean) / std
        
        img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, axis=0)
        return img_input, (h, w)

    def infer_image(self, raw_image):
        import cv2
        import numpy as np
        
        img_tensor, (orig_h, orig_w) = self.preprocess(raw_image)
        output = self.session.run(None, {self.input_name: img_tensor})
        depth = output[0]
        
        if len(depth.shape) == 3:
            depth = depth[0]
        elif len(depth.shape) == 4:
            depth = depth[0, 0]
            
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return depth

@asynccontextmanager
async def lifespan(app: FastAPI):
    # We no longer load models here to ensure fast startup for Lambda Adapter
    print("DEBUG: Application starting up (lazy loading enabled)...")
    yield
    models.clear()
    print("DEBUG: Application shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler
from fastapi.requests import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    error_details = traceback.format_exc()
    print(f"CRITICAL UNHANDLED ERROR: {error_details}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "error": str(exc), "traceback": error_details},
        headers={"Access-Control-Allow-Origin": "*"}
    )

def get_or_load_models():
    """Lazy load models if they are not already in memory. Downloads from S3 if needed."""
    global startup_error
    if models.get('loaded'):
        return

    print("DEBUG: Checking models...")
    
    # Paths (Lambda is read-only except /tmp)
    # If we are in Lambda (determined by env var or path), we might need to use /tmp
    # We prefer /tmp for downloaded models
    
    local_yolo_path = YOLO_MODEL_NAME
    local_depth_path = DEPTH_ONNX_PATH
    
    bucket_name = os.getenv("MODEL_BUCKET")
    
    try:
        import boto3
        s3 = boto3.client('s3')
        
        def download_if_needed(filename, target_path):
            if os.path.exists(target_path):
                print(f"DEBUG: Found {filename} at {target_path}")
                return target_path
            
            if not bucket_name:
                print(f"WARNING: No MODEL_BUCKET set. Cannot download {filename}. Expecting local file.")
                return filename # Fallback to hoping it's in the CWD
                
            print(f"Downloading {filename} from S3 bucket {bucket_name} to {target_path}...")
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            s3.download_file(bucket_name, filename, target_path)
            print(f"Download of {filename} complete.")
            return target_path

        # Determine where to look/save
        # On Lambda, use /tmp. On local, use current dir.
        # simple check: if generic path exists, use it. if not, try /tmp.
        
        real_yolo_path = YOLO_MODEL_NAME
        if not os.path.exists(real_yolo_path):
             # Try /tmp
             tmp_path = f"/tmp/{YOLO_MODEL_NAME}"
             real_yolo_path = download_if_needed(YOLO_MODEL_NAME, tmp_path)

        real_depth_path = DEPTH_ONNX_PATH
        if not os.path.exists(real_depth_path):
             tmp_path = f"/tmp/{DEPTH_ONNX_PATH}"
             real_depth_path = download_if_needed(DEPTH_ONNX_PATH, tmp_path)

        # Lazy import explicitly for loading
        from ultralytics import YOLO
        
        if 'yolo' not in models:
            print(f"Loading YOLO from {real_yolo_path}...")
            models['yolo'] = YOLO(real_yolo_path)
        
        if 'depth' not in models:
            print(f"Loading Depth Model from {real_depth_path}...")
            models['depth'] = OnnxDepthModel(real_depth_path)
            
        models['loaded'] = True
        print("Models loaded successfully.")
    except Exception as e:
        import traceback
        startup_error = traceback.format_exc()
        print(f"CRITICAL MODEL LOAD ERROR: {startup_error}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {startup_error}")

@app.get("/")
def health_check():
    import os # ensure os is available
    if startup_error:
        return {
            "status": "error", 
            "service": "size-estimator", 
            "startup_error": startup_error,
            "cwd": os.getcwd(),
            "files": os.listdir(".")
        }
    return {"status": "healthy", "service": "size-estimator", "models_loaded": models.get('loaded', False)}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    ref_class: str = Form(DEFAULT_REF_CLASS),
    ref_width: float = Form(DEFAULT_REF_WIDTH)
):
    # Lazy imports for request handling
    import cv2
    import numpy as np
    import base64
    
    # Ensure models are loaded
    get_or_load_models()

    if startup_error:
        raise HTTPException(status_code=500, detail=f"Server startup failed: {startup_error}")

    contents = await file.read()
    print(f"DEBUG: Received file '{file.filename}' with size {len(contents)} bytes")

    # Try loading with OpenCV first
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Fallback to Pillow if OpenCV fails
    if img is None:
        print("DEBUG: cv2.imdecode failed, trying PIL...")
        try:
            from PIL import Image
            import io
            image_pil = Image.open(io.BytesIO(contents))
            # Convert to RGB if needed (OpenCV expects BGR usually, but let's stick to standard)
            # Actually OpenCV uses BGR. PIL uses RGB.
            # Let's convert PIL RGB -> OpenCV BGR
            image_pil = image_pil.convert('RGB') 
            img = np.array(image_pil) 
            img = img[:, :, ::-1].copy() # RGB to BGR
            print("DEBUG: PIL loaded image successfully.")
        except Exception as e:
            print(f"DEBUG: PIL also failed: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file. cv2 and PIL failed. Size: {len(contents)}")

    if img is None:
         raise HTTPException(status_code=400, detail="Image decode failed completely.")

    depth_model = models.get('depth')
    yolo_model = models.get('yolo')
    
    if not depth_model or not yolo_model:
        raise HTTPException(status_code=500, detail="Models not loaded")

    depth_map = depth_model.infer_image(img)
    results = yolo_model(img)

    detections = []
    ref_detection = None
    
    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls)
            label = result.names[cls_id]
            x1, y1, x2, y2 = map(int, coords)
            
            w_px = x2 - x1
            h_px = y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            
            center_x = min(max(center_x, 0), depth_map.shape[1] - 1)
            center_y = min(max(center_y, 0), depth_map.shape[0] - 1)
            
            # --- SMART DEPTH SAMPLING START ---
            # Extract the Region of Interest (ROI) from the depth map
            # Ensure coordinates are within bounds
            d_y1 = max(0, y1)
            d_y2 = min(depth_map.shape[0], y2)
            d_x1 = max(0, x1)
            d_x2 = min(depth_map.shape[1], x2)
            
            depth_roi = depth_map[d_y1:d_y2, d_x1:d_x2]
            
            obj_depth = 0.0
            if depth_roi.size > 0:
                # Crop to the central 50% to avoid edge noise/background
                h_roi, w_roi = depth_roi.shape
                crop_h = int(h_roi * 0.25)
                crop_w = int(w_roi * 0.25)
                
                # Ensure we don't crop to nothing
                if crop_h * 2 < h_roi and crop_w * 2 < w_roi:
                    depth_roi_central = depth_roi[crop_h:-crop_h, crop_w:-crop_w]
                else:
                    depth_roi_central = depth_roi
                
                # Use Median depth - robust against outliers
                if depth_roi_central.size > 0:
                    obj_depth = float(np.median(depth_roi_central))
                else:
                    # Fallback to center point if ROI somehow fails
                    obj_depth = float(depth_map[center_y, center_x])
            else:
                 obj_depth = float(depth_map[center_y, center_x])
            # --- SMART DEPTH SAMPLING END ---

            det = {
                "label": label,
                "box": [x1, y1, x2, y2],
                "w_px": w_px,
                "h_px": h_px,
                "depth": obj_depth
            }
            detections.append(det)

            if label == ref_class and ref_detection is None:
                ref_detection = det

    response_data = {
        "reference_found": False,
        "objects": []
    }

    if ref_detection:
        response_data["reference_found"] = True
        K = (ref_width * ref_detection['depth']) / ref_detection['w_px']
        
        for det in detections:
            est_width = K * (det['w_px'] / det['depth'])
            response_data["objects"].append({
                "label": det["label"],
                "box": det["box"],
                "estimated_width_cm": round(est_width, 2)
            })
            
            x1, y1, x2, y2 = det['box']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{det['label']} {est_width:.1f}cm", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        for det in detections:
            response_data["objects"].append({
                "label": det["label"],
                "box": det["box"],
                "estimated_width_cm": None
            })

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    response_data["image_base64"] = img_base64
    
    return JSONResponse(content=response_data)
