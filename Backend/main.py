import io
import time
import psutil
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2

# XAI Imports
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

app = FastAPI(title="Plant Disease XAI API")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 28 # Update this to your actual number of classes from PlantDoc

model = models.mobilenet_v3_large(pretrained=False)
in_ft = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_ft, num_classes)

# Load your saved weights
model.load_state_dict(torch.load("plantdoc_mobilenet_v3.pth", map_location=device))
model.to(device)
model.eval()

# Get the last conv layer for CAM
target_layers = [model.features[-1][0]]

# Initialize CAM objects
cam_g = GradCAM(model=model, target_layers=target_layers)
cam_gp = GradCAMPlusPlus(model=model, target_layers=target_layers)
cam_s = ScoreCAM(model=model, target_layers=target_layers)
cam_e = EigenCAM(model=model, target_layers=target_layers)

# Define transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224)
])

def image_to_base64(img_array):
    """Converts numpy image to base64 string for JSON transfer"""
    img_pil = Image.fromarray(img_array)
    buff = io.BytesIO()
    img_pil.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    # --- Start Benchmarking ---
    start_time = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024) # MB

    # Read Image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Preprocess
    img_resized = transform(img)
    img_np = np.float32(img_resized) / 255.0
    input_tensor = preprocess_image(img_np, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

    # --- Inference ---
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
    
    # --- Generate XAI Heatmaps ---
    heatmaps = {}
    
    # Grad-CAM
    gray_g = cam_g(input_tensor=input_tensor, targets=None)[0, :]
    vis_g = show_cam_on_image(img_np, gray_g, use_rgb=True)
    heatmaps["gradcam"] = image_to_base64(vis_g)

    # Grad-CAM++
    gray_gp = cam_gp(input_tensor=input_tensor, targets=None)[0, :]
    vis_gp = show_cam_on_image(img_np, gray_gp, use_rgb=True)
    heatmaps["gradcam_plus"] = image_to_base64(vis_gp)

    # ScoreCAM
    gray_s = cam_s(input_tensor=input_tensor, targets=None)[0, :]
    vis_s = show_cam_on_image(img_np, gray_s, use_rgb=True)
    heatmaps["scorecam"] = image_to_base64(vis_s)

    # EigenCAM
    gray_e = cam_e(input_tensor=input_tensor, targets=None)[0, :]
    vis_e = show_cam_on_image(img_np, gray_e, use_rgb=True)
    heatmaps["eigencam"] = image_to_base64(vis_e)

    # --- End Benchmarking ---
    end_time = time.time()
    mem_after = process.memory_info().rss / (1024 * 1024)
    
    inference_time_ms = round((end_time - start_time) * 1000, 2)
    ram_used_mb = round(abs(mem_after - mem_before), 2)

    return {
        "status": "success",
        "prediction": int(predicted_idx.item()), 
        "confidence": float(confidence.item()),
        "benchmarks": {
            "inference_time_ms": inference_time_ms,
            "ram_usage_mb": ram_used_mb,
            "cpu_percent": psutil.cpu_percent()
        },
        "heatmaps": heatmaps
    }