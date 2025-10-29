import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageEnhance
import os
import io
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from ultralytics import YOLO
import faiss
import json
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

# --- 1. FastAPI App Initialization & CORS ---
app = FastAPI(title="AgriAssist API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 2. Load All Models on Startup ---
device = torch.device("cpu")
print("--- 📱 Using device: cpu ---")

# Load Image Classification Models
try:
    print("--- 📥 Loading YOLO model... ---")
    yolo_model = YOLO("yolov8n.pt")
    print("--- ✅ YOLO model loaded. ---")
except Exception: yolo_model = None

print(f"--- 📥 Loading classification model... ---")
# 👈 THE FIX IS HERE
checkpoint = torch.load("plant_disease_best.pth", map_location=device, weights_only=False)
class_names = checkpoint['classes']
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(256, len(class_names)))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"--- ✅ Classification model loaded. ---")

# Load RAG Q&A Models
print("--- 📥 Loading RAG system... ---")
docs = []
with open("kb_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        docs.append(json.loads(line))
rag_index = faiss.read_index("faiss_index.bin")
rag_model = SentenceTransformer("all-MiniLM-L6-v2")
print("--- ✅ RAG system loaded. ---")

# --- 3. Image Prediction Helper Functions ---
base_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def predict_single(image):
    with torch.no_grad(): outputs = model(base_transform(image).unsqueeze(0).to(device))
    return torch.softmax(outputs, dim=1)[0]

# --- 4. Multilingual Recommendations ---
recommendations = {
    "Pepper__bell___Bacterial_spot": { "en": ["Remove infected leaves.", "Avoid overhead watering.", "Apply copper-based bactericides."], "hi": ["संक्रमित पत्तियों को हटा दें।", "ऊपर से पानी देने से बचें।", "कॉपर आधारित जीवाणुनाशकों का प्रयोग करें।"] },
    "Tomato_healthy": { "en": ["The plant is healthy."], "hi": ["पौधा स्वस्थ है।"] },
    # PASTE YOUR FULL RECOMMENDATIONS DICTIONARY HERE
}
default_recommendation = {"en": ["No recommendation available."], "hi": ["कोई सिफारिश उपलब्ध नहीं है।"]}

# --- 5. The API Endpoints ---
@app.post("/predict")
async def predict(file: UploadFile = File(...), lang: Optional[str] = 'en'):
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_filename, "wb") as buffer: buffer.write(await file.read())
    try:
        image = Image.open(temp_filename).convert("RGB")
        probs = predict_single(image)
        top_p, top_i = torch.topk(probs, 1)
        
        disease_key = class_names[top_i[0].item()]
        reco_obj = recommendations.get(disease_key, default_recommendation)
        
        return {
            "disease_name": disease_key.replace("_", " "),
            "confidence_score": f"{top_p[0].item():.2f}",
            "recommended_aids": reco_obj.get(lang, reco_obj.get('en'))
        }
    finally:
        os.remove(temp_filename)

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(query: Query):
    question_embedding = rag_model.encode([query.question])
    distances, indices = rag_index.search(np.array(question_embedding, dtype=np.float32), k=1)
    
    best_match = docs[indices[0][0]]
    
    return {
        "answer": best_match['text'],
        "source": best_match['title']
    }
