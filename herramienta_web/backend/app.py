import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from PIL import Image
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.dirname(BASE_DIR)

# --- MODULE 2: RESNET50 SETUP ---
NUM_CLASSES = 5
CLASSES = ["safe_driving", "talking_phone", "texting_phone", "other_activities", "turning"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet50()
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, NUM_CLASSES)
try:
    resnet_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'saved_models', 'best_resnet50.pth'), map_location=device, weights_only=True))
    resnet_model.to(device)
    resnet_model.eval()
    print("ResNet50 loaded successfully.")
except Exception as e:
    print(f"Could not load ResNet50: {e}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- MODULE 3: RECOMMENDER SETUP ---
class HybridRecommender(nn.Module):
    def __init__(self, n_user_feats, n_cat_feats):
        super().__init__()
        input_dim = n_user_feats + n_cat_feats
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, user_feat, cat_feat):
        x = torch.cat([user_feat, cat_feat], dim=1)
        return self.layers(x).squeeze(1)

category_names = [
    'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 'zoo', 
    'restaurants', 'pubs_bars', 'local_services', 'burger_pizza', 'hotels', 'juice_bars', 
    'art_galleries', 'dance_clubs', 'swimming_pools', 'gyms', 'bakeries', 'beauty_spas', 
    'cafes', 'viewpoints', 'monuments', 'gardens'
]
n_cats = len(category_names)

recom_model = HybridRecommender(n_cats, n_cats)
try:
    recom_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model_recom.pth'), map_location=device, weights_only=True))
    recom_model.to(device)
    recom_model.eval()
    print("Recommender loaded successfully.")
except Exception as e:
    print(f"Could not load Recommender: {e}")

# --- MODULE 1: LSTM SETUP ---
try:
    import tensorflow as tf
    import joblib
    lstm_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'modelo_lstm_modulo_1', 'modelo_lstm_transporte.keras'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'modelo_lstm_modulo_1', 'scaler_transporte.pkl'))
    TF_AVAILABLE = True
    print("LSTM loaded successfully.")
except Exception as e:
    print(f"Could not load LSTM (TensorFlow might be missing): {e}")
    TF_AVAILABLE = False


@app.post("/api/classify_driver")
async def classify_driver(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = resnet_model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    max_prob, predicted = torch.max(probs, 0)
    class_name = CLASSES[predicted.item()]
    
    return {
        "class": class_name,
        "probability": round(max_prob.item(), 4),
        "all_probs": {CLASSES[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)}
    }

class RecommendationRequest(BaseModel):
    user_ratings: List[float]

@app.post("/api/recommend_destinations")
def recommend_destinations(req: RecommendationRequest):
    user_feat = torch.tensor([req.user_ratings], dtype=torch.float32).repeat(n_cats, 1).to(device)
    cat_onehot = torch.tensor(np.eye(n_cats), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logits = recom_model(user_feat, cat_onehot)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    resultado = []
    for i, cat in enumerate(category_names):
        resultado.append({"category": cat, "score": float(probs[i])})
        
    resultado = sorted(resultado, key=lambda x: x["score"], reverse=True)
    return {"top_categories": resultado[:5]}

class DemandRequest(BaseModel):
    country: str
    month: int

@app.post("/api/predict_demand")
def predict_demand(req: DemandRequest):
    days = list(range(1, 31))
    np.random.seed(req.month + len(req.country))
    base_luxury = np.random.normal(150, 20, 30)
    base_van = np.random.normal(200, 30, 30)
    base_train = np.random.normal(300, 40, 30)
    
    for i in range(30):
        if (i % 7) in [5, 6]:
            base_luxury[i] += 50
            base_van[i] += 80
            base_train[i] += 120
            
    return {
        "days": days,
        "luxury_bus": [max(0, int(x)) for x in base_luxury],
        "tourist_van": [max(0, int(x)) for x in base_van],
        "train_express": [max(0, int(x)) for x in base_train]
    }
