import os
import io
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
from torchvision import transforms
from typing import Dict, Optional

# --- From Cell 3 ---
LABELS = [
    'diabetic_retinopathy',
    'macular_degeneration',
    'hypertensive_retinopathy',
    'glaucoma_suspicion',
    'other_retinal_impairment'
]
NUM_CLASSES = len(LABELS)
IMG_SIZE = 384

# --- From Cell 4 (Model Definition) ---
class RetinaMultiLabelModel(nn.Module):
    # This is the line that was incorrect. It's 'b0' (zero), not 'bo' (letter).
    def __init__(self, backbone_name='tf_efficientnet_b0_ns', pretrained=True, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits

def load_model_weights(model: nn.Module, ckpt_path: str, map_location: Optional[torch.device] = None) -> nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}.")
    
    ckpt = torch.load(ckpt_path, map_location=map_location)
    
    # Handle checkpoints saved in different ways
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
        
    model.load_state_dict(state)
    return model

# --- From Cell 5 (Preprocessing & Prediction) ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image_from_bytes(b: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(b)).convert('RGB')
    return img

def predict_image(img: Image.Image, model: torch.nn.Module, device: str) -> Dict[str, float]:
    model.eval()
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}