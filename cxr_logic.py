import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import json

# --- Config (No Change) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 224
LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
    "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]
LABEL_PHRASES = {
    "Atelectasis": "Areas of lung collapse/volume loss, suggest atelectasis.",
    "Cardiomegaly": "Enlarged cardiac silhouette, suggest cardiomegaly.",
    "Consolidation": "Dense airspace opacity, could represent consolidation (infection/aspiration).",
    "Edema": "Interstitial or alveolar edema pattern, consider cardiogenic/noncardiogenic edema.",
    "Effusion": "Pleural effusion (fluid along lung base or costophrenic sulcus).",
    "Emphysema": "Hyperlucent lungs with possible hyperinflation, suggest emphysema/COPD.",
    "Fibrosis": "Reticular/linear scarring pattern, suggest chronic fibrosis.",
    "Hernia": "Possible diaphragmatic hernia or subdiaphragmatic abnormality.",
    "Infiltration": "Infiltrative/patchy lung opacities, nonspecific infiltrates.",
    "Mass": "Focal mass-like opacity, consider neoplasm or granuloma.",
    "Nodule": "Small rounded opacity (nodule), consider follow-up imaging as appropriate.",
    "Pleural_Thickening": "Pleural thickening/scar along pleural surface.",
    "Pneumonia": "Findings suspicious for pneumonia (airspace consolidation/infiltrate).",
    "Pneumothorax": "Air in pleural space consistent with pneumothorax."
}
PROB_THRESHOLD = 0.30

# --- Preprocessing & Model Loader (No Change) ---
preprocess_common = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image_tensor(img_pil):
    if img_pil.mode != 'RGB':
        img = img_pil.convert('RGB')
    else:
        img = img_pil
    return preprocess_common(img).unsqueeze(0).to(DEVICE)

def load_chexnet(device=DEVICE, num_classes=14, try_url=True):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.to(device)
    
    if try_url:
        try:
            url = "https://github.com/arnoweng/CheXNet/raw/master/models/densenet121.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location=device)
            model.load_state_dict(state_dict)
            print("Loaded CheXNet weights from URL.")
        except Exception as e:
            print(f"Could not fetch CheXNet weights (will use ImageNet init). Error: {e}")
            try:
                backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
                model.features = backbone.features
                print("Loaded ImageNet backbone for DenseNet121 (classifier random).")
            except Exception as e:
                print(f"Unable to fetch ImageNet weights. Using randomly initialized DenseNet121. {e}")
    
    model.eval()
    return model

# --- UPDATED Report Generator ---
def generate_report_text(probs, labels=LABELS, phrases=LABEL_PHRASES, threshold=PROB_THRESHOLD):
    """
    Generates report text. NOW INCLUDES ALL 14 LABELS
    and formats the narrative as a bulleted list.
    """
    lines = []
    findings = []
    
    # Iterate over all 14 labels and their probabilities
    for i in range(len(labels)):
        p = float(probs[i])
        lines.append(f"{labels[i]}: {p:.2f}") # Add all 14 to the list
        if p > threshold:
            findings.append(labels[i])
    
    if len(findings) == 0:
        narrative = "No acute cardiopulmonary abnormality detected above reporting threshold."
    else:
        # Sort findings by probability (descending)
        sorted_findings = sorted(findings, key=lambda f: probs[labels.index(f)], reverse=True)
        
        # --- THIS IS THE FIX for the narrative ---
        # Create a bulleted list
        phrase_list = [f"* {phrases.get(f, f'{f} present.')}" for f in sorted_findings]
        narrative = "\n".join(phrase_list) # Join with newlines
    
    report = {
        "all_predictions": lines, # Use this key
        "narrative": narrative,
        "selected_findings": findings
    }
    return report

# --- Main Prediction Function (No Change) ---
def predict_and_explain_chest(model, img_pil):
    inp = preprocess_image_tensor(img_pil)
    model.eval()
    with torch.no_grad():
        out = model(inp)
        probs = torch.sigmoid(out).cpu().numpy()[0]

    if probs.size != len(LABELS):
        probs = np.resize(probs, (len(LABELS),))

    # Grad-CAM
    try:
        target_layers = [model.features.denseblock4]
    except Exception:
        target_layers = [list(model.children())[-1]]
        
    vis_label = int(np.argmax(probs))
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(vis_label)]
    grayscale_cam = cam(input_tensor=inp, targets=targets)[0]
    
    rgb_img = np.array(img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE))) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Calls the updated report function
    report = generate_report_text(probs) 
    
    result = {
        'probs': probs.tolist(),
        'cam_viz': visualization,
        'report': report,
        'original_image': img_pil
    }
    return result