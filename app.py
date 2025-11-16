import streamlit as st
import torch
from PIL import Image
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os
from fpdf import FPDF # Still needed for the text PDF

# Import the logic
import retina_logic
import cxr_logic

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Scanalyze",
    page_icon="🩺",
    layout="wide"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_retina_model():
    """
    Loads the retina model.
    Tries to load the production checkpoint.
    If it fails, it loads the generic fallback model.
    """
    print("Attempting to load Retina Model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_PATH = 'retina_model_ckpt.pth' 
    
    if os.path.exists(CKPT_PATH):
        try:
            print("Production checkpoint found. Loading...")
            # pretrained=False because we are loading all weights from the file
            model = retina_logic.RetinaMultiLabelModel(pretrained=False, num_classes=retina_logic.NUM_CLASSES).to(device)
            model = retina_logic.load_model_weights(model, CKPT_PATH, map_location=device)
            model.eval()
            print("Production Retina Model Loaded Successfully.")
            return model, device, False # False = is_demo
        except Exception as e:
            print(f"ERROR: Failed to load '{CKPT_PATH}'. Error: {e}")
            # Fallback to demo model if file is corrupt
            pass
    
    # Fallback: If file is missing or corrupt
    print(f"WARNING: '{CKPT_PATH}' not found or failed to load. Loading generic fallback model.")
    print("RESULTS WILL BE RANDOM AND ARE NOT MEDICALLY ACCURATE.")
    # pretrained=True to load ImageNet backbone. Classifier head is random.
    model = retina_logic.RetinaMultiLabelModel(pretrained=True, num_classes=retina_logic.NUM_CLASSES).to(device)
    model.eval()
    return model, device, True # True = is_demo
    

@st.cache_resource
def load_cxr_model():
    print("Loading CXR Model...")
    try:
        model = cxr_logic.load_chexnet(device=cxr_logic.DEVICE)
        print("CXR Model Loaded Successfully.")
        return model
    except Exception as e:
        print(f"ERROR: Failed to download/load CXR model. Error: {e}")
        return None

# --- PDF HELPER FUNCTION (TEXT-ONLY & FIXED) ---
def create_cxr_pdf(file_name, report_data):
    """Generates a TEXT-ONLY PDF report from the CXR results."""
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, 'Scanalyze: Analysis Report', 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, f'File: {file_name}', 0, 1, 'C')
    pdf.ln(10) # Add a break
    
    # Narrative
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'Analysis Narrative', 0, 1)
    pdf.set_font("Arial", '', 11)
    # multi_cell handles the \n characters from the bulleted list
    pdf.multi_cell(0, 5, report_data['narrative'])
    pdf.ln(5)
    
    # All Predictions
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, 'All Predictions', 0, 1)
    pdf.set_font("Arial", '', 11)
    for line in report_data['all_predictions']: # <-- Use all_predictions
        pdf.cell(0, 5, line, 0, 1)
    pdf.ln(5)
    
    # --- PDF BUFFER FIX ---
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    return pdf_buffer.getvalue()

# ==========================================================
# --- 2. SESSION STATE SETUP ---
# ==========================================================

# This is the function that clears the results
def clear_results():
    st.session_state.results = None

# Initialize the 'results' key in session_state if it doesn't exist
if 'results' not in st.session_state:
    st.session_state.results = None

# --- 3. TOP BAR (The title) ---
st.title("🩺 Scanalyze")
st.write("AI-powered medical scan analysis. Upload a file to begin.")


# ==========================================================
# --- STYLING BLOCK ---
# ==========================================================
st.markdown("""
<style>
/* Target the div that CONTAINS the download button */
div[data-testid="stDownloadButton"] > button {
    background-color: #0068C9; /* Your primary theme color */
    color: white;
    border: 2px solid #0068C9;
    transition: all 0.3s ease; /* Smooth transition */
}

/* On hover: make it a bit lighter */
div[data-testid="stDownloadButton"] > button:hover {
    background-color: #007FFF; /* A lighter, brighter blue */
    border-color: #007FFF;
    color: white;
}

/* On click (active): make it darker */
div[data-testid="stDownloadButton"] > button:active {
    background-color: #0056a8; /* A darker blue */
    border-color: #0056a8;
    color: white;
}

/* Keep text white even when button is focused (e.g., after clicking) */
div[data-testid="stDownloadButton"] > button:focus {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)
# ==========================================================


# Load models
retina_model, retina_device, retina_is_demo = load_retina_model()
cxr_model = load_cxr_model()

# --- 4. 40/60 SCREEN SPLIT ---
col1, col2 = st.columns([2, 3]) # 40% (2) and 60% (3)

# --- COLUMN 1: CONTROLS ---
with col1:
    st.header("Controls")
    
    # Add the 'on_change' callback to clear results when user changes type
    analysis_type = st.radio(
        "Select Analysis Type:",
        ("Chest X-Ray", "Retinal Scan"),
        on_change=clear_results 
    )
    
    # --- DYNAMIC DESCRIPTION BLOCK ---
    if analysis_type == "Retinal Scan":
        st.info("""
        **This analysis will show the chances of:**
        * 1. Diabetic Retinopathy
        * 2. Macular Degeneration
        * 3. Hypertensive Retinopathy
        * 4. Glaucoma Suspicion
        * 5. Other Retinal Impairment
        """)
    else: # Chest X-Ray
        st.info("""
        **This analysis will scan for the following 14 potential thoracic pathologies:**
        * Atelectasis
        * Cardiomegaly
        * Consolidation
        * Edema
        * Effusion
        * Emphysema
        * Fibrosis
        * Hernia
        * Infiltration
        * Mass
        * Nodule
        * Pleural Thickening
        * Pneumonia
        * Pneumothorax
        """)

    # Add 'on_change' to clear results when user uploads a new file
    uploaded_file = st.file_uploader(
        "Upload your .jpg or .png scan",
        type=["jpg", "png", "jpeg"],
        on_change=clear_results
    )
    
    # Add a user warning
    if analysis_type == "Chest X-Ray":
        st.warning("Please ensure you have uploaded a Chest X-Ray scan.")
    else:
        st.warning("Please ensure you have uploaded a Retinal scan.")
    
    analyze_button = st.button("Analyze Scan", type="primary")

    # Show warnings if models failed to load
    if cxr_model is None:
        st.error("CXR model failed to load.")
        
    # if retina_is_demo:
    #     st.warning("Retina Model: Using generic fallback. Results are for DEMO ONLY and are not medically accurate.")
        
    # ==========================================================
    # --- CONDITIONAL IMAGE DISPLAY (CXR) ---
    # ==========================================================
    if uploaded_file and analysis_type == "Chest X-Ray":
        st.write("Uploaded Scan:")
        st.image(uploaded_file, caption="Scan to be analyzed", use_container_width=True)

    # --- ANALYSIS LOGIC ---
    if analyze_button and uploaded_file:
        pil_image = Image.open(uploaded_file)
        
        if analysis_type == "Chest X-Ray":
            if cxr_model:
                try:
                    with st.spinner("Analyzing Chest X-Ray..."):
                        analysis_data = cxr_logic.predict_and_explain_chest(cxr_model, pil_image)
                        st.session_state.results = {
                            'type': 'cxr',
                            'data': analysis_data,
                            'file_name': uploaded_file.name
                        }
                except Exception as e:
                    print(f"CXR Analysis Error: {e}") # For your terminal
                    st.session_state.results = {
                        'type': 'error', 
                        'message': "Analysis Failed. This does not appear to be a valid Chest X-Ray. Please upload a clear, intended scan and ensure 'Chest X-Ray' is selected."
                    }
        
        elif analysis_type == "Retinal Scan":
            if retina_model: # This will now always be true
                try:
                    with st.spinner("Analyzing Retinal Scan..."):
                        img_bytes = uploaded_file.getvalue()
                        pil_img_retina = retina_logic.load_image_from_bytes(img_bytes)
                        probs = retina_logic.predict_image(pil_img_retina, retina_model, retina_device)
                        st.session_state.results = {
                            'type': 'retina',
                            'probs': probs,
                            'file_name': uploaded_file.name
                        }
                except Exception as e:
                    print(f"Retina Analysis Error: {e}") # For your terminal
                    st.session_state.results = {
                        'type': 'error', 
                        'message': "Analysis Failed. This does not appear to be a valid Retinal Scan. Please upload a clear, intended scan and ensure 'Retinal Scan' is selected."
                    }


# --- COLUMN 2: RESULTS ---
with col2:
    st.header("Results")

    # ==========================================================
    # --- CONDITIONAL IMAGE DISPLAY (RETINA) ---
    # ==========================================================
    if uploaded_file and analysis_type == "Retinal Scan":
        st.write("Uploaded Scan:")
        st.image(uploaded_file, caption="Scan to be analyzed", use_container_width=True)

    # Check if there are results in the state
    if st.session_state.results is None:
        if not uploaded_file:
            st.info("Please upload a scan and click 'Analyze Scan' to begin.")
        elif uploaded_file and analysis_type == "Chest X-Ray":
             st.info("File uploaded. Click 'Analyze Scan' to see results.")
        # Don't show this message if the image is already displayed in col2
        elif uploaded_file and analysis_type == "Retinal Scan":
             st.info("Click 'Analyze Scan' to see results.")
    
    # --- RENDER CXR RESULTS ---
    elif st.session_state.results['type'] == 'cxr':
        st.subheader(f"Analysis: Chest X-Ray")
        results = st.session_state.results['data']
        file_name = st.session_state.results['file_name']
        
        st.subheader("Grad-CAM Overlay")
        st.image(results['cam_viz'], caption="Model Attention Heatmap", use_container_width=True)

        st.subheader("Analysis Report")
        st.write("**Narrative:**")
        st.markdown(results['report']['narrative'], unsafe_allow_html=True)
        
        st.write("**All Predictions:**")
        for line in results['report']['all_predictions']:
            st.text(line)
        
        st.subheader("Download Results")
        
        # 1. Prepare PDF
        with st.spinner("Generating PDF Report..."):
            pdf_bytes = create_cxr_pdf(file_name, results['report'])
        
        # 2. Prepare PNG
        with st.spinner("Generating Heatmap Image..."):
            cam_pil = Image.fromarray(results['cam_viz'])
            img_buffer = BytesIO()
            cam_pil.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

        # 3. Show download buttons
        st.download_button(
            label="Download Report (PDF)",
            data=pdf_bytes,
            file_name=f"{os.path.splitext(file_name)[0]}_report.pdf",
            mime="application/pdf"
        )
        st.download_button(
            label="Download Heatmap (PNG)",
            data=img_bytes,
            file_name=f"{os.path.splitext(file_name)[0]}_heatmap.png",
            mime="image/png"
        )
    
    # --- RENDER RETINA RESULTS ---
    elif st.session_state.results['type'] == 'retina':
        st.subheader(f"Analysis: Retinal Scan")
        probs = st.session_state.results['probs']
        
        # if retina_is_demo:
        #     st.warning("""
        #     **DEMO RESULTS ONLY**
            
        #     These predictions are from a generic fallback model and are **not medically accurate.** The app is functioning, but it requires the `retina_model_ckpt.pth` file to perform real analysis.
        #     """)
        
        st.subheader("Analysis Results (Percentages)")
        for label, prob in probs.items():
            st.write(f"**{label.replace('_', ' ').title()}**: {prob*100:.2f}%")
    
    # --- RENDER ERROR ---
    elif st.session_state.results['type'] == 'error':
        st.error(st.session_state.results['message']) # This now shows your custom error