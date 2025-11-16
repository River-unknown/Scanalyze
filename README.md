# 🩺 Scanalyze

A web-based medical imaging tool for analyzing Chest X-Rays and Retinal Scans, built with Streamlit.



---

## 🎯 Purpose & Description

Scanalyze provides a simple, 40/60 split-screen interface for running AI-powered analysis on medical images. It's designed to be a "pass-through" UI: a user uploads a scan and receives an immediate, downloadable report.

The app currently supports two analysis types:

* **Chest X-Ray:** Scans for 14 potential thoracic pathologies (e.g., Pneumonia, Cardiomegaly) and generates a detailed report, including a Grad-CAM (heatmap) visualization.
* **Retinal Scan:** Analyzes retinal fundoscopy images for 5 potential conditions, including diabetic retinopathy and glaucoma.

## ✨ Features

* **Split-Screen UI:** Clean layout with controls on the left and results on the right.
* **Dynamic Descriptions:** The UI provides context for each analysis type.
* **Persistent State:** Analysis results "stick" on the screen until a new scan is uploaded.
* **Chest X-Ray Report:**
    * Bulleted-list narrative generated from findings.
    * Full list of all 14 pathology predictions.
    * Grad-CAM heatmap visualization for model explainability.
* **Multi-Format Download:**
    * Download text-only report as a PDF.
    * Download heatmap as a separate PNG.
* **Custom Theming:** Professional blue-and-white medical color scheme.
* **Robust Error Handling:** Provides clear, user-friendly warnings for "wrong image" uploads.

---

## 🛠️ Architecture & Tech Stack

This project is a single-page web application built entirely in Python, leveraging the **Streamlit** framework. It does not require a separate backend or database.

* **Framework:** **Streamlit**
* **AI/ML:** **PyTorch**, **timm**
* **Visualization:** **Grad-CAM**, **Pillow**, **OpenCV**
* **Report Generation:** **fpdf2**
* **Core Logic:** The analysis logic is refactored from two core Jupyter Notebooks into `cxr_logic.py` and `retina_logic.py`.

---

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/River-unknown/Scanalyze.git](https://github.com/River-unknown/Scanalyze.git)
    cd Scanalyze
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## ⚠️ Disclaimer

This is a demo application. The AI models, especially the Retinal Scan fallback, are **not medically validated** and should **not** be used for real-world clinical decision-making.

* The **Retinal Scan** feature uses a generic model trained on ImageNet (cats, dogs, etc.) as a placeholder. Its results are for **demonstration only** and are **not medically accurate.**
* The **Chest X-Ray** model downloads pre-trained weights, but its output is for informational and demo purposes only.

---

## Acknowledgements

* Credit to **Maharshi Mazumdar** for providing the core logic notebooks (`retina_inference_notebook.ipynb` and `CXRproject.ipynb`) that form the backbone of this application's analysis features.
