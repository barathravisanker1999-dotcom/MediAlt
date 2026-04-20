# MediAlt
AI-powered medical prescription analyzer using TrOCR + Gemini
# 💊 MediAlt — Medical Prescription Analyzer & Brand Suggester

<div align="center">

![MediAlt Banner](icon.png)

**AI-powered prescription reading, drug extraction, and affordable brand alternatives**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-TrOCR-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/BK1999/rx-trocr-prescription)
[![Gemini](https://img.shields.io/badge/Google-Gemini%201.5%20Flash-4285F4?logo=google&logoColor=white)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Model Details](#-model-details)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Evaluation Results](#-evaluation-results)
- [Disclaimer](#-disclaimer)

---

## 🔍 Overview

**MediAlt** is an AI-powered web application that reads handwritten or printed medical prescriptions, extracts structured drug information, and suggests affordable alternative brand names — all in seconds.

It combines two AI models in a pipeline:

1. **TrOCR** — A fine-tuned Microsoft TrOCR model hosted on HuggingFace that reads handwriting from prescription images and converts it to raw text.
2. **Google Gemini 1.5 Flash** — A multimodal LLM that cross-verifies the OCR text against the original image, corrects errors, extracts structured data (patient info, medicines, dosages), and suggests region-aware affordable alternatives.

> **Why two models?** TrOCR is fast and local. Gemini is accurate and multimodal. Together they are more reliable than either model alone.

---

## ✨ Features

- 📤 **Prescription Upload** — Drag & drop JPG, JPEG, or PNG prescription images
- 🔍 **Hybrid OCR** — Local TrOCR reads handwriting; Gemini cross-verifies using the image
- 💊 **Structured Extraction** — Patient name, doctor, clinic, diagnosis, medicines, dosages, frequency, duration
- 💰 **Brand Alternatives** — 3 affordable generic/alternative brands per medicine for India, USA, UK, or Global markets
- 💬 **Interactive Q&A** — Multi-turn chat to ask about drug interactions, side effects, dosage queries
- 🌍 **Region Aware** — Brand suggestions tailored to your selected country/region
- ⚠️ **Warnings Panel** — Highlights special instructions, allergies, or cautions from the prescription
- 🔒 **Local Model Option** — TrOCR runs locally; no prescription image is sent to OCR servers

---

## 🏗️ System Architecture

```
Prescription Image (JPG/PNG)
         │
         ▼
┌─────────────────────┐
│   TrOCR Engine      │  ← Fine-tuned microsoft/trocr-large-handwritten
│   (Local Model)     │    Hosted: HuggingFace BK1999/rx-trocr-prescription
└────────┬────────────┘
         │  Raw OCR Text
         ▼
┌─────────────────────────────────────┐
│   Google Gemini 1.5 Flash           │  ← Receives: Image + OCR Text
│   (Multimodal Cross-Verification)   │    Corrects OCR errors via image
│                                     │    Returns: Structured JSON
└────────┬────────────────────────────┘
         │
         ├── Patient / Doctor Info
         ├── Medicine Cards (name, strength, frequency, duration)
         ├── Generic Name + Brand Alternatives
         └── Warnings / Special Instructions
                  │
                  ▼
         ┌────────────────┐
         │ Streamlit UI   │  ← Dashboard + Interactive Q&A Chat
         └────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **OCR Model** | `microsoft/trocr-large-handwritten` (fine-tuned) |
| **AI Analysis** | Google Gemini 1.5 Flash API |
| **Frontend** | Streamlit + Custom CSS |
| **ML Framework** | PyTorch + HuggingFace Transformers |
| **Training Framework** | PyTorch Lightning |
| **Image Processing** | Pillow (PIL), Albumentations |
| **Model Hosting** | HuggingFace Hub |
| **Fuzzy Matching** | RapidFuzz |
| **Environment** | python-dotenv |

---

## 🧠 Model Details

### Fine-Tuned TrOCR

| Detail | Value |
|--------|-------|
| **Base Model** | `microsoft/trocr-large-handwritten` |
| **Architecture** | Vision Encoder (ViT) + Text Decoder (GPT-2 style) |
| **Dataset** | RxHandBD — 5,578 prescription word images |
| **Training Platform** | Kaggle GPU T4 |
| **Optimizer** | AdamW (lr=3e-5, weight_decay=0.01) |
| **LR Scheduler** | Linear warmup → decay (warmup ratio = 0.1) |
| **Epochs** | 20 |
| **Batch Size** | 2 (effective: 8 via gradient accumulation ×4) |
| **Precision** | FP16 Mixed (AMP) |
| **Best CER** | ~0.11 |
| **Word Accuracy** | ~78% |
| **HuggingFace Repo** | [BK1999/rx-trocr-prescription](https://huggingface.co/BK1999/rx-trocr-prescription) |

### Augmentation Schedule (Gradual)

| Epochs | Stage | Augmentations Applied |
|--------|-------|-----------------------|
| 0–4 | Light | Rotate ±3°, Brightness/Contrast ±10% |
| 5–9 | Medium | + Gaussian Blur, Gaussian Noise, Perspective Warp |
| 10+ | Heavy | + Elastic Transform, Grid Distortion, Random Shadows, Coarse Dropout |
| Validation | None | No augmentation — clean images only |

---

## 📁 Project Structure

```
medical-prescription-ocr/
│
├── app.py                    # Main Streamlit application
├── model_download.py         # Downloads TrOCR model from HuggingFace Hub
├── icon.png                  # App icon
├── run.bat                   # One-click Windows launcher
├── requirements.txt          # Python dependencies
├── .env                      # API keys (NOT committed — see .env.example)
├── .env.example              # Template for environment variables
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
│
├── src/
│   ├── __init__.py
│   ├── ocr_engine.py         # TrOCR model wrapper (lazy loading)
│   └── gemini_analyzer.py    # Gemini API integration + prompt engineering
│
└── model/                    # Downloaded TrOCR model weights (NOT in repo)
    ├── config.json           # Model configuration
    ├── model.safetensors     # ~800MB model weights
    └── tokenizer files...
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- Git
- A Google Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

### 1. Clone the Repository

```bash
git clone https://github.com/BK1999/medical-prescription-ocr.git
cd medical-prescription-ocr
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the TrOCR Model (~800MB)

```bash
python model_download.py
```

This downloads the fine-tuned model from HuggingFace Hub into the `model/` folder. Only needs to be run **once**.

---

## 🔑 Configuration

Create a `.env` file in the project root by copying the example:

```bash
# Windows
copy .env.example .env

# Mac / Linux
cp .env.example .env
```

Then open `.env` and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

> **Alternative:** You can also paste the API key directly into the sidebar when the app is running — no `.env` file required.

---

## 🚀 Usage

### Option A — One Click (Windows only)

Double-click `run.bat`

### Option B — Terminal

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

### How to Use

| Step | Action |
|------|--------|
| 1 | Open the sidebar → enter your Gemini API key |
| 2 | Select your region (India / USA / UK / Global) |
| 3 | Go to **Upload & Analyze** tab → upload a prescription image |
| 4 | Click **🔍 Analyze Prescription** |
| 5 | View extracted medicines, dosages, and alternative brands |
| 6 | Go to **Ask a Question** tab → chat about the prescription |

---

## 📊 Evaluation Results

Evaluated on the held-out test split of the RxHandBD dataset:

| Metric | Value |
|--------|-------|
| Character Error Rate (CER) | **0.11** |
| Character Accuracy | **~89%** |
| Word Accuracy | **~78%** |

---

## ⚠️ Disclaimer

> **This tool is for research and educational purposes only.**
>
> MediAlt is **not validated for clinical use** and should **not** be used to make real medical decisions. Always consult a licensed doctor or pharmacist before taking, changing, or stopping any medication.
>
> The alternative brand suggestions are generated by AI and may not reflect current market availability or medical suitability for individual patients.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Built with ❤️ using TrOCR + Gemini + Streamlit
</div>
