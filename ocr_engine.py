"""
========================================================
 File: src/ocr_engine.py
 PURPOSE:
   This file wraps the Donut OCR model in a simple,
   reusable function called extract_text().

   WHAT IS DONUT?
   Donut (Document Understanding Transformer) is a deep
   learning model from NAVER Clova AI. Unlike traditional
   OCR (which detects characters one by one), Donut treats
   the whole image as input and generates text directly —
   similar to how a human reads a document.

 HOW IT FITS IN THE PROJECT:
   app.py calls extract_text(image)
   ↓
   ocr_engine.py loads the Donut model (once) from model/
   ↓
   Returns a raw text string (e.g., "Dr. Smith  Paracetamol 500mg BD")
   ↓
   That string gets passed to gemini_analyzer.py for correction + structuring

 KEY DESIGN CHOICE — LAZY LOADING:
   The model (800 MB) is only loaded into RAM the FIRST time
   extract_text() is called, not when the module is imported.
   This keeps app startup fast. The model stays in memory
   for all subsequent calls (no re-loading needed).
========================================================
"""

import os      # Used to read environment variables (OCR_MODEL_PATH) and check paths
import torch   # PyTorch — the deep learning framework that runs the Donut model
               # torch.no_grad() disables gradient tracking during inference (faster)
               # torch.cuda.is_available() checks if a GPU is present

from PIL import Image   # PIL (Pillow) — used for the type annotation and image conversion

from transformers import DonutProcessor, VisionEncoderDecoderModel
# transformers is HuggingFace's library of pre-trained AI models.
# DonutProcessor     — handles image preprocessing (resize, normalize pixels)
#                      AND text tokenization (converts strings to token IDs)
# VisionEncoderDecoderModel — the actual Donut neural network
#   - Encoder: looks at the image and extracts visual features
#   - Decoder: generates text tokens one by one based on those features


# ─────────────────────────────────────────────────────
# MODULE-LEVEL VARIABLES (shared across all calls)
# ─────────────────────────────────────────────────────

# These three variables are declared at module level (outside any function)
# so they persist between calls to extract_text().
# None means "not loaded yet" — the lazy loading pattern.

_processor: DonutProcessor | None = None
# Handles image preprocessing and text tokenization.
# The | None syntax means this variable can be either a DonutProcessor OR None.

_model: VisionEncoderDecoderModel | None = None
# The actual Donut neural network with 200M+ parameters.
# Heavy to load (~800 MB), so we load it once and keep it in memory.

_device: str | None = None
# Either "cuda" (GPU) or "cpu". Determined at load time.
# The model and all input tensors must be on the same device.


# ─────────────────────────────────────────────────────
# DEFAULT MODEL PATH
# ─────────────────────────────────────────────────────

# Build the default path to the model/ folder relative to this file's location.
# __file__ = the full path to this file, e.g., D:\project\src\ocr_engine.py
# os.path.abspath(__file__)           → D:\project\src\ocr_engine.py (absolute)
# os.path.dirname(...)                → D:\project\src\
# os.path.dirname(os.path.dirname(...))→ D:\project\        (up one more level)
# os.path.join(..., "model")          → D:\project\model\   (the model folder)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"
)
# Result: points to the model/ folder at the project root.
# This path calculation works regardless of where you run the script from.


# ─────────────────────────────────────────────────────
# PRIVATE FUNCTION: _load_model()
# ─────────────────────────────────────────────────────

def _load_model(model_path: str | None = None) -> None:
    """
    Load the Donut model and processor into memory.

    This is a PRIVATE function (indicated by the underscore prefix _).
    Private means it should only be called from within this file,
    not imported by other modules.

    It uses the GLOBAL keyword to modify the module-level variables
    _processor, _model, and _device — which are shared across all calls.

    Args:
        model_path: Optional override path. If None, uses the default or
                    the OCR_MODEL_PATH environment variable.
    """
    # Access the module-level variables so we can modify them
    global _processor, _model, _device

    # EARLY RETURN (guard clause):
    # If the model is already loaded (_model is not None), skip everything.
    # This is the core of the lazy loading pattern — load once, reuse forever.
    if _model is not None:
        return

    # Determine which path to use:
    # 1. If a path was passed as an argument, use that.
    # 2. Else if OCR_MODEL_PATH environment variable is set, use that.
    # 3. Otherwise, use the default path (model/ next to project root).
    path = model_path or os.getenv("OCR_MODEL_PATH", _DEFAULT_MODEL_PATH)

    # Validate that the model folder exists.
    # os.path.isdir() returns True if the path exists and is a directory.
    if not os.path.isdir(path):
        # Raise a descriptive error so the user knows exactly what to do.
        # FileNotFoundError is the appropriate Python exception for missing files/folders.
        raise FileNotFoundError(
            f"Donut model not found at '{path}'.\n"
            "Run:  python model_download.py\n"
            "Or set OCR_MODEL_PATH to your local model directory."
        )

    print(f"[OCR] Loading Donut model from {path} …")
    # This message appears in the terminal — useful for debugging and
    # confirming which model folder is being used.

    # Load the processor (handles image preprocessing + tokenization)
    # from_pretrained() reads config files from the folder and rebuilds the object.
    _processor = DonutProcessor.from_pretrained(path)

    # Load the model weights (the neural network parameters)
    # from_pretrained() reads model.safetensors and loads all 200M+ parameters.
    # This is the slow step (~10-30 seconds on first load).
    _model = VisionEncoderDecoderModel.from_pretrained(path)

    # Choose device: GPU if available, otherwise CPU.
    # CUDA is NVIDIA's GPU computing framework. PyTorch uses it automatically.
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Move the model to the chosen device.
    # .to(device) transfers all model parameters to GPU memory (if using CUDA)
    # or keeps them in RAM (if using CPU). Model and input MUST be on same device.
    _model.to(_device)

    # Set the model to evaluation mode.
    # CONCEPT: Models have two modes:
    #   training mode (.train()) → dropout layers active, gradients tracked
    #   eval mode    (.eval())  → dropout disabled, more consistent output
    # We always use eval() for inference (generating predictions, not training).
    _model.eval()

    print(f"[OCR] Model loaded on {_device}")


# ─────────────────────────────────────────────────────
# PUBLIC FUNCTION: extract_text()
# ─────────────────────────────────────────────────────

def extract_text(image: Image.Image, model_path: str | None = None) -> str:
    """
    Run Donut OCR on a PIL image and return the predicted text string.

    This is the ONLY function that other modules should call.
    It handles model loading (lazily), preprocessing, inference, and decoding.

    Args:
        image:      PIL Image — the prescription photo to process.
                    Can be in any color mode; this function converts to RGB.
        model_path: Optional path override for the model folder.
                    If None, uses the default model/ directory.

    Returns:
        A string containing the text extracted from the image.
        Example: "doctor_name: Dr. Smith medications: Paracetamol 500mg BD"

    Typical flow:
        1. _load_model() is called (loads model on first call, skips on subsequent)
        2. Image is preprocessed into a tensor
        3. Model generates token IDs
        4. Token IDs are decoded back to a text string
    """

    # Ensure the model is loaded. On first call this loads it; on subsequent calls
    # _model is already set so _load_model() returns immediately (guard clause).
    _load_model(model_path)

    # ── PREPROCESSING ─────────────────────────────────
    # Convert image to RGB (3 color channels: Red, Green, Blue).
    # This handles edge cases:
    #   - RGBA images (PNG with transparency) → strip the alpha channel
    #   - Grayscale images → expand to 3 channels
    #   - CMYK (print format) → convert to RGB
    image = image.convert("RGB")

    # Run the image through the DonutProcessor.
    # Internally, the processor:
    #   1. Resizes the image to the model's expected size (1280×960 pixels)
    #   2. Normalizes pixel values (subtracts mean, divides by std deviation)
    #   3. Converts the image to a PyTorch tensor (a multi-dimensional array)
    #   4. return_tensors="pt" means return PyTorch tensors (not numpy or TensorFlow)
    # .to(_device) moves the tensor to the same device as the model (GPU or CPU)
    encoding = _processor(images=image, return_tensors="pt").to(_device)

    # ── INFERENCE (GENERATING TEXT) ───────────────────
    # torch.no_grad() is a context manager that disables gradient computation.
    # CONCEPT: During training, PyTorch tracks every mathematical operation
    # to compute gradients (used for backpropagation/learning).
    # During inference (prediction), we don't need gradients — disabling them
    # saves memory (~50% less) and speeds up computation (~2x faster).
    with torch.no_grad():
        generated_ids = _model.generate(
            encoding.pixel_values,   # The preprocessed image tensor (batch of pixels)
                                     # Shape: [1, 3, height, width]

            max_length=512,          # Maximum number of tokens to generate.
                                     # One token ≈ one word or word-piece.
                                     # 512 is enough for a typical prescription.

            num_beams=1,             # Beam search width. num_beams=1 = greedy decoding.
                                     # Greedy: always pick the highest-probability next token.
                                     # Beam search (num_beams>1) considers multiple paths
                                     # but is slower. Greedy is fast enough here.

            early_stopping=True,     # Stop generating when the end-of-sequence token
                                     # (</s_ocr>) is produced, even before max_length.
                                     # Prevents generating padding tokens unnecessarily.

            decoder_start_token_id=_processor.tokenizer.convert_tokens_to_ids("<s_ocr>"),
            # Donut needs a "task prompt" token to know what kind of output to generate.
            # <s_ocr> is the special token we added during fine-tuning that tells
            # the decoder "you are doing OCR, not document classification".
            # convert_tokens_to_ids() converts the string "<s_ocr>" to its integer ID.
        )
    # generated_ids is now a tensor of integer token IDs, e.g., [3, 482, 91, 237, ...]
    # Each integer corresponds to a word/subword in the tokenizer's vocabulary.

    # ── DECODING (TOKEN IDs → TEXT) ───────────────────
    # batch_decode() converts integer token IDs back to human-readable text.
    # skip_special_tokens=True removes <s_ocr>, </s_ocr>, <pad> etc. from output.
    # [0] takes the first (and only) item in the batch — we always process one image.
    # .strip() removes leading/trailing whitespace.
    text = _processor.tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0]

    return text.strip()
    # Example return value:
    # "doctor_name: Dr. F. Gomez clinic_name: Meadowview Health
    #  patient_name: Michael Brown medications: - Ibuprofen 5mg Take twice daily"
