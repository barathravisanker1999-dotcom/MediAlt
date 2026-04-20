"""
========================================================
 File: model_download.py
 PURPOSE:
   Downloads the fine-tuned Donut OCR model from HuggingFace Hub
   into a local model/ folder. Run this ONCE before launching the app.

   WHY DO WE NEED THIS?
   The trained Donut model is ~814 MB — too large to include in
   the Git repository (GitHub has a 100 MB file size limit).
   Instead, we host it on HuggingFace Hub (free model hosting)
   and download it on demand.

 RUN:  python model_download.py
 OUTPUT: model/ folder with config.json, model.safetensors, tokenizer files
========================================================
"""

#!/usr/bin/env python3
# #!/usr/bin/env python3 is called a "shebang" line.
# On Linux/Mac, it tells the OS to run this file with Python 3.
# On Windows, it is ignored but harmless.

import os    # os.makedirs() creates directories if they don't exist
import sys   # sys.exit(1) terminates the program with an error code

from huggingface_hub import snapshot_download
# HuggingFace Hub is like GitHub but specifically for AI models and datasets.
# snapshot_download() downloads ALL files from a HuggingFace repository
# to a local directory. It handles resuming interrupted downloads.


def download_model(model_dir: str = "model") -> None:
    """
    Download all model files from HuggingFace Hub.

    Args:
        model_dir: Local folder name where model files will be saved.
                   Default is "model" (creates ./model/ in project root).
    """

    # The HuggingFace repository ID in the format "username/repo-name".
    # This points to the model you trained and pushed to HuggingFace Hub.
    # Change "BK1999" to your actual HuggingFace username if different.
    repo_id = "BK1999/medical-prescription-ocr"

    print(f"Downloading model from {repo_id}...")
    print(f"This will download ~800MB of model files to '{model_dir}/'")

    try:
        # Create the model directory if it doesn't already exist.
        # exist_ok=True means no error if the folder already exists.
        # Without this, os.makedirs() would raise an error for existing folders.
        os.makedirs(model_dir, exist_ok=True)

        # Download all model files from HuggingFace.
        # snapshot_download() is smarter than a simple file download:
        #   - Downloads multiple files in parallel (faster)
        #   - Skips files that are already downloaded (resume support)
        #   - Verifies file integrity with checksums
        snapshot_download(
            repo_id=repo_id,           # Which HuggingFace repo to download from
            local_dir=model_dir,       # Where to save the files locally (./model/)
            local_dir_use_symlinks=False,
            # local_dir_use_symlinks=False means copy files directly instead of
            # creating symbolic links (shortcuts). This is safer on Windows
            # where symlinks require special permissions.
        )

        print(f"\n✅ Model downloaded successfully to {model_dir}/")
        print("You can now run: python app.py")

    except Exception as e:
        # Catch any error (network issues, wrong repo name, disk full, etc.)
        print(f"\n❌ Error downloading model: {e}")
        print("\nPlease check your internet connection and try again.")
        sys.exit(1)
        # sys.exit(1) terminates the program immediately.
        # Exit code 1 conventionally means "an error occurred".
        # Exit code 0 means "success". This helps scripts detect failures.


# ─────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    # This block only runs when the file is executed directly:
    #   python model_download.py  ← runs this block
    #
    # It does NOT run when the file is imported by another module:
    #   import model_download     ← skips this block
    #
    # This pattern is the standard Python way to make a file work
    # both as a runnable script and as an importable module.
    download_model()
