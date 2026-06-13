"""
Enhanced base player with better download progress tracking.
"""

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

def download_model_with_progress(model_name: str, cache_dir: str = None):
    """Download model with visible progress."""
    logger.info(f"Starting download of {model_name}")
    
    try:
        # Enable progress bars
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        
        # Download with progress
        logger.info(f"Downloading {model_name} - this may take 5-10 minutes...")
        
        # Use snapshot_download for better progress tracking
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False
        )
        
        logger.info(f"✓ Model {model_name} downloaded successfully to {local_path}")
        return local_path
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}")
        raise

# Update the original test to use this if needed
def quick_model_test():
    """Quick test with download progress."""
    try:
        logger.info("Testing model download with progress tracking...")
        download_model_with_progress("gpt2")  # Small test model
        logger.info("✓ Download system working")
        return True
    except Exception as e:
        logger.error(f"Download test failed: {e}")
        return False

if __name__ == "__main__":
    quick_model_test()