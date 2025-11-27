"""
Minimal Modal test - just verify imports work and print something.
"""

import modal
import os

app = modal.App("srag-simple-test")

local_dir = os.path.dirname(os.path.abspath(__file__))

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
        "peft>=0.16.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600
)
def simple_test():
    """Just test that everything loads and prints."""
    import sys
    print("=" * 80, flush=True)
    print("SIMPLE TEST STARTING", flush=True)
    print("=" * 80, flush=True)

    # Change to workspace
    import os
    os.chdir("/workspace/srag")
    print(f"Working directory: {os.getcwd()}", flush=True)

    # Check files exist
    print(f"Files in directory: {len(os.listdir('.'))}", flush=True)
    print(f"src/ exists: {os.path.exists('src')}", flush=True)
    print(f"config/ exists: {os.path.exists('config')}", flush=True)

    # Add to path
    sys.path.insert(0, "/workspace/srag/src")
    print(f"Python path updated", flush=True)

    # Try importing
    print("Attempting imports...", flush=True)
    try:
        from sragv.orchestrator import SRAGVOrchestrator
        print("✅ Successfully imported SRAGVOrchestrator", flush=True)
    except Exception as e:
        print(f"❌ Import failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    # Try loading base model
    print("\nAttempting to load base model...", flush=True)
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "Qwen/Qwen2.5-Coder-7B"
        print(f"Loading {model_name}...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✅ Tokenizer loaded", flush=True)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True
        )
        print("✅ Model loaded successfully", flush=True)
        print(f"Model device: {model.device}", flush=True)

    except Exception as e:
        print(f"❌ Model loading failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

    print("\n" + "=" * 80, flush=True)
    print("✅ ALL CHECKS PASSED", flush=True)
    print("=" * 80, flush=True)

    return {"success": True}

@app.local_entrypoint()
def main():
    """Run simple test."""
    print("Launching simple test on Modal...")
    print("This will just verify imports and model loading work")
    print()

    result = simple_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ TEST PASSED - Ready for full GRPO test")
    else:
        print(f"❌ TEST FAILED: {result.get('error', 'Unknown error')}")
    print("=" * 80)
