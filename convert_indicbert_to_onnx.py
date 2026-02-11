#!/usr/bin/env python3
"""
Fixed ONNX conversion script for IndicBERT
Handles optimum library version compatibility
"""

import os
import shutil
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("Converting IndicBERT to ONNX (Version-Safe)")
print("=" * 60)

# Method 1: Try optimum (with error handling)
try:
    print("\n[Method 1] Trying optimum export...")
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer
    
    pytorch_path = "hindi_intent_model_final"
    onnx_path = "hindi_intent_model_onnx_int8"
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
    
    # Export with compatibility fix
    try:
        # Try new API (optimum >= 1.13)
        model = ORTModelForSequenceClassification.from_pretrained(
            pytorch_path,
            export=True,
            provider="CPUExecutionProvider"
        )
    except TypeError:
        # Fallback for older optimum versions
        print("   Trying fallback export method...")
        from optimum.exporters.onnx import main_export
        from pathlib import Path
        
        main_export(
            pytorch_path,
            output=Path(onnx_path),
            task="text-classification",
            opset=14,
            device="cpu"
        )
        
        model = ORTModelForSequenceClassification.from_pretrained(onnx_path)
    
    # Save model and tokenizer
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)
    
    # Copy label map
    shutil.copy(
        os.path.join(pytorch_path, "label_map.json"),
        os.path.join(onnx_path, "label_map.json")
    )
    
    print(f"✓ Conversion successful!")
    print(f"✓ Model saved to: {onnx_path}")
    
    # Check file size
    import glob
    onnx_files = glob.glob(f"{onnx_path}/*.onnx")
    if onnx_files:
        size_mb = os.path.getsize(onnx_files[0]) / (1024 * 1024)
        print(f"✓ ONNX model size: {size_mb:.1f} MB")
    
    print("\n" + "=" * 60)
    print("✅ CONVERSION COMPLETE")
    print("=" * 60)
    print("Restart voice_assistant.py to use optimized model")
    
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    print("\n[Method 2] Using direct transformers export...")
    
    # Method 2: Direct export using transformers
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        pytorch_path = "hindi_intent_model_final"
        onnx_path = "hindi_intent_model_onnx_int8"
        
        os.makedirs(onnx_path, exist_ok=True)
        
        # Load PyTorch model
        print("   Loading PyTorch model...")
        model = AutoModelForSequenceClassification.from_pretrained(pytorch_path)
        tokenizer = AutoTokenizer.from_pretrained(pytorch_path)
        model.eval()
        
        # Create dummy input
        print("   Creating dummy input...")
        dummy_input = tokenizer("नमस्ते", return_tensors="pt", padding=True, truncation=True, max_length=64)
        
        # Export to ONNX
        print("   Exporting to ONNX...")
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            os.path.join(onnx_path, "model.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"}
            },
            opset_version=14,
            do_constant_folding=True
        )
        
        # Save tokenizer and config
        tokenizer.save_pretrained(onnx_path)
        model.config.save_pretrained(onnx_path)
        
        # Copy label map
        shutil.copy(
            os.path.join(pytorch_path, "label_map.json"),
            os.path.join(onnx_path, "label_map.json")
        )
        
        print(f"✓ Direct export successful!")
        print(f"✓ Model saved to: {onnx_path}")
        
        # Check file size
        onnx_file = os.path.join(onnx_path, "model.onnx")
        if os.path.exists(onnx_file):
            size_mb = os.path.getsize(onnx_file) / (1024 * 1024)
            print(f"✓ ONNX model size: {size_mb:.1f} MB")
        
        print("\n" + "=" * 60)
        print("✅ CONVERSION COMPLETE (FP32)")
        print("=" * 60)
        print("Note: No INT8 quantization (still faster than PyTorch)")
        print("Restart voice_assistant.py to use optimized model")
        
    except Exception as e2:
        print(f"❌ Method 2 also failed: {e2}")
        print("\n⚠️  ONNX conversion not possible with current setup")
        print("Your assistant will use PyTorch model (still works fine!)")
