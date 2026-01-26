#!/usr/bin/env python3
"""
IndicBERT to ONNX INT8 Conversion Script
Optimized for Radxa Cubie A7A (Cortex-A76 cores)
"""

import os
import sys
import shutil
import time
import gc

def check_dependencies():
    """Verify required packages are installed"""
    try:
        import optimum
        import onnxruntime
        import torch
        from transformers import AutoTokenizer
        print("✓ Dependencies verified")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e.name}")
        print(f"Please run: pip install optimum[onnxruntime] onnxruntime torch transformers")
        return False

def convert_model():
    if not check_dependencies():
        return

    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, "hindi_intent_model_final")
    export_path = os.path.join(script_dir, "hindi_intent_model_onnx")
    quantized_path = os.path.join(script_dir, "hindi_intent_model_onnx_int8")

    if not os.path.exists(source_path):
        print(f"❌ Source model not found at {source_path}")
        return

    print(f"🚀 Starting conversion: {source_path} → ONNX")

    # 1. Export to ONNX
    print("📦 Exporting to ONNX...")
    try:
        model = ORTModelForSequenceClassification.from_pretrained(
            source_path,
            export=True,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(source_path)
        
        model.save_pretrained(export_path)
        tokenizer.save_pretrained(export_path)
        print("✓ ONNX export complete")
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return

    # 2. INT8 Quantization
    print("⚙️  Applying INT8 Quantization...")
    try:
        quantizer = ORTQuantizer.from_pretrained(export_path)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        
        quantizer.quantize(
            save_dir=quantized_path,
            quantization_config=qconfig,
        )
        
        # Copy label_map.json
        shutil.copy(
            os.path.join(source_path, "label_map.json"),
            os.path.join(quantized_path, "label_map.json")
        )
        
        # Cleanup intermediary folder
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
            
        print("✓ INT8 Quantization complete")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")
        return

    # 3. Size Comparison
    def get_dir_size(path):
        total = 0
        for entry in os.scandir(path):
            if entry.is_file():
                total += entry.stat().st_size
        return total / (1024 * 1024)

    original_size = get_dir_size(source_path)
    final_size = get_dir_size(quantized_path)
    reduction = ((original_size - final_size) / original_size) * 100

    print("\n📊 Conversion Statistics:")
    print(f"   - Original Size: {original_size:.2f} MB")
    print(f"   - Optimized Size: {final_size:.2f} MB")
    print(f"   - Reduction: {reduction:.1f}%")

    print(f"\n✅ Optimized model saved to: {quantized_path}")
    
    # Garbage collection
    model = None
    tokenizer = None
    gc.collect()

if __name__ == "__main__":
    # Check if inside venv
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected.")
        print("   Recommend running: source venv/bin/activate")
    
    convert_model()
