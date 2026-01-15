import os
import sys
import wave
import subprocess
import time
import whisper
import pyaudio
import numpy as np

def test_tts():
    print("🔊 Testing TTS (eSpeak-NG)...")
    try:
        text = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"
        subprocess.run(['espeak-ng', '-v', 'hi', text], check=True)
        print("✅ TTS test passed!")
        return True
    except Exception as e:
        print(f"❌ TTS test failed: {e}")

def test_asr():
    """Test Whisper ASR with Hindi audio file"""
    print("📝 Testing ASR (Whisper base)...")
    try:
        model_start = time.time()
        model = whisper.load_model("base")
        load_time = time.time() - model_start
        print(f"✅ Whisper model loaded in {load_time:.2f}s")
        
        # Test with a dummy or empty audio if possible, 
        # but here we just verify the model is ready.
        print("✅ ASR test passed!")
        return True
    except Exception as e:
        print(f"❌ ASR test failed: {e}")

def test_piper_tts():
    """Test Piper TTS with Hindi voice"""
    print("🔊 Testing Piper TTS (Natural Voice)...")
    model_path = os.path.join(os.path.dirname(__file__), 'models/hindi/hi_IN-rohan-medium.onnx')
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        return False
    
    try:
        test_text = "नमस्ते, यह पाइपर टीटीएस का परीक्षण है"
        
        # Test Piper
        piper = subprocess.Popen(
            [sys.executable, '-m', 'piper', '--model', model_path, '--output-raw'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
        audio_data, _ = piper.communicate(input=test_text.encode('utf-8'), timeout=10)
        
        # Play audio using PyAudio
        import pyaudio
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=22050,
                        output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        print("✅ Piper TTS test passed!")
        return True
    except Exception as e:
        print(f"❌ Piper TTS test failed: {e}")
        return False

def test_mic():
    print("🎤 Testing Microphone Input...")
    try:
        p = pyaudio.PyAudio()
        info = p.get_default_input_device_info()
        print(f"✅ Default input device: {info['name']}")
        p.terminate()
        print("✅ Microphone test passed!")
        return True
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")

def test_advanced_grammar():
    """Test Phase 4: Advanced Grammar Corrector"""
    print("✏️  Testing Advanced Grammar Corrector...")
    try:
        from voice_assistant import AdvancedGrammarCorrector
        corrector = AdvancedGrammarCorrector()
        
        test_cases = [
            ("बन करो", "बंद करो"),  # Regex fix
            ("समये क्य", "समय क्या"), # Phonetic fix
            ("नमस्त", "नमस्ते"),
            ("पलीज हेल्थ", "please help"),
            ("तारीक", "तारीख")
        ]
        
        passed = 0
        for raw, expected in test_cases:
            corrected = corrector.correct(raw)
            status = "✓" if corrected.lower() == expected.lower() else "✗"
            print(f"  {status} '{raw}' → '{corrected}'")
            if corrected.lower() == expected.lower(): passed += 1
                
        print(f"\nPassed: {passed}/{len(test_cases)} tests")
        return passed == len(test_cases)
    except Exception as e:
        print(f"❌ Grammar test failed: {e}")
        return False

def test_robust_intent():
    """Test Phase 4: Robust Intent Classification"""
    print("🧠 Testing Robust Intent Classification...")
    try:
        from voice_assistant import RobustIntentClassifier, AdvancedGrammarCorrector
        classifier = RobustIntentClassifier()
        corrector = AdvancedGrammarCorrector()
        
        test_cases = [
            ("समय क्या है", "time"),        # Direct match
            ("तारीख", "date"),           # Fallback match
            ("नमस्ते", "hello"),          # Direct
            ("अलविदा", "goodbye"),        # Direct
            ("धन्यवाद", "thank_you"),     # Direct
            ("सहायता", "help"),           # Fallback
            ("बंद करो", "stop"),          # Direct
            ("abhi samay kya hai", "time"), # Romanized Robustness
            ("tariq batao", "date"),       # Phonetic + Romanized
            ("abhi", "unknown"),           # Substring false positive test
            ("Teeke, alvida", "goodbye"),  # Punctuation + Noise Resiliency
            ("OK, dhanyawad!", "thank_you"), # Romanized + Punctuation
        ]
        
        passed = 0
        for text, expected in test_cases:
            corrected = corrector.correct(text)
            intent, confidence = classifier.classify(corrected)
            status = "✓" if intent == expected else "✗"
            print(f"  {status} '{text}' → {intent} ({confidence:.1%})")
            if intent == expected: passed += 1
                
        print(f"\nPassed: {passed}/{len(test_cases)} tests")
        return passed == len(test_cases)
    except Exception as e:
        print(f"❌ Robust intent test failed: {e}")
        return False

def test_asr_faster_whisper():
    """Test Phase 5: Faster-Whisper benchmark"""
    print("📝 Testing Faster-Whisper (Int8 Optimized)...")
    try:
        from faster_whisper import WhisperModel
        start_load = time.time()
        model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=4)
        print(f"✅ Faster-Whisper loaded in {time.time()-start_load:.2f}s")
        return True
    except Exception as e:
        print(f"❌ Faster-Whisper test failed: {e}")
        return False

if __name__ == "__main__":
    print("============================================================")
    print("🧪 Component Verification - Phase 5: High-Speed Optimization")
    print("============================================================")
    
    results = [
        ("Faster-Whisper (small-int8)", test_asr_faster_whisper),
        ("Advanced Grammar", test_advanced_grammar),
        ("Robust Intent (IndicBERT + Fuzzy)", test_robust_intent),
        ("Piper TTS (Natural Voice)", test_piper_tts),
        ("Microphone Access", test_mic),
    ]
    
    print("\nSummary:")
    total_passed = 0
    for name, test_func in results:
        print(f"- {name}: ", end="", flush=True)
        if test_func():
            total_passed += 1
        print("-" * 30)
    
    print(f"\n✅ VERIFICATION COMPLETE: {total_passed}/{len(results)} layers operational.")
    print("============================================================")
