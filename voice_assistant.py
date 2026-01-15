#!/usr/bin/env python3
"""
Real-time Hindi Voice Assistant with Faster-Whisper
Optimized for Raspberry Pi 5 (4GB RAM)
Bharat AI-SoC Challenge Submission

Architecture:
- Layer 1: Faster-Whisper (High-speed, Ind8-quantized Hindi ASR)
- Layer 2: Advanced phonetic grammar correction (Regex + RapidFuzz)
- Layer 3: Robust intent classification with IndicBERT + Fuzzy Fallback
"""

import os
import sys
import time
import wave
import json
import torch
import re
import pyaudio
import numpy as np
import collections
import subprocess
import gc
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# LAYER 2: ADVANCED GRAMMAR CORRECTION
# ============================================================

class AdvancedGrammarCorrector:
    """Layer 2: Phonetic grammar correction with fuzzy matching"""
    
    def __init__(self):
        # Core vocabulary by intent category
        self.core_vocabulary = {
            'stop': ['बंद', 'बन्द', 'स्टॉप', 'स्टप', 'stop', 'रुको', 'रूको', 'रुक'],
            'command_stop': ['करो', 'करदो', 'कर', 'कर do', 'हो', 'हो जाओ'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे'],
            'time_query': ['क्या', 'कितने', 'कितना', 'बताओ', 'बतओ', 'what', 'कैसा'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'दिन', 'आज'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम'],
            'goodbye': ['अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'चलता', 'जाता'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत'],
        }
        
        # Critical error patterns (regex)
        self.error_patterns = [
            (r'\bबन\b', 'बंद'),
            (r'बन करो', 'बंद करो'),
            (r'वन करो', 'बंद करो'),
            (r'बंदकरो', 'बंद करो'),
            (r'समये', 'समय'),
            (r'बतओ', 'बताओ'),
            (r'क्य\b', 'क्या'),
            (r'कितन\b', 'कितने'),
            (r'मुझ\b', 'मुझे'),
            (r'तिथ\b', 'तिथि'),
            (r'तारिख', 'तारीख'),
            (r'करदो', 'कर दो'),
            (r'होजाओ', 'हो जाओ'),
            (r'कोन\b', 'कौन'),
            (r'काउन', 'कौन'),
            (r'नमसते', 'नमस्ते'),
            (r'नमस्त', 'नमस्ते'),
            (r'पलीज', 'please'),
            (r'हेल्प', 'help'),
            (r'हेल्थ', 'help'),
            (r'\bwat\b', 'what'),
            (r'\btym\b', 'time'),
            (r'\bplz\b', 'please'),
            (r'\bstap\b', 'stop'),
        ]
        
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
            self.use_fuzzy = True
            self.fuzzy_threshold = 75
        except ImportError:
            self.use_fuzzy = False

    def correct(self, text):
        if not text: return ""
        original_text = text
        
        # Pass 1: Regex patterns
        corrected = text
        for pattern, replacement in self.error_patterns:
            corrected = re.sub(pattern, replacement, corrected)
        
        # Pass 2: Word-level fuzzy correction
        words = corrected.split()
        corrected_words = []
        for word in words:
            corrected_word = self._correct_word(word)
            corrected_words.append(corrected_word)
        
        final_text = ' '.join(corrected_words)
        if final_text != original_text:
            print(f"✏️  Corrected: '{original_text}' → '{final_text}'")
        return final_text
    
    def _correct_word(self, word):
        if not self.use_fuzzy or len(word) < 2: return word
        word_lower = word.lower()
        
        for category, vocab_list in self.core_vocabulary.items():
            for vocab_word in vocab_list:
                if word_lower == vocab_word.lower():
                    return word
                similarity = self.fuzz.ratio(word_lower, vocab_word.lower())
                if similarity >= self.fuzzy_threshold:
                    return vocab_word
        return word

# ============================================================
# LAYER 3: ROBUST INTENT CLASSIFICATION
# ============================================================

class RobustIntentClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'hindi_intent_model_final')
            
        print(f"⚙️  Initializing Robust Intent Classifier from {model_path}...")
        
        # Load IndicBERT
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            self.id2label = json.load(f)['id2label']
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32 # Optimized for CPU
        )
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Fuzzy fallback patterns
        self.fallback_patterns = {
            'stop': ['बंद', 'स्टॉप', 'stop', 'रुको', 'रूको', 'exit', 'quit', 'close', 'बन्द', 'समाप्त', 'खत्म'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'आज', 'दिन', 'कैलेंडर'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम'],
            'goodbye': ['अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'चलता', 'जाता'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'शुक्रीया'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत'],
        }

    def classify(self, text):
        if not text.strip(): return "unknown", 0.0
        
        # Stage 1: IndicBERT
        inputs = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            
        intent = self.id2label.get(str(idx.item()), "unknown")
        confidence = conf.item()
        
        # High confidence? Trust IndicBERT
        if confidence >= 0.70:
            return intent, confidence
            
        # Low confidence? Try fuzzy fallback
        print(f"⚠️  Low confidence ({confidence:.1%}), trying fuzzy fallback...")
        fallback_intent = self._fuzzy_fallback(text)
        if fallback_intent:
            print(f"✓ Fuzzy fallback matched: {fallback_intent}")
            return fallback_intent, 0.85
            
        # Medium confidence (50-70%)? Use IndicBERT result
        if confidence >= 0.50:
            return intent, confidence
            
        return "unknown", confidence

    def _fuzzy_fallback(self, text):
        from rapidfuzz import fuzz
        text_lower = text.lower()
        
        scores = {}
        for intent, keywords in self.fallback_patterns.items():
            max_score = 0
            for keyword in keywords:
                score = fuzz.partial_ratio(text_lower, keyword.lower())
                max_score = max(max_score, score)
            scores[intent] = max_score
            
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] >= 75:
            return best_intent
        return None

# ============================================================
# MAIN ASSISTANT CLASS
# ============================================================

class RealtimeVoiceAssistant:
    def __init__(self):
        print("=" * 60)
        print("Initializing Real-time Hindi Voice Assistant")
        print("High-Speed Optimization (Pi 5)")
        print("=" * 60)
        
        self.RATE = 16000
        self.CHUNK = 480 
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        self.vad = webrtcvad.Vad(2) 
        self.silence_threshold = 1.0 
        self.min_speech_duration = 0.5 
        self.max_recording_duration = 10.0
        
        self.audio = pyaudio.PyAudio()
        
        # Layer 1: ASR Loading (Faster-Whisper with Fallback)
        try:
            from faster_whisper import WhisperModel
            print("\n[Layer 1] Loading Faster-Whisper (Base, Int8 quantized)...")
            self.asr_model = WhisperModel(
                "base",                     # Model size (Base for RPi 5 speed)
                device="cpu",               # CPU inference
                compute_type="int8",        # 8-bit quantization (Speed boost)
                cpu_threads=4,              # Pi 5 optimization
                num_workers=1               # Single worker for stability
            )
            self.use_faster_whisper = True
            print("✓ Faster-Whisper loaded (optimized for Pi 5)")
        except Exception as e:
            print(f"⚠️  Faster-Whisper failed: {e}")
            print("   Falling back to standard Whisper (will be slower)")
            import whisper
            self.asr_standard = whisper.load_model("base")
            self.use_faster_whisper = False
            
        self.TEMP_WAV = "temp_input.wav"
        
        # Layer 2: Advanced Grammar Corrector
        print("\n[Layer 2] Initializing Advanced Grammar Corrector...")
        self.corrector = AdvancedGrammarCorrector()
        
        # Layer 3: Robust Intent Classifier
        print("\n[Layer 3] Loading Robust Intent Classifier...")
        self.intent_classifier = RobustIntentClassifier()
        
        # TTS Settings
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.piper_model = os.path.join(script_dir, "models/hindi/hi_IN-rohan-medium.onnx")
        self.piper_sample_rate = 22050
        
        self.HINDI_MONTHS = {
            'January': 'जनवरी', 'February': 'फ़रवरी', 'March': 'मार्च',
            'April': 'अप्रैल', 'May': 'मई', 'June': 'जून',
            'July': 'जुलाई', 'August': 'अगस्त', 'September': 'सितंबर',
            'October': 'अक्टूबर', 'November': 'नवंबर', 'December': 'दिसंबर'
        }
        
        gc.collect() # Clean up after model loading
        print("\n✓ All systems ready!\n")

    def record_with_vad(self):
        print("\n🎤 Listening... (speak now)")
        stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS,
                               rate=self.RATE, input=True,
                               frames_per_buffer=self.CHUNK)
        
        frames = []
        ring_buffer = collections.deque(maxlen=10)
        triggered = False
        silence_frames = 0
        start_time = time.time()
        speech_start = 0
        
        while True:
            frame = stream.read(self.CHUNK, exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, self.RATE)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, s in ring_buffer if s])
                if num_voiced > 0.6 * ring_buffer.maxlen:
                    triggered = True
                    print("🔴 Recording...")
                    speech_start = time.time()
                    for f, s in ring_buffer: frames.append(f)
                    ring_buffer.clear()
            else:
                frames.append(frame)
                if not is_speech:
                    silence_frames += 1
                else:
                    silence_frames = 0
                
                curr_time = time.time()
                silence_dur = (silence_frames * self.CHUNK) / self.RATE
                if silence_dur >= self.silence_threshold:
                    print("⏸️  Silence detected, processing...")
                    break
                if (curr_time - start_time) > self.max_recording_duration:
                    print("⏸️  Max duration reached, processing...")
                    break
                    
        stream.stop_stream()
        stream.close()
        
        duration = time.time() - speech_start
        if triggered and duration >= self.min_speech_duration:
            with wave.open(self.TEMP_WAV, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            return True
        return False

    def generate_response(self, intent):
        now = datetime.now()
        if intent == "time":
            return f"अभी समय है {now.strftime('%I:%M %p')}"
        elif intent == "date":
            month_hindi = self.HINDI_MONTHS.get(now.strftime('%B'), now.strftime('%B'))
            return f"आज की तारीख है {now.day} {month_hindi} {now.year}"
        elif intent == "hello":
            return "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?"
        elif intent == "goodbye":
            return "अलविदा! फिर मिलेंगे।"
        elif intent == "thank_you":
            return "आपका स्वागत है!"
        elif intent == "help":
            return "मैं समय, तारीख बता सकता हूं। आप क्या जानना चाहते हैं?"
        elif intent == "stop":
            return "ठीक है, बंद कर रहा हूं।"
        return "माफ़ करें, मैं समझ नहीं पाया। कृपया फिर से बोलें।"

    def speak(self, text):
        print(f"🔊 Speaking (Natural Voice)...")
        if os.path.exists(self.piper_model):
            try:
                process = subprocess.Popen(
                    [sys.executable, '-m', 'piper', '--model', self.piper_model, '--output-raw'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                audio_data, _ = process.communicate(input=text.encode('utf-8'))
                if audio_data:
                    p = pyaudio.PyAudio()
                    stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.piper_sample_rate, output=True)
                    stream.write(audio_data)
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    return
            except Exception: pass
        subprocess.run(['espeak-ng', '-v', 'hi', text], check=False)

    def run(self):
        try:
            while True:
                if self.record_with_vad():
                    start = time.time()
                    
                    if self.use_faster_whisper:
                        # Transcribe using faster-whisper
                        # Returns: (segments_generator, transcription_info)
                        segments, info = self.asr_model.transcribe(
                            self.TEMP_WAV,
                            beam_size=1,            # Greedy decoding (faster)
                            language="hi",          # Hindi
                            vad_filter=False,       # Already using webrtcvad
                            condition_on_previous_text=False # Faster
                        )
                        raw_text = " ".join([segment.text for segment in segments]).strip()
                    else:
                        # Fallback to standard whisper
                        result = self.asr_standard.transcribe(self.TEMP_WAV, language="hi", fp16=False)
                        raw_text = result['text'].strip()
                        
                    print(f"📝 Raw transcription: '{raw_text}' ({time.time()-start:.2f}s)")
                    
                    corrected = self.corrector.correct(raw_text)
                    
                    start = time.time()
                    intent, conf = self.intent_classifier.classify(corrected)
                    print(f"🎯 Intent: {intent} (confidence: {conf:.1%}, {time.time()-start:.3f}s)")
                    
                    response = self.generate_response(intent)
                    print(f"💬 Response: {response}")
                    self.speak(response)
                    
                    if intent in ["stop", "goodbye"]:
                        print("\n👋 Goodbye!")
                        break
                    print("-" * 60)
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
        finally:
            if os.path.exists(self.TEMP_WAV): os.remove(self.TEMP_WAV)
            self.audio.terminate()

if __name__ == "__main__":
    assistant = RealtimeVoiceAssistant()
    assistant.run()
