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
import webrtcvad
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
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'wakt', 'waqt'],
            'time_query': ['क्या', 'कितने', 'कितना', 'बताओ', 'बतओ', 'what', 'कैसा'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'दिन', 'आज'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम'],
            'goodbye': ['अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'चलता', 'जाता'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत'],
            # Dance intent
            'dance': ['नाच', 'नाचो', 'डांस', 'नाचना', 'नाचकर', 'natch', 'nath', 'naach'],
            'weather': ['मौसम', 'weather', 'बारिश', 'ठंड', 'गर्मी', 'तापमान'],
            'joke': ['जोक', 'joke', 'मजाक', 'hansaao', 'mazaq', 'चुटकुला'],
            # Music intent
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play'],
            # Alarm intent
            'alarm': ['अलार्म', 'alarm', 'रिमाइंडर', 'जगाओ', 'wake', 'timer'],
            # News intent
            'news': ['समाचार', 'न्यूज़', 'news', 'खबर', 'headlines', 'अपडेट', 'social', 'society', 'samacar', 'topic', 'society', 'knife'],
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
            
            # --- Heavy-Duty Romanized to Devanagari Bridge ---
            (r'\bsamay\b', 'समय'), (r'\bsamae\b', 'समय'), (r'\bsmae\b', 'समय'), (r'\bsama\b', 'समय'), (r'\bsame\b', 'समय'),
            (r'\bkya\b', 'क्या'), (r'\bkiya\b', 'क्या'), (r'\bkae\b', 'क्या'),
            (r'\bhai\b', 'है'), (r'\bha\b', 'है'), (r'\bhura\b', 'हो रहा'), (r'\bho\b', 'हो'), (r'\bhai\b', 'है'),
            (r'\btariq\b', 'तारीख'), (r'\btarikh\b', 'तारीख'),
            (r'\bnamaste\b', 'नमस्ते'), (r'\bnamasitai\b', 'नमस्ते'),
            (r'\bshukriya\b', 'शुक्रिया'), (r'\bshukriyaa\b', 'शुक्रिया'), (r'\bsukriya\b', 'शुक्रिया'), (r'\bsukria\b', 'शुक्रिया'),
            (r'\baaj\b', 'आज'), (r'\baach\b', 'आज'), (r'\baj\b', 'आज'), (r'\bad\b', 'आज'),
            (r'\bmadad\b', 'मदद'), (r'\bmodot\b', 'मदद'), (r'\bmodat\b', 'मदद'),
            (r'\balarm\b', 'अलार्म'), (r'\balum\b', 'अलार्म'), (r'\balurm\b', 'अलार्म'), (r'\balbum\b', 'अलार्म'), (r'\alaam\b', 'अलार्म'),
            (r'\bvither\b', 'मौसम'), (r'\bweather\b', 'मौसम'), (r'\bwather\b', 'मौसम'), (r'\bmoasam\b', 'मौसम'), (r'\bwethar\b', 'मौसम'), (r'\bmosaam\b', 'मौसम'), (r'\bmonsam\b', 'मौसम'), (r'\bmousam\b', 'मौसम'),
            (r'\bjoke\b', 'जोक'), (r'\bjok\b', 'जोक'),
            (r'\bmazaq\b', 'मजाक'), (r'\bmazak\b', 'मजाक'),
            (r'\bgana\b', 'गाना'), (r'\bgaana\b', 'गाना'), (r'\bsong\b', 'गाना'),
            (r'\bnaaj\b', 'नाच'), (r'\bnaach\b', 'नाच'), (r'\bdance\b', 'डांस'), (r'\bnaacu\b', 'नाचो'), (r'\bnaachu\b', 'नाचो'), (r'\bnachiye\b', 'नाचो'),
            (r'\bnathke\b', 'नाचके'), (r'\bnatchke\b', 'नाचके'), (r'\bnath\b', 'नाच'), (r'\bnatch\b', 'नाच'),
            (r'\balvida\b', 'अलविदा'), (r'\bbye\b', 'bye'),
            (r'\bdhanyawad\b', 'धन्यवाद'), (r'\bdhanyavad\b', 'धन्यवाद'), (r'\bdhani\b', 'धन्य'), (r'\bavad\b', 'वाद'), (r'\bdanny\b', 'धन्य'),
            (r'\bbatau\b', 'बताओ'), (r'\bbata\b', 'बताओ'), (r'\bbatai\b', 'बताओ'), (r'\bbatah\b', 'बताओ'), (r'\bbato\b', 'बताओ'),
            (r'\baapka\b', 'आपका'), (r'\baap\b', 'आप'), (r'\bmyri\b', 'मेरी'), (r'\bmerili\b', 'मेरे लिए'), (r'\bleah\b', 'लिए'), (r'\blea\b', 'लिए'),
            (r'\bbhaaut\b', 'बहुत'), (r'\bbhaut\b', 'बहुत'), (r'\btoda\b', 'थोड़ा'), (r'\btora\b', 'थोड़ा'),
            (r'\bband\b', 'बंद'), (r'\bbanth\b', 'बंद'), (r'\bbandh\b', 'बंद'), (r'\bkaro\b', 'करो'), (r'\bkaru\b', 'करो'),
            (r'\bkesar\b', 'कैसा'), (r'\bkaisa\b', 'कैसा'),
            (r'\bdhikh\b', 'दिख'), (r'\bdikh\b', 'दिख'), (r'\bkao\b', 'दिखाओ'), (r'\bdhikao\b', 'दिखाओ'), (r'\bdikao\b', 'दिखाओ'), (r'\bkautura\b', 'दिखाओ'), (r'\bkarwek\b', 'करके'),
            (r'\bwaqt\b', 'वक्त'), (r'\bwakt\b', 'वक्त'),
            (r'samayakya', 'समय क्या'), (r'samaybatau', 'समय बताओ'), (r'samaykyahai', 'समय क्या है'),
            (r'samaykyahora', 'समय क्या हो रहा'), (r'samayhora', 'समय हो रहा'),
            (r'bandhojao', 'बंद करो'), (r'mandojao', 'बंद करो'), (r'bantujao', 'बंद करो'), (r'bandho', 'बंद करो'),
            (r'\bguit\b', 'quit'), (r'\bshuit\b', 'quit'), (r'\bquit\b', 'quit'), (r'\bexit\b', 'exit'),
            (r"today's mosam", 'आज का मौसम'), (r'what will happen', 'weather बताओ'), (r'how will it live', 'कैसा रहेगा'),
            (r'banthkaru', 'बंद करो'), (r'banthkaro', 'बंद करो'), (r'sukriya', 'शुक्रिया'), (r'sukria', 'शुक्रिया'),
            (r'\bsocial\b', 'समाचार'), (r'\bsociety\b', 'समाचार'), (r'\bsamachar\b', 'समाचार'), (r'\bsamacar\b', 'समाचार'),
            (r'\btopic\b', 'समाचार'), (r'\bknife\b', 'समाचार'), (r'\buse\b', 'समाचार'), (r'\blet us know\b', 'बताओ'),
            (r'dhannewad', 'धन्यवाद्'), (r'dhanewad', 'धन्यवाद्'),
            
            # --- Transliterated Urdu Fragment Bridge ---
            (r'दहनय', 'धन्य'), (r'आवअद', 'वाद'), (r'अवाअद', 'वाद'), (r'दनग', 'धन्य'), (r'दहनगय', 'धन्य'), (r'वअज', 'वाद'), (r'कअ', 'का'), (r'मदद', 'मदद'),
            
            # --- Urdu Script Bridge (Unicode Hallucination Fix) ---
            (r"\u0622\u0686", "आज"), (r"\u0633\u0645", "सम"), (r"\u0686\u0627\u0631", "चार"), (r"\u062c\u0648\u06a9", "जोक"), (r"\u0645\u0630\u0627\u06a9", "मजाक"),
            (r"\u06a9\u06cc\u0627", "क्या"), (r"\u06c1\u06d2", "है"), (r"\u0628\u062a\u0620", "बताओ"),
            
            # --- Phonetic Devanagari Corrections ---
            (r'\bमसम\b', 'मौसम'), (r'\bमोसम\b', 'मौसम'),
            (r'\bबरश\b', 'बारिश'), (r'\bठड\b', 'ठंड'),
            (r'\bगरम\b', 'गर्मी'), (r'\bजक\b', 'जोक'),
            (r'\बमजक\b', 'मजाक'), (r'\bचटकल\b', 'चुटकुला'),
            (r'\bगन\b', 'गाना'), (r'\bसगत\b', 'संगीत'),
            (r'\bअलरम\b', 'अलार्म'), (r'\bरमइडर\b', 'रिमाइंडर'),
            (r'\bसमचर\b', 'समाचार'), (r'\bनयज़\b', 'न्यूज़'),
            (r'\bखबर\b', 'खबर'),
        ]
        
        # Heavy-Duty Perso-Arabic (Urdu) to Devanagari character mapping
        self.urdu_map = {
            '\u0622': 'आ', '\u0627': 'अ', '\u0628': 'ब', '\u067e': 'प', '\u062a': 'त', 
            '\u0672': 'ट', '\u062b': 'स', '\u062c': 'ज', '\u0686': 'च', '\u062d': 'ह', 
            '\u062e': 'ख', '\u062f': 'द', '\u0688': 'ड', '\u0630': 'ज', '\u0631': 'र', 
            '\u0632': 'ज', '\u0698': 'झ', '\u0633': 'स', '\u0634': 'श', '\u0635': 'स', 
            '\u0636': 'ज', '\u0637': 'त', '\u0638': 'ज', '\u0639': 'अ', '\u063a': 'ग', 
            '\u0641': 'फ', '\u0642': 'क', '\u06a9': 'क', '\u06af': 'ग', '\u0644': 'ल', 
            '\u0645': 'म', '\u0646': 'न', '\u06ba': 'न', '\u0648': 'व', '\u06c1': 'ह', 
            '\u06be': 'ह', '\u06d2': 'ए', '\u06a4': 'व', '\u06cc': 'य', '\u064a': 'य',
            '\u0626': 'ए', '\u064b': 'न', '\u0621': 'इ', '\u0624': 'ओ'
        }
        
        try:
            from rapidfuzz import fuzz
            self.fuzz = fuzz
            self.use_fuzzy = True
            self.fuzzy_threshold = 75
        except ImportError:
            self.use_fuzzy = False

    def _transliterate_perso_arabic_to_devanagari(self, text):
        """Character-level conversion of Urdu script to Devanagari"""
        result = []
        for char in text:
            if '\u0600' <= char <= '\u06FF':
                result.append(self.urdu_map.get(char, ''))
            else:
                result.append(char)
        return "".join(result)

    def correct(self, text):
        if not text: return ""
        original_text = text
        
        # Pass 0: Aggressive Urdu-to-Hindi character transliteration
        text = self._transliterate_perso_arabic_to_devanagari(text)
        
        # Pass 1: Regex patterns (Case-insensitive for Romanized parts)
        corrected = text
        for pattern, replacement in self.error_patterns:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
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
        
        # Pass 1: Check for exact match first (Avoid over-correction like naacho -> naach)
        for category, vocab_list in self.core_vocabulary.items():
            for vocab_word in vocab_list:
                if word_lower == vocab_word.lower():
                    return word
        
        # Pass 2: Fuzzy matching only if no exact match found
        best_match = word
        best_score = 0
        
        for category, vocab_list in self.core_vocabulary.items():
            for vocab_word in vocab_list:
                similarity = self.fuzz.ratio(word_lower, vocab_word.lower())
                if similarity > best_score:
                    best_score = similarity
                    best_match = vocab_word
        
        if best_score >= self.fuzzy_threshold:
            return best_match
            
        return word

# ============================================================
# LAYER 3: ROBUST INTENT CLASSIFICATION
# ============================================================

class RobustIntentClassifier:
    def __init__(self, model_path=None, use_onnx=True):
        """
        Initialize intent classifier with ONNX optimization
        Falls back to PyTorch if ONNX model not found
        """
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, 'hindi_intent_model_final')
        
        # Check for ONNX model
        onnx_path = model_path.replace('_final', '_onnx_int8')
        
        if use_onnx and os.path.exists(onnx_path):
            print(f"⚙️  Loading ONNX-optimized classifier from {os.path.basename(onnx_path)}...")
            try:
                self._load_onnx_model(onnx_path)
                return  # Success, skip PyTorch loading
            except Exception as e:
                print(f"⚠️  ONNX loading failed: {e}")
                print(f"   Falling back to PyTorch model...")
        
        # Load PyTorch model (original or fallback)
        if use_onnx and not os.path.exists(onnx_path):
            print(f"ℹ️  ONNX model not found at {os.path.basename(onnx_path)}")
            print(f"   Using PyTorch model (run convert_indicbert_to_onnx.py to optimize)")
        
        print(f"⚙️  Loading PyTorch classifier from {os.path.basename(model_path)}...")
        self._load_pytorch_model(model_path)

    def _load_onnx_model(self, model_path):
        """Load ONNX-optimized model"""
        from optimum.onnxruntime import ORTModelForSequenceClassification
        
        # Load label map
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            self.id2label = json.load(f)['id2label']
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_path,
            provider="CPUExecutionProvider"
        )
        
        self.device = torch.device("cpu")
        self.model_type = "onnx"
        
        print("   ✓ ONNX INT8 model loaded")
        
        # Pin to A76 cores (Cubie A7A optimization)
        try:
            import psutil
            p = psutil.Process()
            p.cpu_affinity([0, 1])  # Cores 0-1 are Cortex-A76
            print("   ✓ Process pinned to Cortex-A76 cores")
        except Exception as e:
            pass  # Not critical
        
        # Set thread limits for 6GB RAM
        import os
        os.environ['OMP_NUM_THREADS'] = '2'
        torch.set_num_threads(2)
        
        # Robust fallback keywords for 13 intents
        self.fallback_patterns = {
            'stop': ['बंद', 'स्टॉप', 'stop', 'रुको', 'रूको', 'exit', 'quit', 'close', 'बन्द', 'समाप्त', 'खत्म', 'band'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'samay', 'samai', 'time'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'आज', 'दिन', 'कैलेंडर', 'tariq', 'tarikh', 'tithi'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम', 'namaste'],
            'goodbye': ['अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'चलता', 'जाता', 'alvida'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'शुक्रीया', 'shukriya'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत', 'madad'],
            'dance': ['नाच', 'dance', 'नाचो', 'डांस'],
            'weather': ['मौसम', 'weather', 'बारिश' ,'ठंड', 'गर्मी', 'तापमान'],
            'joke': ['जोक', 'joke', 'मजाक', 'हँसाओ', 'funny', 'चुटकुला', 'कॉमेडी'],
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play'],
            'alarm': ['अलार्म', 'alarm', 'रिमाइंडर', 'जगाओ', 'wake', 'timer'],
            'news': [' समाचार', ' न्यूज़', 'news', 'खबर', 'headlines', 'अपडेट'],
        }

    def _load_pytorch_model(self, model_path):
        """Load original PyTorch model (fallback)"""
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            self.id2label = json.load(f)['id2label']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model_type = "pytorch"
        
        print("   ✓ PyTorch float32 model loaded")
        
        # Keep fallback patterns (add them here too)
        self.fallback_patterns = {
            'stop': ['बंद', 'स्टॉप', 'stop', 'रुको', 'रूको', 'exit', 'quit', 'close', 'बन्द', 'समाप्त', 'खत्म', 'band'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'samay', 'tim'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'आज', 'दिन', 'कैलेंडर', 'tariq', 'tarikh', 'tithi'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम', 'namaste'],
            'goodbye': ['अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'चलता', 'जाता', 'alvida'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'शुक्रीया', 'shukriya'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत', 'madad'],
            'dance': ['नाच', 'dance', 'नाचो', 'डांस'],
            'weather': ['मौसम', 'weather', 'बारिश' ,'ठंड', 'गर्मी', 'तापमान'],
            'joke': ['जोक', 'joke', 'मजाक', 'हँसाओ', 'funny', 'चुटकुला', 'कॉमेडी'],
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play'],
            'alarm': ['अलार्म', 'alarm', 'रिमाइंडर', 'जगाओ', 'wake', 'timer'],
            'news': [' समाचार', ' न्यूज़', 'news', 'खबर', 'headlines', 'अपडेट'],
        }

    def classify(self, text):
        if not text.strip(): return "unknown", 0.0
        
        # Robust Pre-processing (Strip punctuation, Urdu script residue, and Noise)
        text = re.sub(r'[.,!?।|]', '', text).strip()
        # Strip remaining Urdu/Arabic characters if any leaked
        text = re.sub(r'[\u0600-\u06FF]', '', text).strip()
        text = re.sub(r'(?i)\b(teeke|theke|thek|tik|ok|hlo|hey)\b', '', text).strip()
        
        # Stage 1: IndicBERT
        inputs = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            
        intent = self.id2label.get(str(idx.item()), "unknown")
        confidence = conf.item()
        
        # High confidence? Trust IndicBERT (Increased to 0.82 for better robustness)
        if confidence >= 0.82:
            return intent, confidence
            
        # Try fuzzy fallback for EVERYTHING else
        fallback_intent = self._fuzzy_fallback(text)
        if fallback_intent:
            print(f"✓ Fuzzy fallback matched: {fallback_intent}")
            return fallback_intent, 0.90
            
        # Only trust IndicBERT if confidence is very high (82%+) and fallback failed
        if confidence >= 0.82:
            return intent, confidence
            
        return "unknown", confidence

    def _fuzzy_fallback(self, text):
        from rapidfuzz import fuzz
        text_lower = text.lower()
        
        # Pass 1: Local token-based presence (Strict)
        words = set(text_lower.split())
        for intent, keywords in self.fallback_patterns.items():
            for kw in keywords:
                if kw.lower() in words:
                    return intent
                    
        # Pass 2: Fuzzy Set Ratio (with safety checks)
        scores = {}
        for intent, keywords in self.fallback_patterns.items():
            max_score = 0
            for kw in keywords:
                # Ignore very short keywords for fuzzy matching to avoid "hi" in "abhi"
                if len(kw) < 3: continue 
                
                score = fuzz.token_set_ratio(text_lower, kw.lower())
                max_score = max(max_score, score)
            scores[intent] = max_score
            
        if scores:
            best_intent = max(scores, key=scores.get)
            if scores[best_intent] >= 95: # Higher threshold for safety
                return best_intent
            
        return None

# ============================================================
# MAIN ASSISTANT CLASS
# ============================================================

class RealtimeVoiceAssistant:
    def _check_memory_safety(self):
        """Ensure sufficient RAM for 6GB system"""
        try:
            import psutil
            mem = psutil.virtual_memory()
            
            free_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            
            print(f"💾 Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            
            if mem.available < 2.5 * 1024**3:  # Less than 2.5GB free
                print("⚠️  WARNING: Low memory!")
                print(f"   Available: {free_gb:.1f}GB")
                print(f"   Recommended: 2.5GB minimum")
                print("   Close other applications for best performance.")
            
        except ImportError:
            print("⚠️  psutil not installed (pip install psutil)")

    def __init__(self):
        # Memory safety check
        self._check_memory_safety()
        
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
                "base",                     # Model size (Base for speed)
                device="cpu",               # CPU inference
                compute_type="int8",        # 8-bit quantization (Speed boost)
                cpu_threads=4,              # SBC optimization
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
        elif intent == 'dance':
            import random
            dance_moves = [
                "मैं नाच रहा हूं... धिन धिन धा! लेकिन मेरे पास पैर नहीं हैं!",
                "नाचने के लिए मुझे स्पीकर की जरूरत है, वरना मैं सिर्फ डाटा डांस कर सकता हूं!",
                "मैं अभी नाच सीख रहा हूं। जल्दी ही आपके साथ डांस करूंगा!",
                "डांस मोड ऑन! लेकिन मैं ऑफलाइन हूं, इसलिए सिर्फ वर्चुअल डांस कर सकता हूं!"
            ]
            return random.choice(dance_moves)
        elif intent == 'weather':
            return "मौसम की जानकारी उपलब्ध नहीं है। मैं ऑफलाइन काम करता हूं। लेकिन आज दिन अच्छा लग रहा है!"
        elif intent == 'joke':
            import random
            jokes = [
                "मैं अभी जोक सीख रहा हूं। जल्दी ही आपको हंसा दूंगा!",
                "एक रोबोट डॉक्टर के पास गया। डॉक्टर बोला: आप तो बिल्कुल फिट हैं... बस थोड़ा ऑयल चाहिए!",
                "मैं तो AI हूं, मुझे सिर्फ डाटा से प्यार है!",
                "मेरा एक दोस्त है, वह भी AI है। हम दोनों बहुत स्मार्ट हैं!",
                "मजाक: मैंने एक बार कहा था मैं ऑफलाइन हूं, लेकिन कोई मान ही नहीं रहा था!"
            ]
            return random.choice(jokes)
        elif intent == 'music':
            return "गाना बजा रहा हूं... धुन धुन धु! वैसे मैं अभी स्पीकर से जुड़ा नहीं हूं।"
        elif intent == 'alarm':
            return "ठीक है, सुबह 7 बजे के लिए अलार्म सेट कर दिया है।"
        elif intent == 'news':
            return "समाचार सेवा ऑफलाइन है। लेकिन आज का दिन बहुत अच्छा है!"
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
                    start_thinking = time.time()
                    
                    if self.use_faster_whisper:
                        # Transcribe using faster-whisper
                        segments, info = self.asr_model.transcribe(
                            self.TEMP_WAV,
                            beam_size=1,
                            language="hi",
                            initial_prompt="Hindi Assistant. No Urdu script.",
                            vad_filter=False,
                            condition_on_previous_text=False
                        )
                        raw_text = " ".join([segment.text for segment in segments]).strip()
                    else:
                        # Fallback to standard whisper
                        result = self.asr_standard.transcribe(self.TEMP_WAV, language="hi", fp16=False)
                        raw_text = result['text'].strip()
                        
                    print(f"📝 Raw transcription: '{raw_text}' ({time.time()-start_thinking:.2f}s)")
                    
                    corrected = self.corrector.correct(raw_text)
                    
                    intent_start = time.time()
                    intent, conf = self.intent_classifier.classify(corrected)
                    print(f"🎯 Intent: {intent} (confidence: {conf:.1%}, {time.time()-intent_start:.3f}s)")
                    
                    response = self.generate_response(intent)
                    
                    # 5-second thinking timeout check
                    total_thinking_time = time.time() - start_thinking
                    if total_thinking_time > 5.0:
                        print(f"⚠️  Thinking timeout ({total_thinking_time:.2f}s > 5s). Using fallback.")
                        response = "माफ़ करें, मैं समझ नहीं पाया। कृपया फिर से बोलें।"
                    
                    print(f"💬 Response: {response}")
                    self.speak(response)
                    
                    if intent in ["stop", "goodbye"] and total_thinking_time <= 5.0:
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
