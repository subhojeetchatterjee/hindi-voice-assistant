#!/usr/bin/env python3
"""
Real-time Hindi Voice Assistant with Faster-Whisper
Optimized for SBC (Single Board Computer)
Bharat AI-SoC Challenge Submission

Architecture:
- Layer 1: Faster-Whisper (High-speed, Int8-quantized Hindi ASR)
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
import concurrent.futures
import unicodedata

# ============================================================
# LAYER 2: ADVANCED GRAMMAR CORRECTION
# ============================================================

class AdvancedGrammarCorrector:
    """Layer 2: Phonetic grammar correction with fuzzy matching"""
    
    def __init__(self):
        # Core vocabulary by intent category
        self.core_vocabulary = {
            'stop': ['बंद', 'बन्द', 'स्टॉप', 'स्टप', 'stop', 'रुको', 'रूको', 'रुक', 'बन्त', 'बन्ते', 'बंद्ते', 'बन्तोजा', 'अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय'],
            'command_stop': ['करो', 'करदो', 'कर', 'कर do', 'हो', 'हो जाओ'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'wakt', 'waqt'],
            'time_query': ['क्या', 'कितने', 'कितना', 'बताओ', 'बतओ', 'what', 'कैसा'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'दिन', 'आज'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम', 'naam', 'name', 'नाम'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'जुप्रिया', 'सुक्रिया', 'सुप्रिया', 'सुक्या', 'धनिवाद', 'जुक्रिया', 'जोक्रिया'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत'],
            # Dance intent
            'dance': ['नाच', 'नाचो', 'डांस', 'नाचना', 'नाचकर', 'natch', 'nath', 'naach', 'राच', 'नज', 'दिकार', 'तिकाओ', 'दिखाओ'],
            'weather': ['मौसम', 'weather', 'बारिश', 'ठंड', 'गर्मी', 'तापमान'],
            'joke': ['जोक', 'joke', 'मजाक', 'hansaao', 'mazaq', 'चुटकुला', 'जुक', 'मचाक', 'मजक', 'जुक्र', 'जुक्रा'],
            # Music intent
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play', 'काना', 'कना', 'सुला', 'बाना', 'खाना', 'बंदाओ', 'बंदानाओ', 'पदाओ'],
            # News intent
            'news': ['समाचार', 'न्यूज़', 'news', 'खबर', 'headlines', 'अपडेट', 'social', 'society', 'samacar', 'topic', 'society', 'knife', 'समजार', 'समाद्यार'],
        }
        
        # Critical error patterns (regex)
        self.error_patterns = [
            # STOP INTENT - Critical phonetic fixes
            (r'\bबन\b', 'बंद'),
            (r'\bबन्त\b', 'बंद'),
            (r'\bबन्ते\b', 'बंद'),
            (r'\bबंद्ते\b', 'बंद'),
            (r'\bबन्तोजा\b', 'बंद करो'),
            (r'\bबंद्तोजा\b', 'बंद करो'),
            (r'बन करो', 'बंद करो'),
            (r'वन करो', 'बंद करो'),
            (r'बंदकरो', 'बंद करो'),
            (r'समये', 'समय'),
            (r'बतओ', 'बताओ'),
            (r'क्य\b', 'क्या'),
            (r'कितन\b', 'कितने'),
            (r'मुझ\b', 'मुझे'), (r'\bमुजे\b', 'मुझे'),
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
            # THANK_YOU INTENT - Critical phonetic fixes
            (r'\bshukriya\b', 'शुक्रिया'), (r'\bshukriyaa\b', 'शुक्रिया'), (r'\bsukriya\b', 'शुक्रिया'), (r'\bsukria\b', 'शुक्रिया'),
            (r'\bजुप्रिया\b', 'शुक्रिया'), (r'\bसुक्रिया\b', 'शुक्रिया'), (r'\bसुप्रिया\b', 'शुक्रिया'), (r'\bजुक्रिया\b', 'शुक्रिया'),
            # JOKE INTENT - Critical phonetic fixes
            (r'\bजुक्र\b', 'जोक'), (r'\bजुक्रा\b', 'जोक'), (r'\bजुक\b', 'जोक'), (r'\bमजाक्र\b', 'मजाक'),
            # MUSIC INTENT - Phonetic consistency
            (r'\bबंदाओ\b', 'बजाओ'), (r'\bबंदानाओ\b', 'बजाओ'), (r'\bपदाओ\b', 'बजाओ'), (r'\bकाना\b', 'गाना'),
            # NEWS INTENT - Phonetic consistency
            (r'\bसमजार\b', 'समाचार'), (r'\bसमचार\b', 'समाचार'), (r'\bसमाद्यार\b', 'समाचार'),
            (r'\bसुक्या\b', 'शुक्रिया'), (r'\bदहनिवाद\b', 'धन्यवाद'), (r'\bधनिवाद\b', 'धन्यवाद'),
            (r'\baaj\b', 'आज'), (r'\baach\b', 'आज'), (r'\baj\b', 'आज'), (r'\bad\b', 'आज'),
            (r'\bmadad\b', 'मदद'), (r'\bmodot\b', 'मदद'), (r'\bmodat\b', 'मदद'),
            (r'\bvither\b', 'मौसम'), (r'\bweather\b', 'मौसम'), (r'\bwather\b', 'मौसम'), (r'\bmoasam\b', 'मौसम'), (r'\bwethar\b', 'मौसम'), (r'\bmosaam\b', 'मौसम'), (r'\bmonsam\b', 'मौसम'), (r'\bmousam\b', 'मौसम'),
            # JOKE INTENT - Critical phonetic fixes
            (r'\bjoke\b', 'जोक'), (r'\bjok\b', 'जोक'), (r'\bजुक\b', 'जोक'),
            (r'\bmazaq\b', 'मजाक'), (r'\bmazak\b', 'मजाक'), (r'\bमचाक\b', 'मजाक'), (r'\bमजक\b', 'मजाक'),
            # MUSIC INTENT - Critical phonetic fixes
            (r'\bgana\b', 'गाना'), (r'\bgaana\b', 'गाना'), (r'\bsong\b', 'गाना'), (r'\bकाना\b', 'गाना'), (r'\bकना\b', 'गाना'), (r'\bकानो\b', 'गाना'), (r'\bबाना\b', 'गाना'), (r'\bखाना\b', 'गाना'),
            # DANCE INTENT - Critical phonetic fixes
            (r'\bnaaj\b', 'नाच'), (r'\bnaach\b', 'नाच'), (r'\bdance\b', 'डांस'), (r'\bnaacu\b', 'नाचो'), (r'\bnaachu\b', 'नाचो'), (r'\bnachiye\b', 'नाचो'), (r'\bराच\b', 'नाच'), (r'\bनज\b', 'नाच'),
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
            (r'\bsama\s*chhar\b', 'समाचार'), (r'\bsamachhar\b', 'समाचार'), (r'\bsamachar\b', 'समाचार'),
            (r'\bchhar\b', 'समाचार'), (r'\bchahar\b', 'समाचार'), (r'\bchar\b', 'समाचार'),
            (r'\bnews\b', 'समाचार'), (r'\bnuse\b', 'समाचार'), (r'\bnuze\b', 'समाचार'),
            (r'\bbantuja\b', 'बंद करो'), (r'\bbantuja\s*ho\b', 'बंद करो'), (r'\bbanthoja\b', 'बंद करो'),
            (r'\bbantujao\b', 'बंद करो'), (r'\bbandoja\b', 'बंद करो'), (r'\bbanthuja\b', 'बंद करो'),
            (r'अज के दिकार', 'नाच के दिखाओ'), (r'अज के तिकाओ', 'नाच के दिखाओ'), (r'अज के', 'हमें'),
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
            
            # Music intent variants (गाना)
            (r'\bganna\b', 'गाना'), (r'\bgana\b', 'गाना'), (r'\bkanna\b', 'गाना'),
            (r'\bkana\b', 'गाना'), (r'\bganaa\b', 'गाना'),
            (r'\bmujhe\s+ganna\b', 'गाना'), (r'\bmujee\s+kanna\b', 'गाना'),
            (r'\bsunao\b', 'सुनाओ'), (r'\bsuna\b', 'सुनाओ'), (r'\bsunaai\b', 'सुनाओ'), (r'\bसुला\b', 'सुनाओ'),
            
            # Weather intent variants (मौसम)
            (r'\bviter\b', 'मौसम'), (r'\bwither\b', 'मौसम'), (r'\bvether\b', 'मौसम'),
            (r'\bviter\s+batal\b', 'मौसम बताओ'),
            (r'\bbatal\b', 'बताओ'), (r'\bbata\b', 'बताओ'),
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
            self.fuzzy_threshold = 80
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
        
        # Pass 0.5: Normalize spaces (fixes "Sama Chhar" → "samachhar")
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
        text = text.strip()
        
        # Pass 0.75: Noise cleanup
        noise_words = r'\b(umm|uh|hmm|aah|uhh|like|you know|bhujhey|mujee|aa|eh)\b'
        text = re.sub(noise_words, '', text, flags=re.IGNORECASE)
        text = re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)  # "Gannna" → "Ganna"
        text = re.sub(r'\s+', ' ', text).strip()
        
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
        import os
        import json
        import torch
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
        except Exception:
            pass  # Not critical
        
        # Set thread limits for 6GB RAM
        os.environ['OMP_NUM_THREADS'] = '2'
        torch.set_num_threads(2)
        
        # Robust fallback keywords for 13 intents
        self.fallback_patterns = {
            'stop': ['बंद', 'स्टॉप', 'stop', 'रुको', 'रूको', 'exit', 'quit', 'close', 'बन्द', 'समाप्त', 'खत्म', 'band', 'bantuja', 'अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'alvida'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'samay', 'samai', 'time', 'samaya'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'आज', 'दिन', 'कैलेंडर', 'tariq', 'tarikh', 'tithi', 'din'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम', 'namaste', 'naam', 'name', 'नाम'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'शुक्रीया', 'shukriya', 'जुक्रिया', 'जुप्रिया'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत', 'madad'],
            'dance': ['नाच', 'dance', 'नाचो', 'डांस', 'दिकार', 'तिकाओ', 'दिखाओ'],
            'weather': ['मौसम', 'weather', 'बारिश' ,'ठंड', 'गर्मी', 'तापमान', 'viter', 'wither', 'vether', 'batal'],
            'joke': ['जोक', 'joke', 'मजाक', 'हँसाओ', 'funny', 'चुटकुला', 'कॉमेडी', 'जुक्र', 'जुक्रा'],
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play', 'ganna', 'gana', 'kanna', 'kana', 'sunao', 'suna', 'बंदानाओ', 'बंदाना', 'बजा', 'bajao', 'बंदाओ'],
            'news': ['समाचार', 'न्यूज़', 'news', ' खबर', 'headlines', 'अपडेट', 'chhar', 'char', 'चार', 'चर', 'samachhar', 'समजार', 'समाद्यार'],
        }

    def _load_pytorch_model(self, model_path):
        """Load original PyTorch model (fallback)"""
        with open(os.path.join(model_path, 'label_map.json'), 'r') as f:
            self.id2label = json.load(f)['id2label']
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.model.eval()
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model_type = "pytorch"
        
        print("   ✓ PyTorch float32 model loaded")
        
        # Keep fallback patterns (add them here too)
        self.fallback_patterns = {
            'stop': ['बंद', 'स्टॉप', 'stop', 'रुको', 'रूको', 'exit', 'quit', 'close', 'बन्द', 'समाप्त', 'खत्म', 'band', 'bantuja', 'अलविदा', 'अलवीदा', 'बाय', 'bye', 'टाटा', 'गुडबाय', 'alvida'],
            'time': ['समय', 'टाइम', 'time', 'बजे', 'घड़ी', 'वक्त', 'घंटा', 'घंटे', 'samay', 'samai', 'time', 'samaya'],
            'date': ['तारीख', 'तिथि', 'डेट', 'date', 'आज', 'दिन', 'कैलेंडर', 'tariq', 'tarikh', 'tithi', 'din'],
            'hello': ['नमस्ते', 'नमस्कार', 'हैलो', 'हेलो', 'hello', 'hi', 'हाय', 'प्रणाम', 'namaste', 'naam', 'name', 'नाम'],
            'thank_you': ['धन्यवाद', 'शुक्रिया', 'thanks', 'thank', 'थैंक', 'आभार', 'शुक्रीया', 'shukriya', 'जुक्रिया', 'जुप्रिया'],
            'help': ['मदद', 'हेल्प', 'help', 'सहायता', 'सहायत', 'madad'],
            'dance': ['नाच', 'dance', 'नाचो', 'डांस', 'दिकार', 'तिकाओ', 'दिखाओ'],
            'weather': ['मौसम', 'weather', 'बारिश' ,'ठंड', 'गर्मी', 'तापमान', 'viter', 'wither', 'vether', 'batal'],
            'joke': ['जोक', 'joke', 'मजाक', 'हँसाओ', 'funny', 'चुटकुला', 'कॉमेडी', 'जुक्र', 'जुक्रा'],
            'music': ['गाना', 'संगीत', 'music', 'song', 'बजाओ', 'चलाओ', 'play', 'ganna', 'gana', 'kanna', 'kana', 'sunao', 'suna', 'बंदानाओ', 'बंदाना', 'बजा', 'bajao', 'बंदाओ', 'काना', 'पदाओ'],
            'news': ['समाचार', 'न्यूज़', 'news', ' खबर', 'headlines', 'अपडेट', 'chhar', 'char', 'चार', 'चर', 'samachhar', 'समजार', 'समाद्यार'],
        }

    def classify(self, text):
        if not text.strip(): return "unknown", 0.0
        
        # Robust Pre-processing (Strip punctuation, Urdu script residue, and Noise)
        text = re.sub(r'[.,!?।|]', '', text).strip()
        # Strip remaining Urdu/Arabic characters if any leaked
        text = re.sub(r'[\u0600-\u06FF]', '', text).strip()
        text = re.sub(r'(?i)\b(teeke|theke|thek|tik|ok|hlo|hey)\b', '', text).strip()
        
        # Stage 0: Keyword Guardrails (Hard override for absolute clarity)
        words = set(text.lower().split())
        if any(w in words for w in ['दिन', 'तारीख', 'तिथि', 'date', 'तारीक']):
            return "date", 0.99
        if any(w in words for w in ['बजाओ', 'बंदानाओ', 'बंदाना', 'गाना', 'संगीत', 'music', 'song', 'बजा', 'बंदाओ', 'काना', 'पदाओ']):
            return "music", 0.99
        if any(w in words for w in ['जोक', 'joke', 'मजाक', 'चुटकुला', 'जुक्र', 'जुक्रा']):
            return "joke", 0.99
        if any(w in words for w in ['धन्यवाद', 'शुक्रिया', 'thx', 'thanks', 'जुक्रिया']):
            return "thank_you", 0.99
        if any(w in words for w in ['समाद्यार', 'समाचार', 'news', 'खबर', 'न्यूज़', 'समजार']):
            return "news", 0.99
        if any(w in words for w in ['नाच', 'नाचो', 'डांस', 'दिकार', 'तिकाओ']):
            return "dance", 0.99
        
        # Stage 1: IndicBERT
        inputs = self.tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            
        intent = self.id2label.get(str(idx.item()), "unknown")
        confidence = conf.item()
        
        if confidence >= 0.82:
            # Stage 2: Stop Intent Sanity Check (Prevention of accidental exits)
            if intent == "stop":
                text_lower = text.lower()
                words = set(text_lower.split())
                # Must contain a stop keyword OR have extreme confidence
                has_stop_word = any(kw.lower() in words for kw in self.fallback_patterns['stop'])
                # Also check for substring match for compound Hindi phrases
                if not has_stop_word:
                    has_stop_word = any(kw.lower() in text_lower for kw in ['बंद', 'रुको', 'stop', 'exit'])
                
                if not has_stop_word and confidence < 0.97:
                    print(f"⚠️  Stop intent rejected (no keyword match). Conf: {confidence:.2f}")
                    return "unknown", confidence
            
            return intent, confidence
            
        # Try fuzzy fallback for EVERYTHING else
        fallback_intent = self._fuzzy_fallback(text)
        if fallback_intent:
            print(f"✓ Fuzzy fallback matched: {fallback_intent}")
            return fallback_intent, 0.90
            
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
            if scores[best_intent] >= 80:
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
        print("High-Speed Optimization")
        print("=" * 60)
        
        self.RATE = 16000
        self.CHUNK = 480 
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        
        self.vad = webrtcvad.Vad(2) 
        self.silence_threshold = 1.0 
        self.min_speech_duration = 0.5 
        self.max_recording_duration = 6.0  # Shorter = less noise accumulation
        
        self.audio = pyaudio.PyAudio()
        
        
        # Layer 1: ASR Loading (Faster-Whisper with Fallback)
        try:
            from faster_whisper import WhisperModel
            print("\n[Layer 1] Loading Faster-Whisper (Base, Int8 quantized)...")
            self.asr_model = WhisperModel(
                "base",                    
                device="cpu",               # CPU inference
                compute_type="int8",        # 8-bit quantization (Speed boost)
                cpu_threads=2,              # Only A76 cores (faster)
                num_workers=1               # Single worker for stability
            )
            self.use_faster_whisper = True
            print("✓ Faster-Whisper loaded (optimized for SBC)")
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
        
        # Pre-cache ALL static responses for instant playback (Parallel)
        print("\n[TTS] Pre-generating all static responses (Parallel 2-workers)...")
        self.audio_cache = {}
        
        # Static responses that never change - Normalized to NFC
        raw_responses = [
            "ठीक है, बंद कर रहा हूं।",
            "नमस्ते! मेरा नाम भारत AI है, मैं आपकी कैसे मदद कर सकता हूं?",
            "आपका स्वागत है!",
            "मैं जोक सुना सकता हूँ, संगीत बजा सकता हूँ और नाच भी सकता हूँ। समाचार और मौसम अभी ऑफलाइन हैं, लेकिन मैं समय और तारीख बता सकता हूँ। आप क्या जानना चाहते हैं?",
            "माफ़ करें, मैं समझ नहीं पाया। कृपया फिर से बोलें।",
            "मौसम की जानकारी उपलब्ध नहीं है। मैं ऑफलाइन काम करता हूं। लेकिन आज दिन अच्छा लग रहा है!",
            "गाना बजा रहा हूं... धुन धुन धु! वैसे मैं अभी स्पीकर से जुड़ा नहीं हूं।",
            "समाचार सेवा ऑफलाइन है। लेकिन आज का दिन बहुत अच्छा है!",
            "मैं नाच रहा हूं... धिन धिन धा! लेकिन मेरे पास पैर नहीं हैं!",
            "एक रोबोट डॉक्टर के पास गया। डॉक्टर बोला: आप तो बिल्कुल फिट हैं... बस थोड़ा ऑयल चाहिए!",
            "मेरा एक दोस्त है, वह भी AI है। हम दोनों बहुत स्मार्ट हैं!",
            "मजाक: मैंने एक बार कहा था मैं ऑफलाइन हूं, लेकिन कोई मान ही नहीं रहा था!"
        ]
        
        # Normalize all phrases to NFC for consistent matching
        common_responses = [unicodedata.normalize('NFC', p) for p in raw_responses]
        
        def cache_audio(phrase):
            try:
                # Use absolute path to python if needed, but sys.executable is usually right
                process = subprocess.Popen(
                    [sys.executable, '-m', 'piper', '--model', self.piper_model, '--output-raw'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                # Increased timeout for SBC stability
                audio_data, stderr_data = process.communicate(input=phrase.encode('utf-8'), timeout=30)
                
                if process.returncode != 0:
                    return phrase, None, f"Piper exited with code {process.returncode}: {stderr_data.decode()}"
                
                return phrase, audio_data, None
            except subprocess.TimeoutExpired:
                return phrase, None, "Timeout (30s) expired during generation"
            except Exception as e:
                return phrase, None, str(e)

        # Use only 2 workers to avoid CPU/RAM starvation on SBC
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(cache_audio, common_responses))
            for phrase, audio, error in results:
                if audio:
                    self.audio_cache[phrase] = audio
                    # print(f"   ✓ Cached (len={len(audio)}): {phrase}")
                else:
                    print(f"   ⚠️ Failed to cache '{phrase[:20]}...': {error}")
        
        print(f"   ✓ Successfully cached {len(self.audio_cache)}/{len(common_responses)} responses (~{len(self.audio_cache) * 0.05:.1f}MB RAM)")
        
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
        response = ""
        
        if intent == "time":
            response = f"अभी समय है {now.strftime('%I:%M %p')}"
        elif intent == "date":
            month_hindi = self.HINDI_MONTHS.get(now.strftime('%B'), now.strftime('%B'))
            response = f"आज की तारीख है {now.day} {month_hindi} {now.year}"
        elif intent == "hello":
            response = "नमस्ते! मेरा नाम भारत AI है, मैं आपकी कैसे मदद कर सकता हूं?"
        elif intent == "thank_you":
            response = "आपका स्वागत है!"
        elif intent == "help":
            response = "मैं जोक सुना सकता हूँ, संगीत बजा सकता हूँ और नाच भी सकता हूँ। समाचार और मौसम अभी ऑफलाइन हैं, लेकिन मैं समय और तारीख बता सकता हूँ। आप क्या जानना चाहते हैं?"
        elif intent == "stop":
            response = "ठीक है, बंद कर रहा हूं।"
        elif intent == 'dance':
            response = "मैं नाच रहा हूं... धिन धिन धा! लेकिन मेरे पास पैर नहीं हैं!"
        elif intent == 'weather':
            response = "मौसम की जानकारी उपलब्ध नहीं है। मैं ऑफलाइन काम करता हूं। लेकिन आज दिन अच्छा लग रहा है!"
        elif intent == 'joke':
            if not hasattr(self, '_joke_index'):
                self._joke_index = 0
            
            jokes = [
                "एक रोबोट डॉक्टर के पास गया। डॉक्टर बोला: आप तो बिल्कुल फिट हैं... बस थोड़ा ऑयल चाहिए!",
                "मेरा एक दोस्त है, वह भी AI है। हम दोनों बहुत स्मार्ट हैं!",
                "मजाक: मैंने एक बार कहा था मैं ऑफलाइन हूं, लेकिन कोई मान ही नहीं रहा था!"
            ]
            response = jokes[self._joke_index % len(jokes)]
            self._joke_index += 1
        elif intent == 'music':
            response = "गाना बजा रहा हूं... धुन धुन धु! वैसे मैं अभी स्पीकर से जुड़ा नहीं हूं।"
        elif intent == 'news':
            response = "समाचार सेवा ऑफलाइन है। लेकिन आज का दिन बहुत अच्छा है!"
        else:
            response = "माफ़ करें, मैं समझ नहीं पाया। कृपया फिर से बोलें।"
        
        # CRITICAL: Normalize to NFC before returning to ensure cache hit
        return unicodedata.normalize('NFC', response)

    def speak(self, text):
        print(f"🔊 Speaking (Natural Voice)...")
        start_tts = time.time()
        
        # Check cache first for instant playback
        # Normalize input text to NFC for consistent matching
        norm_text = unicodedata.normalize('NFC', text)
        
        if hasattr(self, 'audio_cache') and norm_text in self.audio_cache:
            print(f"   ✓ Using cached audio (0.0s)")
            audio_data = self.audio_cache[norm_text]
            
            # Play cached audio immediately
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, 
                            rate=self.piper_sample_rate, output=True,
                            frames_per_buffer=256)
            stream.write(audio_data)
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            total_time = time.time() - start_tts
            print(f"   Total latency: {total_time:.2f}s")
            return
        
        # If not cached, generate fresh audio
        print(f"   Generating fresh audio...")
        
        if os.path.exists(self.piper_model):
            try:
                process = subprocess.Popen(
                    [sys.executable, '-m', 'piper', '--model', self.piper_model, '--output-raw'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                audio_data, stderr_data = process.communicate(input=text.encode('utf-8'), timeout=10)
                
                # Show Piper errors if any
                if process.returncode != 0:
                    error_msg = stderr_data.decode()[:200] if stderr_data else "Unknown error"
                    print(f"   ⚠️ Piper failed: {error_msg}")
                    raise Exception("Piper TTS failed")
                
                if audio_data:
                    p = pyaudio.PyAudio()
                    stream = p.open(format=pyaudio.paInt16, channels=1, 
                                    rate=self.piper_sample_rate, output=True,
                                    frames_per_buffer=1024)
                    stream.write(audio_data)
                    stream.stop_stream()
                    stream.close()
                    p.terminate()
                    
                    total_time = time.time() - start_tts
                    print(f"   Total latency: {total_time:.2f}s")
                    return
            except subprocess.TimeoutExpired:
                print("   ⚠️  Piper timeout, using fallback")
                process.kill()
            except Exception as e:
                print(f"   ⚠️  Piper failed: {e}")
        
        # Fallback to eSpeak
        subprocess.run(['espeak-ng', '-v', 'hi', '-s', '150', text], check=False)

    def run(self):
        try:
            while True:
                if self.record_with_vad():
                    if self.use_faster_whisper:
                        # Transcribe using faster-whisper (SPEED-OPTIMIZED)
                        segments, info = self.asr_model.transcribe(
                            self.TEMP_WAV,
                            beam_size=3,
                            language="hi",
                            task="transcribe",
                            initial_prompt="यह हिंदी वॉयस असिस्टेंट है। बंद करो। बंद हो जाओ। समय क्या है। आज कौन सा दिन है। गाना सुनाओ। मजाक सुनाओ। मौसम बताओ। नमस्ते। धन्यवाद। शुक्रिया।",
                            vad_filter=True,
                            condition_on_previous_text=False,
                            best_of=1,
                            temperature=0.0,
                            compression_ratio_threshold=2.4,
                            log_prob_threshold=-1.0,
                            no_speech_threshold=0.6
                        )
                        
                        # Check if Hindi was detected
                        if info.language != "hi":
                            print(f"⚠️  Wrong language: {info.language} (prob: {info.language_probability:.0%})")
                            print(f"   Forcing Hindi retry...")
                            segments, info = self.asr_model.transcribe(
                                self.TEMP_WAV,
                                language="hi",
                                task="transcribe",
                                beam_size=5,
                                initial_prompt="हिंदी हिंदी। बंद करो। बंद हो जाओ। समय क्या है। गाना सुनाओ। मजाक सुनाओ।"
                            )

                        raw_text = " ".join([segment.text for segment in segments]).strip()
                    else:
                        # Fallback to standard whisper
                        result = self.asr_standard.transcribe(self.TEMP_WAV, language="hi", fp16=False)
                        raw_text = result['text'].strip()
                        
                    print(f"📝 Raw transcription: '{raw_text}'")
                    
                    corrected = self.corrector.correct(raw_text)
                    
                    intent, conf = self.intent_classifier.classify(corrected)
                    print(f"🎯 Intent: {intent} (confidence: {conf:.1%})")
                    
                    response = self.generate_response(intent)
                    
                    print(f"💬 Response: {response}")
                    self.speak(response)
                    
                    # Exit commands (no timeout condition)
                    if intent == "stop":
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
