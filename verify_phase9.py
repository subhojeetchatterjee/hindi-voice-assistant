
import os
import sys
import json
import torch
import re
from voice_assistant import RobustIntentClassifier, AdvancedGrammarCorrector

def test_phase9():
    classifier = RobustIntentClassifier()
    corrector = AdvancedGrammarCorrector()
    
    test_cases = [
        ("\u0622\u067e \u0645\u06cc\u0631\u06cc \u0645\u062f\u062f \u06a9\u0631 \u0633\u06a9\u062a\u06d2 \u06c1\u06cc\u06ba", "help"),       
        ("Aapka Bhaaut Bhaaut Sukriya", "thank_you"), 
        ("Banthkaru Banthkaru", "stop"),           
        ("Band Karob Hai", "stop"),                
        ("Ad, Monsam, Kesar, Arheeda", "weather"),   
        ("Abhi kiya same ho raha hai", "time"),      
        ("Nathke di kao, Natho", "dance"),           
        ("Natchke di Kautura", "dance"),            
        ("Nath ke dhikhao", "dance"),               
        ("Mazaq Karwek, Mazaq", "joke"),            
        ("Samayakya hai, Samaybatau", "time"),       # Phase 12: Joined Time
        ("TK, Mandojao, Guit", "stop"),              # Phase 12: Closure 1
        ("Bantujao, Bantujao, bye", "stop"),         
        ("Bandho bhai", "stop"),                     
        ("Today, what will happen?", "weather"),    
        ("Today's social is...", "news"),           # Phase 13: News Social Artifact
        ("Today's society is called...", "news"),   
        ("Samacar Batau", "news"),                  
        ("Today's topic is called...", "news"),     # Phase 14: News Topic Artifact
        ("What is the use of a knife?", "news"),    # Phase 14: News Knife Artifact
        ("Today's topic is, let us know.", "news"), # Phase 14: News Topic Variation
    ]
    
    print("\n🧠 Testing Phase 9 Robustness...")
    passed = 0
    for text, expected in test_cases:
        corrected = corrector.correct(text)
        intent, conf = classifier.classify(corrected)
        status = "✓" if intent == expected else "✗"
        print(f"  {status} '{text}' -> {corrected} -> {intent} ({conf:.1%})")
        if intent == expected: passed += 1
    
    print(f"\nResult: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

if __name__ == "__main__":
    if test_phase9():
        print("\n✅ All Phase 9 tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some Phase 9 tests failed!")
        sys.exit(1)
