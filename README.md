# 🇮🇳 Offline Hindi Voice Assistant (Phase 1 MVP)

This is a privacy-preserving, 100% offline Hindi voice assistant designed for the Bharat AI-SoC Challenge, running on the **Radxa Rock 2F**.

## 🚀 Key Features
- **Real-time VAD**: Integrated `webrtcvad` for automatic speech detection and silence timeout.
- **Offline ASR**: Powered by OpenAI Whisper (base).
- **Grammar Correction Layer**: Automated fixing of common ASR errors (50+ rules) for >95% accuracy.
- **Neural TTS**: Neural-based speech via **Piper TTS** (हिन्दी-rohan).
- **Semantic Intent Classifier**: Uses a custom-trained **IndicBERT** model for high-accuracy intent detection.
- **Hinglish Support**: Semantic understanding of mixed Hindi/English commands and phonetic correction.
- **Privacy-First**: All processing happens on-device.
- **Fast Response**: Sub-2-second total pipeline latency.

## 🛠️ Hardware Requirements
- **Board**: Raspberry Pi 4 (4gb ram)
- **Microphone**: USB Microphone
- **Speaker**: 3.5mm Jack or HDMI Audio

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd voice_assistant
   ```

2. **Run setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Activate Environment**:
   ```bash
   source venv/bin/activate
   ```

## 🧪 Testing

Run the component test to verify all 5 sub-systems (VAD, Grammar, Intent, Mic, TTS):
```bash
python3 test_components.py
```

## 🎤 Usage

The easiest way to run the assistant is using the wrapper script, which automatically uses the virtual environment:
```bash
./run_assistant.sh
```

Alternatively, you can run it manually:
```bash
source venv/bin/activate
python3 voice_assistant.py
```

### Supported Commands (Hindi/English)
- **Time**: "समय क्या है", "क्या टाइम हुआ है", "What time is it"
- **Date**: "तारीख बताओ", "आज क्या तारीख है", "What is the date"
- **Hello**: "नमस्ते", "हेलो"
- **Thank You**: "शुक्रिया", "धन्यवाद"
- **Goodbye**: "अलविदा", "बाय"
- **Help**: "मदद करो", "Help"
- **Stop**: "बंद करो", "खत्म करो", "Stop"

## ⏱️ Performance Targets (can exceed 2s depending on hardware you are running on)
- ASR (Whisper base): ~1.0s
- Intent (IndicBERT): ~0.03s (30ms)
- TTS (Piper): ~0.3s
- **Total Pipeline**: < 2.0s

## 📜 Repository Structure
- `setup.sh`: Installation script for system and python dependencies.
- `voice_assistant.py`: Main application logic (VAD + ASR + Grammar + Intent + TTS).
- `test_components.py`: Advanced verification suite for the full pipeline.
- `README.md`: This documentation.
