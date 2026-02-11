# 🇮🇳 Offline Hindi Voice Assistant (SBC Optimized)

This is a privacy-preserving, 100% offline Hindi voice assistant designed for the Bharat AI-SoC Challenge, optimized for the **Radxa Cubie A7A** and for any SBC in general as far as it meets the hardware requirements! But I mentioned this particular SBC for the project as I am using this for the project (Specs: 6GB RAM, 2 Cortex-A76 cores and 6 A55 (efficiency) cores).

## 🚀 Key Features
- **ONNX INT8 Integration**: Highly optimized intent classification for low latency (~10ms).
- **Core Affinity Scaling**: Processes are automatically pinned to Cortex-A76 performance cores.
- **Hardware-Aware Tuning**: Automated CPU governor and memory management optimization.
- **Real-time VAD**: Integrated `webrtcvad` for automatic speech detection.
- **Offline ASR**: Powered by OpenAI Whisper (base).
- **Grammar Correction Layer**: Phonetic-to-Devanagari mapping for >95% accuracy.
- **Neural TTS**: Natural speech synthesis via **Piper TTS** (हिन्दी-rohan).
- **Privacy-First**: All processing happens on-device.

## 🛠️ Hardware Requirements
- **Board Used**: Radxa Cubie A7A 
Minimum Requirements:
- **RAM**: 2GB  
- **Processor**: 2 Cortex-A76 cores(minimum)
- **Microphone**: USB Microphone or any external audio input source
- **Speaker**: 3.5mm Jack / HDMI / USB Audio/ Bluetooth speaker

## 📦 Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd voice_assistant

# 2. Run the consolidated setup script
# This handles system dependencies, ONNX conversion, and hardware tuning
bash setup.sh

# 3. Activate Environment
source venv/bin/activate

# 4. Run Assistant
python3 voice_assistant.py
```

## 🎤 Usage

The easiest way to run the assistant is using the wrapper script:
```bash
./run_assistant.sh
```

### Supported Intents (Hindi/English)
- **Time/Date**: "समय क्या है", "आज क्या तारीख है"
- **Navigation/Control**: "बंद करो", "मदद करो"
- **Interaction**: "नमस्ते", "धन्यवाद", "अलविदा"
- **Entertainment**: "गाना बजाओ", "जोक सुनाओ", "नाचो"
- **Information**: "समाचार बताओ", "मौसम कैसा है"

## ⏱️ Performance Targets
- **ASR Latency**: ~1.5s - 2.0s
- **Intent (ONNX)**: ~0.1s
- **TTS (Piper)**: ~0.1s - 0.2s (for static cached responses only)
- **Total Pipeline**: **1.7s - 2.3s** on Radxa A7A.

Please note that the performance metrics is calculated on the Radxa A7A and may vary on other SBCs!

### IMPORTANT‼️ ⚠️ 🚨
The performance metrics may change on dynamic responses like date and time queries because the responses are generated on the fly and cannot be cached moreover most importantly PiperTTS natural voice generation takes good amount of processing time around 10 seconds. Output comes very fast but that voice generation takes time.

## 💾 Memory Management
- **Footprint**: < 1.5GB RAM
- **Safety**: Assistant will warn/failsafe if available RAM drops below 2.5GB.

## 📜 Repository Structure
- `setup.sh`: Unified installation and optimization script.
- `voice_assistant.py`: Main application logic.
- `convert_indicbert_to_onnx.py`: Quantization script.
- `optimize_system.py`: Hardware-level performance tuning.
- `README.md`: Combined project documentation.

---
*Developed for the Bharat AI-SoC Challenge*
