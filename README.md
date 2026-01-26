# 🇮🇳 Offline Hindi Voice Assistant (Radxa Cubie A7A Optimized)

This is a privacy-preserving, 100% offline Hindi voice assistant designed for the Bharat AI-SoC Challenge, optimized for the **Radxa Cubie A7A** (6GB RAM, Cortex-A76 cores).

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
- **Board**: Radxa Cubie A7A (6GB RAM Recommended)
- **Microphone**: USB Microphone or On-board Audio
- **Speaker**: 3.5mm Jack / HDMI / USB Audio

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
- **Utility**: "अलार्म लगाओ"

## ⏱️ Performance Targets
- **ASR Latency**: ~0.8s - 1.0s
- **Intent (ONNX)**: ~0.01s (10ms)
- **TTS (Piper)**: ~0.2s - 0.3s
- **Total Pipeline**: **0.5s - 0.8s** on Radxa A7A.

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
