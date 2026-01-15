#!/bin/bash

# ============================================================
# 🇮🇳 Offline Hindi Voice Assistant Setup Script
# ============================================================

set -e

echo "🚀 Starting installation..."

# Detect OS
OS="$(uname)"
if [ "$OS" == "Darwin" ]; then
    echo "🍎 detected MacOS"
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install it from https://brew.sh/"
        exit 1
    fi
    echo "📦 Installing system dependencies via Homebrew..."
    brew install espeak-ng portaudio ffmpeg
elif [ "$OS" == "Linux" ]; then
    echo "🐧 detected Linux"
    echo "📦 Installing system dependencies via apt..."
    sudo apt-get update
    sudo apt-get install -y espeak-ng portaudio19-dev ffmpeg python3-dev python3-venv
else
    echo "⚠️ Unsupported OS: $OS"
    exit 1
fi

# 2. Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install Python packages
echo "🛠️ Installing Python dependencies..."
pip install --upgrade pip
# We use torch and transformers for Phase 2 IndicBERT intent recognition
# webrtcvad is added for Phase 3 real-time voice activity detection
# faster-whisper is added for Phase 5 high-speed optimization (Pi 5)
pip install openai-whisper faster-whisper pyaudio numpy piper-tts rapidfuzz torch transformers webrtcvad

# 4. Pre-download Whisper base model
echo "📥 Pre-downloading Whisper 'base' model (optimized for Pi 5)..."
python3 -c "import whisper; whisper.load_model('base')"

# 5. Setup Piper TTS for natural voice
echo "🎤 Setting up Piper TTS for natural voice..."
mkdir -p models/hindi

# Download model
echo "📥 Downloading Hindi neural voice model..."
curl -L --progress-bar \
     "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx" \
     -o models/hindi/hi_IN-rohan-medium.onnx

# Download config
echo "📥 Downloading Hindi neural voice config..."
curl -L --progress-bar \
     "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json" \
     -o models/hindi/hi_IN-rohan-medium.onnx.json

if [ -f models/hindi/hi_IN-rohan-medium.onnx ]; then
    echo "✅ Piper TTS setup complete"
else
    echo "⚠️  Piper model download failed, will use eSpeak-NG as primary"
fi

echo ""
echo "============================================================"
echo "✅ Setup Complete!"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run tests: python3 test_components.py"
echo "3. Run assistant: python3 voice_assistant.py"
echo "============================================================"
