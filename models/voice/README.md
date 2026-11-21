# JERO i2 Voice Models

This directory contains the local ONNX and CTranslate2 neural models required for JERO's hands-free voice mode. All models run completely offline.

## Required Directory Structure:

```
models/voice/
├── stt/
│   └── faster-whisper-tiny/   (CTranslate2 faster-whisper tiny English model)
├── tts/
│   ├── en_US-lessac-medium.onnx        (Piper TTS voice model)
│   └── en_US-lessac-medium.onnx.json   (Piper TTS voice config)
└── vad/
    └── silero_vad.onnx                 (Silero VAD ONNX model)
```

## How to Download:

1. **Silero VAD (ONNX):**
   Download `silero_vad.onnx` from [snakers4/silero-vad](https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx) and place it in `models/voice/vad/silero_vad.onnx`.

2. **Piper TTS (ONNX):**
   Download `en_US-lessac-medium.onnx` and `en_US-lessac-medium.onnx.json` from [rhasspy/piper releases](https://github.com/rhasspy/piper/releases/download/v1.0.0/voice-en-us-lessac-medium.tar.gz) or HuggingFace [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium) and place them in `models/voice/tts/`.

3. **faster-whisper (tiny English):**
   In development mode, `faster-whisper` will automatically download the `tiny` model to `models/voice/stt/` on first launch. For PyInstaller bundling, ensure the directory `models/voice/stt/faster-whisper-tiny` contains all CTranslate2 model files (`model.bin`, `vocabulary.txt`, `config.json`).
