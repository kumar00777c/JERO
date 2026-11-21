"""
JERO Configuration - Centralized settings for voice mode and other features.
All tunable parameters in one place for easy maintenance.
"""

from pathlib import Path
import sys


def _project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def _models_dir() -> Path:
    """Return the models directory (bundled in exe or local in dev)."""
    if getattr(sys, "frozen", False):
        # In PyInstaller bundle: sys._MEIPASS points to extracted bundle
        base = Path(sys._MEIPASS)
    else:
        base = _project_root()
    return base / "models" / "voice"


# ============================================================
# VOICE SETTINGS
# ============================================================

class VoiceConfig:
    """All voice-related configuration in one class."""

    # --- Model Paths ---
    MODELS_DIR = _models_dir()

    # STT: faster-whisper tiny (English only)
    STT_MODEL_DIR = MODELS_DIR / "stt" / "faster-whisper-tiny"
    STT_MODEL_SIZE = "tiny"  # or "base" for better accuracy
    STT_LANGUAGE = "en"      # English only, no auto-detect
    STT_COMPUTE_TYPE = "int8"  # CPU inference: int8, int8_float16, float16, float32
    STT_BEAM_SIZE = 1        # 1 = greedy, faster
    STT_VAD_FILTER = True    # Use faster-whisper's built-in VAD filtering
    STT_VAD_PARAMETERS = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "max_speech_duration_s": 30,
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 200,
    }

    # VAD: Silero VAD (ONNX) - standalone, more control
    VAD_MODEL_PATH = MODELS_DIR / "vad" / "silero_vad.onnx"
    VAD_THRESHOLD = 0.5           # Speech probability threshold (0-1)
    VAD_MIN_SILENCE_MS = 500      # Min silence to consider speech ended
    VAD_MIN_SPEECH_MS = 250       # Min speech duration to accept
    VAD_SAMPLE_RATE = 16000       # Silero expects 16kHz
    VAD_WINDOW_MS = 32            # Window size for VAD (32ms @ 16kHz = 512 samples)

    # Audio Input
    AUDIO_SAMPLE_RATE = 16000     # Match VAD/STT expectation
    AUDIO_CHANNELS = 1            # Mono
    AUDIO_DTYPE = "int16"         # 16-bit PCM
    AUDIO_BLOCKSIZE = 512         # 32ms @ 16kHz
    AUDIO_DEVICE = None           # None = default input device

    # TTS: Piper
    TTS_MODEL_PATH = MODELS_DIR / "tts" / "en_US-ryan-medium.onnx"
    TTS_CONFIG_PATH = MODELS_DIR / "tts" / "en_US-ryan-medium.onnx.json"
    TTS_SAMPLE_RATE = 22050       # Piper output sample rate
    TTS_LENGTH_SCALE = 1.0        # Speed (1.0 = normal, >1 = slower)
    TTS_NOISE_SCALE = 0.667       # Voice variation
    TTS_NOISE_W = 0.8             # Phoneme variation
    TTS_SENTENCE_SILENCE = 0.2    # Seconds of silence between sentences

    # Wake / Standby Phrases (case-insensitive, fuzzy match)
    STANDBY_PHRASES = [
        "stay on standby",
        "stay on stand by",
        "standby",
        "go to standby",
        "enter standby",
    ]
    WAKE_PHRASES = [
        "wake up jero",
        "wake up, jero",
        "hey jero wake up",
        "jero wake up",
    ]
    PHRASE_MATCH_THRESHOLD = 0.8  # Fuzzy match threshold (0-1)

    # State Machine Timing
    STATE_IDLE_TIMEOUT = 30.0     # Seconds before auto-standby (0 = disabled)
    THINKING_MAX_WAIT = 5.0       # Max seconds to wait for first LLM token

    # UI Animation
    ORB_ANIMATION_FPS = 60
    ORB_IDLE_PULSE_SPEED = 1.5    # Radians per second
    ORB_HEARING_PULSE_SPEED = 3.0
    ORB_THINKING_ARC_SPEED = 2.0
    ORB_SPEAKING_PULSE_SPEED = 4.0
    ORB_STANDBY_PULSE_SPEED = 0.5
    ORB_STANDBY_ARC_SPEED = 0.2

    # Caption Settings
    CAPTION_MAX_LINES = 3
    CAPTION_FONT_SIZE = 12
    CAPTION_FADE_MS = 5000        # Auto-hide after this many ms

    # Error Handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0             # Seconds between retries


# ============================================================
# LLM SETTINGS (from local_qwen.py, kept for reference)
# ============================================================

class LLMConfig:
    MODEL_REPO = "SoAIHQ/Qwen3.5-2B-GGUF:Q4_K_M"
    HOST = "127.0.0.1"
    PORT = 1235
    STARTUP_TIMEOUT_SECONDS = 180
    CTX_SIZE = 10240
    TEMPERATURE = 0.85
    MAX_TOKENS = 4000


# ============================================================
# MEMORY SETTINGS
# ============================================================

class MemoryConfig:
    @staticmethod
    def database_path() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent / "memory.db"
        return _project_root() / "memory.db"


# ============================================================
# EXPORT: Easy access to all configs
# ============================================================

voice = VoiceConfig()
llm = LLMConfig()
memory = MemoryConfig()


# ============================================================
# VALIDATION (run at import to catch misconfiguration early)
# ============================================================

def validate_config() -> list[str]:
    """Validate configuration, return list of warnings (empty if OK)."""
    warnings = []

    # Check model directories exist (only in dev mode; in frozen they're bundled)
    if not getattr(sys, "frozen", False):
        if not voice.MODELS_DIR.exists():
            warnings.append(f"Models directory not found: {voice.MODELS_DIR}")
        if not voice.STT_MODEL_DIR.exists():
            warnings.append(f"STT model not found: {voice.STT_MODEL_DIR} (will download on first run)")
        if not voice.VAD_MODEL_PATH.exists():
            warnings.append(f"VAD model not found: {voice.VAD_MODEL_PATH}")
        if not voice.TTS_MODEL_PATH.exists():
            warnings.append(f"TTS model not found: {voice.TTS_MODEL_PATH}")
        if not voice.TTS_CONFIG_PATH.exists():
            warnings.append(f"TTS config not found: {voice.TTS_CONFIG_PATH}")

    return warnings


# Run validation on import (in dev mode)
if not getattr(sys, "frozen", False):
    _warnings = validate_config()
    for w in _warnings:
        print(f"[CONFIG WARNING] {w}")