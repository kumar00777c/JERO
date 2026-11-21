"""
Speech-to-Text Engine - faster-whisper wrapper for English transcription.

Uses faster-whisper (CTranslate2) with tiny English model for fast, accurate
offline transcription. Supports both file and streaming transcription.
"""

import numpy as np
from pathlib import Path
from typing import Callable, Optional, Generator
import threading
import time

from jero.config import voice


class STTEngine:
    """
    faster-whisper STT engine for English.

    Features:
    - English-only model (smaller, more accurate)
    - int8 quantization for CPU speed
    - Built-in VAD filtering option
    - Streaming and file transcription
    """

    def __init__(
        self,
        model_dir: Path = None,
        language: str = "en",
        compute_type: str = "int8",
        beam_size: int = 1,
        vad_filter: bool = True,
        vad_parameters: dict = None,
    ):
        self.model_dir = model_dir or voice.STT_MODEL_DIR
        self.language = language
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.vad_parameters = vad_parameters or voice.STT_VAD_PARAMETERS

        self._model = None
        self._lock = threading.Lock()
        self._downloading = False

    def _model_exists_locally(self) -> bool:
        """Check if model files exist locally (handles nested HF-style dirs)."""
        if not self.model_dir.exists():
            return False
        # Top-level model.bin
        if any(f.name == "model.bin" for f in self.model_dir.iterdir() if f.is_file()):
            return True
        # Nested HuggingFace snapshot structure: .../snapshots/<hash>/model.bin
        for f in self.model_dir.rglob("model.bin"):
            if f.is_file():
                return True
        return False

    def _find_model_path(self) -> str:
        """Find the directory containing model.bin."""
        if (self.model_dir / "model.bin").is_file():
            return str(self.model_dir)
        for f in self.model_dir.rglob("model.bin"):
            if f.is_file():
                return str(f.parent)
        return str(self.model_dir)

    def _ensure_model(self) -> bool:
        """Load model, downloading if necessary (dev mode only)."""
        with self._lock:
            if self._model is not None:
                return True

            if self._downloading:
                # Wait for another thread to finish downloading
                while self._downloading:
                    time.sleep(0.5)
                return self._model is not None

            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")

            # Check if model exists locally (directory must contain model.bin,
            # either at top level or nested in a HuggingFace-style snapshot dir)
            model_has_files = self._model_exists_locally()

            if not model_has_files:
                import sys
                if getattr(sys, "frozen", False):
                    raise FileNotFoundError(
                        f"STT model not found in bundle: {self.model_dir}. "
                        "Ensure models/voice/stt/faster-whisper-tiny is included in PyInstaller build."
                    )
                # In dev mode, download model into the local directory
                print(f"[STT] Downloading faster-whisper {voice.STT_MODEL_SIZE} model to {self.model_dir}...")
                self._downloading = True
                try:
                    self.model_dir.mkdir(parents=True, exist_ok=True)
                    self._model = WhisperModel(
                        voice.STT_MODEL_SIZE,
                        device="cpu",
                        compute_type=self.compute_type,
                        download_root=str(self.model_dir),
                    )
                finally:
                    self._downloading = False
            else:
                self._model = WhisperModel(
                    self._find_model_path(),
                    device="cpu",
                    compute_type=self.compute_type,
                )

            return True

    def transcribe_file(self, audio_path: Path) -> str:
        """Transcribe an audio file."""
        if not self._ensure_model():
            raise RuntimeError("Failed to load STT model")

        segments, info = self._model.transcribe(
            str(audio_path),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        text = " ".join(seg.text for seg in segments)
        return text.strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy audio array.

        Args:
            audio: int16 or float32 array, mono
            sample_rate: Sample rate (must be 16000 for Whisper)

        Returns:
            Transcribed text
        """
        if not self._ensure_model():
            raise RuntimeError("Failed to load STT model")

        # Convert to float32 normalized
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            import scipy.signal
            audio = scipy.signal.resample(
                audio,
                int(len(audio) * 16000 / sample_rate)
            )

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        text = " ".join(seg.text for seg in segments)
        return text.strip()

    def transcribe_streaming(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        on_segment: Callable[[str], None] = None,
    ) -> str:
        """
        Transcribe streaming audio chunks.

        Note: faster-whisper doesn't natively support true streaming.
        This accumulates chunks and transcribes periodically.
        For true streaming, consider using a separate approach.

        Args:
            audio_generator: Yields int16/float32 audio chunks
            on_segment: Callback for each transcribed segment

        Returns:
            Full transcribed text
        """
        if not self._ensure_model():
            raise RuntimeError("Failed to load STT model")

        # Accumulate audio
        audio_chunks = []
        for chunk in audio_generator:
            audio_chunks.append(chunk)

        if not audio_chunks:
            return ""

        audio = np.concatenate(audio_chunks)

        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0

        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters,
        )

        full_text = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                full_text.append(text)
                if on_segment:
                    on_segment(text)

        return " ".join(full_text)


class StreamingSTT:
    """
    True streaming STT using faster-whisper's transcribe with chunked input.

    This uses a sliding window approach - not truly streaming but close enough
    for voice chat. For production streaming, consider whisper.cpp streaming.
    """

    def __init__(self, engine: STTEngine = None):
        self.engine = engine or STTEngine()
        self._buffer = []
        self._buffer_duration = 0.0
        self._last_transcribe = 0.0
        self._min_buffer_sec = 2.0   # Minimum audio before transcribing
        self._max_buffer_sec = 30.0  # Maximum buffer before forced transcribe

    def add_audio(self, chunk: np.ndarray) -> Optional[str]:
        """
        Add audio chunk, return transcribed text if ready.

        Args:
            chunk: int16 array (512 samples @ 16kHz = 32ms)

        Returns:
            Transcribed text if transcription triggered, else None
        """
        self._buffer.append(chunk)
        self._buffer_duration += len(chunk) / 16000.0

        now = time.monotonic()
        should_transcribe = (
            self._buffer_duration >= self._min_buffer_sec and
            (now - self._last_transcribe) >= 1.0  # At most once per second
        ) or self._buffer_duration >= self._max_buffer_sec

        if should_transcribe:
            audio = np.concatenate(self._buffer)
            self._buffer = []
            self._buffer_duration = 0.0
            self._last_transcribe = now

            try:
                text = self.engine.transcribe_array(audio, 16000)
                return text
            except Exception as e:
                print(f"[STT] Transcription error: {e}")
                return None

        return None

    def flush(self) -> Optional[str]:
        """Force transcribe remaining buffer."""
        if not self._buffer:
            return None

        audio = np.concatenate(self._buffer)
        self._buffer = []
        self._buffer_duration = 0.0

        try:
            return self.engine.transcribe_array(audio, 16000)
        except Exception as e:
            print(f"[STT] Flush transcription error: {e}")
            return None

    def reset(self):
        """Clear buffer."""
        self._buffer = []
        self._buffer_duration = 0.0