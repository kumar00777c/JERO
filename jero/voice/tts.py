"""
Text-to-Speech Engine - Piper TTS wrapper for natural offline speech.

Features:
- Piper neural TTS (en_US-lessac-medium)
- Sentence-by-sentence streaming for low latency
- Interrupt support (stop speaking mid-sentence)
- Text preprocessing (strip markdown, code blocks, URLs)
"""

import numpy as np
import re
import threading
import queue
from pathlib import Path
from typing import Callable, Optional, Generator, List
import json

from jero.config import voice


class TextPreprocessor:
    """Clean text for TTS: remove markdown, code blocks, URLs, etc."""

    # Patterns to remove/replace
    CODE_BLOCK = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    INLINE_CODE = re.compile(r"`[^`]+`")
    URL = re.compile(r"https?://\S+|www\.\S+")
    MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
    BOLD_ITALIC = re.compile(r"(\*\*|__)(.*?)\1")
    BOLD = re.compile(r"\*\*(.*?)\*\*")
    ITALIC = re.compile(r"__(.*?)__")
    HEADERS = re.compile(r"^#+\s*", re.MULTILINE)
    LIST_MARKERS = re.compile(r"^[\s]*[-*+]\s+", re.MULTILINE)
    EMPHASIS = re.compile(r"\*(.*?)\*")
    STRIKETHROUGH = re.compile(r"~~(.*?)~~")

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean text for TTS."""
        if not text:
            return ""

        # Remove code blocks first (they contain special chars)
        text = cls.CODE_BLOCK.sub(" ", text)
        text = cls.INLINE_CODE.sub(" ", text)

        # Replace URLs with "link"
        text = cls.URL.sub(" link ", text)

        # Markdown links: keep link text
        text = cls.MARKDOWN_LINK.sub(r"\1", text)

        # Bold/italic
        text = cls.BOLD_ITALIC.sub(r"\2", text)
        text = cls.BOLD.sub(r"\1", text)
        text = cls.ITALIC.sub(r"\1", text)
        text = cls.EMPHASIS.sub(r"\1", text)
        text = cls.STRIKETHROUGH.sub(r"\1", text)

        # Headers
        text = cls.HEADERS.sub("", text)

        # List markers
        text = cls.LIST_MARKERS.sub("", text)

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        return text

    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """Split text into sentences for streaming TTS."""
        if not text:
            return []

        # Clean first
        text = cls.clean(text)

        # Split on sentence boundaries (., !, ?) followed by space + capital
        # This is a simple heuristic; for production consider nltk/spacy
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences


class TTSEngine:
    """
    Piper TTS Engine.

    Uses piper-tts Python bindings for ONNX-based neural TTS.
    Supports sentence-by-sentence streaming with interrupt.
    """

    def __init__(
        self,
        model_path: Path = None,
        config_path: Path = None,
        sample_rate: int = 22050,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        sentence_silence: float = 0.2,
    ):
        self.model_path = model_path or voice.TTS_MODEL_PATH
        self.config_path = config_path or voice.TTS_CONFIG_PATH
        self.sample_rate = sample_rate
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        self._piper = None
        self._config = None
        self._lock = threading.Lock()
        self._interrupted = False
        self._current_thread: Optional[threading.Thread] = None

    def _ensure_model(self) -> bool:
        """Load Piper model and config."""
        with self._lock:
            if self._piper is not None:
                return True

            if not self.model_path.exists():
                import sys
                if getattr(sys, "frozen", False):
                    raise FileNotFoundError(
                        f"TTS model not found in bundle: {self.model_path}. "
                        "Ensure models/voice/tts/en_US-lessac-medium.onnx is included in PyInstaller build."
                    )
                raise FileNotFoundError(
                    f"TTS model not found: {self.model_path}. "
                    "Download from https://github.com/rhasspy/piper/releases"
                )

            if not self.config_path.exists():
                raise FileNotFoundError(f"TTS config not found: {self.config_path}")

            try:
                from piper import PiperVoice
            except ImportError:
                raise RuntimeError("piper-tts not installed. Run: pip install piper-tts")

            # Load config
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._config = json.load(f)

            # Load voice
            self._piper = PiperVoice.load(
                model_path=str(self.model_path),
                config_path=str(self.config_path),
                use_cuda=False,  # CPU only
            )
            return True

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio array (blocking).

        Returns:
            float32 audio array at self.sample_rate
        """
        if not self._ensure_model():
            raise RuntimeError("Failed to load TTS model")

        # Clean text
        clean_text = TextPreprocessor.clean(text)
        if not clean_text:
            return np.array([], dtype=np.float32)

        from piper.config import SynthesisConfig
        syn_config = SynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w,
        )

        # Generate audio
        audio_chunks = []
        for chunk in self._piper.synthesize(
            clean_text,
            syn_config=syn_config,
        ):
            audio_chunks.append(chunk.audio_float_array)

        if not audio_chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(audio_chunks)
        return audio.astype(np.float32)

    def synthesize_streaming(
        self,
        text: str,
        on_audio: Callable[[np.ndarray], None],
        on_finished: Callable[[], None] = None,
    ) -> threading.Thread:
        """
        Synthesize and stream sentence by sentence.

        Args:
            text: Full text to speak
            on_audio: Callback for each audio chunk (float32 array)
            on_finished: Callback when all done

        Returns:
            Thread handling the synthesis
        """
        self._interrupted = False

        def worker():
            try:
                if not self._ensure_model():
                    if on_finished:
                        on_finished()
                    return

                sentences = TextPreprocessor.split_sentences(text)
                if not sentences:
                    if on_finished:
                        on_finished()
                    return

                from piper.config import SynthesisConfig
                syn_config = SynthesisConfig(
                    length_scale=self.length_scale,
                    noise_scale=self.noise_scale,
                    noise_w_scale=self.noise_w,
                )

                for i, sentence in enumerate(sentences):
                    if self._interrupted:
                        break

                    clean_sentence = TextPreprocessor.clean(sentence)
                    if not clean_sentence:
                        continue

                    # Synthesize this sentence
                    for chunk in self._piper.synthesize(
                        clean_sentence,
                        syn_config=syn_config,
                    ):
                        if self._interrupted:
                            break
                        if on_audio:
                            on_audio(chunk.audio_float_array)

                    # Add silence between sentences (except last)
                    if not self._interrupted and i < len(sentences) - 1:
                        silence_samples = int(self.sample_rate * self.sentence_silence)
                        silence = np.zeros(silence_samples, dtype=np.float32)
                        if on_audio:
                            on_audio(silence)

            except Exception as e:
                print(f"[TTS] Streaming error: {e}")
            finally:
                if on_finished:
                    on_finished()

        self._current_thread = threading.Thread(target=worker, daemon=True)
        self._current_thread.start()
        return self._current_thread

    def interrupt(self):
        """Stop current synthesis immediately."""
        self._interrupted = True
        if self._current_thread and self._current_thread.is_alive():
            self._current_thread.join(timeout=1.0)

    def is_speaking(self) -> bool:
        """Check if currently synthesizing."""
        return self._current_thread is not None and self._current_thread.is_alive()

    def wait_finished(self, timeout: float = 30.0) -> bool:
        """Wait for current synthesis to complete."""
        if self._current_thread:
            self._current_thread.join(timeout=timeout)
            return not self._current_thread.is_alive()
        return True


class StreamingTTS:
    """
    High-level streaming TTS that integrates with AudioOutput.

    Handles:
    - Sentence splitting
    - AudioOutput coordination
    - Interrupt handling
    - Callbacks for UI state
    """

    def __init__(
        self,
        engine: TTSEngine = None,
        audio_output = None,  # AudioOutput instance
        on_start: Callable[[], None] = None,
        on_finished: Callable[[], None] = None,
        on_error: Callable[[str], None] = None,
    ):
        self.engine = engine or TTSEngine()
        self.audio_output = audio_output
        self.on_start = on_start
        self.on_finished = on_finished
        self.on_error = on_error

        # If audio_output is provided, connect its completion callback
        if self.audio_output:
            self._external_on_finished = self.audio_output.on_finished
            self.audio_output.on_finished = self._on_audio_output_finished

        self._speaking = False
        self._interrupted = False

    def _on_audio_output_finished(self):
        """Called by AudioOutput when buffer drains and playback completes."""
        self._speaking = False
        if self._external_on_finished:
            self._external_on_finished()
        if self.on_finished:
            self.on_finished()

    def speak(self, text: str) -> bool:
        """Start speaking text. Returns True if started."""
        if self._speaking:
            return False

        self._speaking = True
        self._interrupted = False

        def on_audio_chunk(chunk: np.ndarray):
            if self._interrupted or not self._speaking:
                return
            if self.audio_output:
                self.audio_output.write(chunk)

        def on_done():
            if self.audio_output:
                self.audio_output.drain()
            else:
                self._speaking = False
                if self.on_finished:
                    self.on_finished()

        if self.on_start:
            self.on_start()

        # Start audio output if not running
        if self.audio_output and not self.audio_output._playing:
            self.audio_output.start(self.engine.sample_rate)

        self.engine.synthesize_streaming(text, on_audio_chunk, on_done)
        return True

    def interrupt(self):
        """Stop speaking immediately."""
        self._interrupted = True
        self._speaking = False
        self.engine.interrupt()
        if self.audio_output:
            self.audio_output.stop()

    def is_speaking(self) -> bool:
        return self._speaking

    def wait_finished(self, timeout: float = 30.0) -> bool:
        return self.engine.wait_finished(timeout)