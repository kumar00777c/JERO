"""
Voice Conversation Worker - Orchestrates the complete voice pipeline.

Pipeline:
AudioInput (mic) → VAD → STT → WakeDetector → Brain (LLM) → TTS → AudioOutput

Runs in a QThread, communicates via Qt signals.
"""

import numpy as np
import threading
import time
from pathlib import Path
from typing import Optional, Callable
from PySide6.QtCore import QObject, Signal, Slot, QThread, QTimer

from jero.voice.state_machine import VoiceState, VoiceStateMachine
from jero.voice.audio_input import AudioInput, AudioOutput
from jero.voice.stt import STTEngine, StreamingSTT
from jero.voice.tts import TTSEngine, StreamingTTS, TextPreprocessor
from jero.voice.wake_detector import WakeDetector, PhraseType
from jero.config import voice
from jero.brain.local_qwen import LocalQwenBrain
from jero.memory.database import MemoryDatabase


class VoiceConversationWorker(QObject):
    """
    Main worker for voice conversation.

    Manages:
    - Audio input/output
    - VAD (Voice Activity Detection)
    - STT (Speech-to-Text)
    - Wake/Standby phrase detection
    - LLM generation (same brain as text chat)
    - TTS (Text-to-Speech)
    - State machine

    Emits signals for UI updates.
    """

    # Signals for UI
    state_changed = Signal(str)              # State label
    volume_level = Signal(float)             # 0.0 - 1.0 mic volume
    user_transcript = Signal(str)            # What user said (final)
    user_transcript_partial = Signal(str)    # Partial STT result
    jero_response = Signal(str)              # JERO's response (streaming)
    jero_response_chunk = Signal(str)        # Streaming chunk
    error_occurred = Signal(str)             # Error message
    standby_activated = Signal()             # Entered standby
    wake_activated = Signal()                # Exited standby
    finished = Signal()                      # Worker stopped

    def __init__(self, brain: LocalQwenBrain = None, conversation_id: int = None):
        super().__init__()
        self.brain = brain
        self.conversation_id = conversation_id
        self.memory = brain.memory if brain else MemoryDatabase()

        # Components
        self.state_machine = VoiceStateMachine()
        self.audio_input: Optional[AudioInput] = None
        self.audio_output: Optional[AudioOutput] = None
        self.stt_engine: Optional[STTEngine] = None
        self.streaming_stt: Optional[StreamingSTT] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.streaming_tts: Optional[StreamingTTS] = None
        self.wake_detector: Optional[WakeDetector] = None

        # Threading
        self._running = False
        self._transcribing = False
        self._llm_thread: Optional[threading.Thread] = None
        self._llm_cancel = threading.Event()
        self._current_response = ""
        self._response_lock = threading.Lock()

        # Timing
        self._thinking_start_time = 0.0

        # Connect state machine signals
        self.state_machine.state_changed.connect(self._on_state_changed)
        self.state_machine.error_occurred.connect(self.error_occurred.emit)

    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Audio Input
            self.audio_input = AudioInput(
                on_level=self._on_volume_level,
                on_speech_start=self._on_speech_start,
                on_speech_end=self._on_speech_end,
                on_error=self._on_audio_error,
            )

            # Audio Output
            self.audio_output = AudioOutput(
                on_finished=self._on_tts_finished,
                on_error=self._on_audio_error,
            )

            # STT
            self.stt_engine = STTEngine()
            self.streaming_stt = StreamingSTT(self.stt_engine)

            # TTS
            self.tts_engine = TTSEngine()
            self.streaming_tts = StreamingTTS(
                engine=self.tts_engine,
                audio_output=self.audio_output,
                on_start=self._on_tts_start,
                on_finished=self._on_tts_finished,
                on_error=self._on_audio_error,
            )

            # Wake Detector
            self.wake_detector = WakeDetector()

            # Initialize audio input (loads VAD model)
            if not self.audio_input.initialize():
                return False

            # Pre-warm STT and TTS models in worker thread so first speech doesn't freeze
            try:
                self.stt_engine._ensure_model()
            except Exception as e:
                self.error_occurred.emit(f"STT init: {e}")

            try:
                self.tts_engine._ensure_model()
            except Exception as e:
                self.error_occurred.emit(f"TTS init: {e}")

            return True

        except Exception as e:
            self.error_occurred.emit(f"Initialization failed: {e}")
            return False

    def start(self):
        """Start the voice conversation."""
        if self._running:
            return

        if not self.initialize():
            return

        self._running = True
        self._llm_cancel.clear()

        # Start listening
        self.audio_input.start()
        self.state_machine.start_listening()

    def stop(self):
        """Stop the voice conversation."""
        self._running = False
        self._llm_cancel.set()

        # Stop all components
        if self.audio_input:
            self.audio_input.stop()
        if self.audio_output:
            self.audio_output.close()
        if self.streaming_tts:
            self.streaming_tts.interrupt()
        if self.tts_engine:
            self.tts_engine.interrupt()

        # Cancel LLM generation
        if self._llm_thread and self._llm_thread.is_alive():
            self._llm_thread.join(timeout=2.0)

        self.state_machine.stop()
        self.finished.emit()

    def interrupt(self):
        """Handle user interrupt (stop speaking, start listening)."""
        if self.state_machine.state in (VoiceState.SPEAKING, VoiceState.THINKING):
            # Stop TTS
            if self.streaming_tts:
                self.streaming_tts.interrupt()
            if self.tts_engine:
                self.tts_engine.interrupt()
            if self.audio_output:
                self.audio_output.stop()

            # Cancel LLM
            self._llm_cancel.set()

            # Transition to listening
            self.state_machine.handle_interrupt()

    def mute(self):
        """Mute microphone (pause listening)."""
        if self.audio_input and self.audio_input.is_running():
            self.audio_input.stop()

    def unmute(self):
        """Unmute microphone (resume listening)."""
        if self.audio_input and not self.audio_input.is_running():
            self.audio_input.start()
            if self.state_machine.state == VoiceState.IDLE:
                self.state_machine.start_listening()

    # ========================================================
    # Audio Input Callbacks
    # ========================================================

    def _on_volume_level(self, rms: float):
        """Microphone volume level (0.0 - 1.0)."""
        self.volume_level.emit(rms)

    def _on_speech_start(self):
        """VAD detected speech start."""
        if self.state_machine.state == VoiceState.LISTENING:
            self.state_machine.handle_vad_speech_start()
        elif self.state_machine.state == VoiceState.STANDBY:
            # In standby, we still detect speech for wake phrase
            pass  # STT will process and wake detector will check

    def _on_speech_end(self, speech_audio: np.ndarray):
        """VAD detected speech end - transcribe."""
        if self.state_machine.state == VoiceState.HEARING:
            self.state_machine.handle_vad_speech_end()
            # Transcribe in background
            threading.Thread(
                target=self._transcribe_speech,
                args=(speech_audio,),
                daemon=True
            ).start()
        elif self.state_machine.state == VoiceState.STANDBY:
            # Check for wake phrase
            threading.Thread(
                target=self._transcribe_for_wake,
                args=(speech_audio,),
                daemon=True
            ).start()

    def _on_audio_error(self, msg: str):
        self.error_occurred.emit(f"Audio: {msg}")

    # ========================================================
    # STT Processing
    # ========================================================

    def _transcribe_speech(self, audio: np.ndarray):
        """Transcribe user speech (normal mode)."""
        if self._transcribing:
            return
        self._transcribing = True
        try:
            text = self.stt_engine.transcribe_array(audio, voice.AUDIO_SAMPLE_RATE)
            if text:
                # Emit partial for UI
                self.user_transcript_partial.emit(text)
                # Process through wake detector
                self._process_transcript(text)
        except Exception as e:
            self.error_occurred.emit(f"STT error: {e}")
        finally:
            self._transcribing = False

    def _transcribe_for_wake(self, audio: np.ndarray):
        """Transcribe speech in standby mode (only for wake detection)."""
        try:
            text = self.stt_engine.transcribe_array(audio, voice.AUDIO_SAMPLE_RATE)
            if text:
                phrase_type, matched, confidence = self.wake_detector.detect(text)
                if phrase_type == PhraseType.WAKE:
                    # Wake up!
                    self.wake_activated.emit()
                    self.state_machine.handle_wake_phrase()
                    # Confirm aloud
                    self._speak_confirmation("I'm back.")
        except Exception as e:
            print(f"[Wake STT] Error: {e}")

    def _process_transcript(self, text: str):
        """Process transcribed text: check wake/standby, then send to LLM."""
        # Check for standby phrase
        phrase_type, matched, confidence = self.wake_detector.detect(text)

        if phrase_type == PhraseType.STANDBY:
            self.standby_activated.emit()
            self.state_machine.handle_standby_phrase()
            self._speak_confirmation("Going to standby.")
            return

        if phrase_type == PhraseType.WAKE:
            # Already awake, but confirm
            self.wake_activated.emit()
            self._speak_confirmation("I'm already awake.")
            return

        # Normal conversation: send to LLM
        self.user_transcript.emit(text)
        self._generate_llm_response(text)

    # ========================================================
    # LLM Generation
    # ========================================================

    def _generate_llm_response(self, user_text: str):
        """Generate LLM response in background thread."""
        if not self.brain:
            self.error_occurred.emit("No brain available")
            self.state_machine.transition(VoiceState.LISTENING)
            return

        self._last_user_text = user_text
        self._thinking_start_time = time.monotonic()
        self._llm_cancel.clear()
        self._current_response = ""

        def llm_worker():
            try:
                # Stream from brain
                first_token = True

                def on_chunk(chunk: str):
                    nonlocal first_token
                    if self._llm_cancel.is_set():
                        return False  # Signal to stop
                    with self._response_lock:
                        self._current_response += chunk
                    if first_token and chunk.strip():
                        first_token = False
                        self.state_machine.handle_llm_first_token()
                    self.jero_response_chunk.emit(chunk)
                    return True  # Continue

                self.brain.generate_stream(user_text, on_chunk)

                # Finished
                if not self._llm_cancel.is_set():
                    with self._response_lock:
                        full_response = self._current_response
                    self.jero_response.emit(full_response)
                    self._on_llm_finished(full_response)

            except Exception as e:
                if not self._llm_cancel.is_set():
                    self.error_occurred.emit(f"LLM error: {e}")
                    self.state_machine.transition(VoiceState.LISTENING)

        self._llm_thread = threading.Thread(target=llm_worker, daemon=True)
        self._llm_thread.start()

    def _check_thinking_timeout(self):
        """Check if LLM is taking too long to produce first token."""
        if self.state_machine.state == VoiceState.THINKING:
            with self._response_lock:
                if not self._current_response:
                    # No tokens yet, but don't abort - just notify
                    self.error_occurred.emit("JERO is thinking...")

    def _on_llm_finished(self, response: str):
        """LLM generation complete, start TTS."""
        if self._llm_cancel.is_set():
            return

        if not response.strip():
            self.state_machine.transition(VoiceState.LISTENING)
            return

        # Save to conversation history
        if self.conversation_id:
            self.memory.add_conversation_message(self.conversation_id, "user", self._last_user_text)
            self.memory.add_conversation_message(self.conversation_id, "assistant", response)

        # Start speaking
        self.state_machine.handle_llm_first_token()
        self.streaming_tts.speak(response)

    # ========================================================
    # TTS Callbacks
    # ========================================================

    def _on_tts_start(self):
        """TTS started playing."""
        pass  # State already transitioned by handle_llm_first_token

    def _on_tts_finished(self):
        """TTS playback complete."""
        if self.state_machine.state == VoiceState.SPEAKING:
            self.state_machine.handle_tts_finished()

    def _speak_confirmation(self, text: str):
        """Speak a short confirmation (standby/wake)."""
        if self.streaming_tts:
            self.streaming_tts.speak(text)

    # ========================================================
    # State Machine Callback
    # ========================================================

    def _on_state_changed(self, old_state: VoiceState, new_state: VoiceState):
        """State machine state changed."""
        self.state_changed.emit(new_state.label)
        if new_state == VoiceState.IDLE:
            self._last_user_text = ""

    # Store last user text for saving to memory
    _last_user_text = ""


class VoiceWorkerThread(QThread):
    """QThread wrapper for VoiceConversationWorker."""

    def __init__(self, brain: LocalQwenBrain = None, conversation_id: int = None):
        super().__init__()
        self.worker = VoiceConversationWorker(brain, conversation_id)

    def run(self):
        """Run the worker (starts event loop)."""
        if self.worker.initialize():
            self.worker.start()
            self.exec()  # Start Qt event loop
        else:
            self.worker.finished.emit()

    def stop(self):
        """Stop the worker."""
        try:
            self.worker.stop()
        except Exception:
            pass
        self.quit()
        self.wait(1500)