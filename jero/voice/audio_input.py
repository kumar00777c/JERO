"""
Audio Input - Microphone capture with Voice Activity Detection (VAD).

Uses sounddevice for audio I/O and Silero VAD (ONNX) for speech detection.
Runs in a separate thread, emits callbacks for audio level, speech start/end.
"""

import numpy as np
import sounddevice as sd
import onnxruntime as ort
from pathlib import Path
import threading
import queue
import time
from typing import Callable, Optional
import collections

from jero.config import voice


class SileroVAD:
    """Silero VAD wrapper using ONNX Runtime."""

    def __init__(self, model_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")

        # ONNX Runtime session
        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        # Get input/output names (Silero VAD has inputs: input, state, sr)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.state_name = inputs[1].name
        # sr may be present as a third input (newer models) or absent (older)
        self.sr_name = None
        if len(inputs) > 2:
            self.sr_name = inputs[2].name
        self.output_name = self.session.get_outputs()[0].name
        self.state_out_name = self.session.get_outputs()[1].name

        # VAD expects 512 samples (32ms @ 16kHz) per inference
        self.window_size = 512
        self.sample_rate = 16000
        self.context_size = 64

        # Internal state (2 layers, 1 batch, 128 hidden units each)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._last_context = np.zeros((1, self.context_size), dtype=np.float32)

    def reset(self):
        """Reset VAD state for new utterance."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._last_context = np.zeros((1, self.context_size), dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """
        Run VAD on audio chunk.

        Args:
            audio_chunk: int16 array of 512 samples (32ms @ 16kHz)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        # Convert int16 to float32 normalized to [-1, 1]
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        # Ensure correct shape: (1, 512)
        if audio_float.ndim == 1:
            audio_float = audio_float.reshape(1, -1)

        # Prepend context as required by Silero VAD v5 architecture
        model_input = np.concatenate([self._last_context, audio_float], axis=1)
        self._last_context = audio_float[:, -self.context_size:]

        # Prepare inputs
        feed = {
            self.input_name: model_input,
            self.state_name: self._state,
        }
        if self.sr_name:
            feed[self.sr_name] = np.array(self.sample_rate, dtype=np.int64)

        # Run inference
        outputs = self.session.run(
            [self.output_name, self.state_out_name],
            feed
        )

        speech_prob = float(outputs[0][0, 0])
        self._state = outputs[1]
        return speech_prob


class AudioInput:
    """
    Microphone audio capture with VAD.

    Runs capture in a background thread, processes in real-time.
    Callbacks (on_level, on_speech_start, on_speech_end, on_audio_chunk)
    are called from the audio thread - use thread-safe patterns.
    """

    def __init__(
        self,
        on_level: Callable[[float], None] = None,
        on_speech_start: Callable[[], None] = None,
        on_speech_end: Callable[[np.ndarray], None] = None,
        on_audio_chunk: Callable[[np.ndarray], None] = None,
        on_error: Callable[[str], None] = None,
    ):
        self.on_level = on_level
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_audio_chunk = on_audio_chunk
        self.on_error = on_error

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=100)
        self._vad: Optional[SileroVAD] = None

        # VAD state tracking
        self._speech_active = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._speech_buffer: list[np.ndarray] = []

        # Volume tracking (RMS)
        self._rms_window = collections.deque(maxlen=10)
        self._current_rms = 0.0

        # Pre-calculate frame counts from config
        self._frames_per_buffer = voice.AUDIO_BLOCKSIZE  # 512 samples
        self._vad_threshold = voice.VAD_THRESHOLD
        self._min_silence_frames = int(voice.VAD_MIN_SILENCE_MS / (1000 * voice.AUDIO_BLOCKSIZE / voice.AUDIO_SAMPLE_RATE))
        self._min_speech_frames = int(voice.VAD_MIN_SPEECH_MS / (1000 * voice.AUDIO_BLOCKSIZE / voice.AUDIO_SAMPLE_RATE))
        self._max_speech_frames = int(15.0 / (voice.AUDIO_BLOCKSIZE / voice.AUDIO_SAMPLE_RATE))  # Max 15s before auto-cut

    def initialize(self) -> bool:
        """Initialize VAD model and audio device."""
        try:
            self._vad = SileroVAD(voice.VAD_MODEL_PATH)
        except Exception as e:
            self._emit_error(f"Failed to load VAD model: {e}")
            return False

        # Check audio device
        try:
            devices = sd.query_devices()
            if voice.AUDIO_DEVICE is not None:
                sd.check_input_settings(device=voice.AUDIO_DEVICE, samplerate=voice.AUDIO_SAMPLE_RATE, channels=voice.AUDIO_CHANNELS)
            else:
                # Use default
                default = sd.default.device[0]
                sd.check_input_settings(device=default, samplerate=voice.AUDIO_SAMPLE_RATE, channels=voice.AUDIO_CHANNELS)
        except Exception as e:
            self._emit_error(f"Audio device error: {e}")
            return False

        return True

    def start(self) -> bool:
        """Start audio capture."""
        if self._running:
            return True

        if self._vad is None:
            if not self.initialize():
                return False

        self._running = True
        self._speech_active = False
        self._silence_frames = 0
        self._speech_frames = 0
        self._speech_buffer.clear()
        self._vad.reset()

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop audio capture."""
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def is_running(self) -> bool:
        return self._running

    def get_current_rms(self) -> float:
        """Get current RMS level (0.0 to 1.0)."""
        return self._current_rms

    def _capture_loop(self):
        """Main audio capture loop (runs in background thread)."""
        def callback(indata, frames, time_info, status):
            # Ignore routine input overflow (occurs momentarily during CPU spikes, e.g. when LLM starts)
            if status and status.input_overflow:
                pass
            elif status:
                self._emit_error(f"Audio callback status: {status}")
            if self._running:
                try:
                    self._audio_queue.put_nowait(indata.copy())
                except queue.Full:
                    pass  # Drop frames if queue full

        try:
            self._stream = sd.InputStream(
                device=voice.AUDIO_DEVICE,
                samplerate=voice.AUDIO_SAMPLE_RATE,
                channels=voice.AUDIO_CHANNELS,
                dtype=voice.AUDIO_DTYPE,
                blocksize=voice.AUDIO_BLOCKSIZE,
                callback=callback,
            )
            self._stream.start()

            while self._running:
                try:
                    chunk = self._audio_queue.get(timeout=0.1)
                    self._process_chunk(chunk)
                except queue.Empty:
                    continue
                except Exception as e:
                    self._emit_error(f"Capture loop error: {e}")
                    break

        except Exception as e:
            self._emit_error(f"Failed to start audio stream: {e}")
        finally:
            if self._stream:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def _process_chunk(self, chunk: np.ndarray):
        """Process a single audio chunk (512 samples)."""
        # chunk shape: (512, 1) or (512,)
        if chunk.ndim > 1:
            chunk = chunk[:, 0]  # Take first channel

        # Calculate RMS for volume visualization
        rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0
        self._rms_window.append(rms)
        self._current_rms = float(np.mean(self._rms_window))

        if self.on_level:
            self.on_level(self._current_rms)

        # Run VAD
        if self._vad is None:
            return

        # VAD expects exactly 512 samples
        if len(chunk) != 512:
            # Pad or truncate
            if len(chunk) < 512:
                chunk = np.pad(chunk, (0, 512 - len(chunk)), mode='constant')
            else:
                chunk = chunk[:512]

        speech_prob = self._vad(chunk)

        # VAD state machine
        if speech_prob >= self._vad_threshold:
            self._silence_frames = 0
            self._speech_frames += 1

            if not self._speech_active and self._speech_frames >= self._min_speech_frames:
                # Speech started
                self._speech_active = True
                self._speech_buffer = [chunk]
                if self.on_speech_start:
                    self.on_speech_start()
            elif self._speech_active:
                self._speech_buffer.append(chunk)
                # Safeguard: if continuous sound exceeds max duration, force finish
                if len(self._speech_buffer) >= self._max_speech_frames:
                    self._speech_active = False
                    if self.on_speech_end:
                        speech_audio = np.concatenate(self._speech_buffer)
                        self.on_speech_end(speech_audio)
                    self._speech_buffer.clear()
                    self._speech_frames = 0
                    self._silence_frames = 0

        else:
            self._speech_frames = 0

            if self._speech_active:
                self._silence_frames += 1
                self._speech_buffer.append(chunk)

                if self._silence_frames >= self._min_silence_frames:
                    # Speech ended
                    self._speech_active = False
                    if self.on_speech_end:
                        # Concatenate all speech chunks
                        speech_audio = np.concatenate(self._speech_buffer)
                        self.on_speech_end(speech_audio)
                    self._speech_buffer.clear()
            else:
                # Not in speech, just pass chunk for potential STT streaming
                pass

        # Forward chunk for streaming STT if callback set
        if self.on_audio_chunk:
            self.on_audio_chunk(chunk)

    def _emit_error(self, msg: str):
        if self.on_error:
            self.on_error(msg)


class AudioOutput:
    """
    Audio playback for TTS output.

    Uses sounddevice OutputStream for low-latency playback.
    Supports interrupt (stop current playback).
    """

    def __init__(self, on_finished: Callable[[], None] = None, on_error: Callable[[str], None] = None):
        self.on_finished = on_finished
        self.on_error = on_error

        self._stream: Optional[sd.OutputStream] = None
        self._playing = False
        self._interrupted = False
        self._draining = False
        self._buffer: queue.Queue = queue.Queue(maxsize=2000)
        self._thread: Optional[threading.Thread] = None

    def start(self, sample_rate: int = 22050) -> bool:
        """Start playback stream."""
        if self._playing:
            self._draining = False
            return True

        try:
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._callback,
            )
            self._stream.start()
            self._playing = True
            self._interrupted = False
            self._draining = False
            return True
        except Exception as e:
            self._emit_error(f"Failed to start audio output: {e}")
            return False

    def drain(self):
        """Signal that no more audio will be written; playback finishes when buffer empties."""
        self._draining = True
        if self._buffer.empty() and self._playing:
            self._playing = False
            self._draining = False
            if self.on_finished:
                self.on_finished()

    def stop(self):
        """Stop playback immediately."""
        self._interrupted = True
        self._playing = False
        self._draining = False
        # Clear buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

    def wait_finished(self, timeout: float = 10.0) -> bool:
        """Wait for playback to finish."""
        if self._thread:
            self._thread.join(timeout=timeout)
        return not self._playing

    def write(self, audio_data: np.ndarray):
        """Queue audio data for playback (float32, -1 to 1)."""
        if not self._playing or self._interrupted:
            return
        # Split into smaller chunks for smooth playback
        chunk_size = 1024
        for i in range(0, len(audio_data), chunk_size):
            if self._interrupted:
                break
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            while self._playing and not self._interrupted:
                try:
                    self._buffer.put(chunk, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _callback(self, outdata, frames, time_info, status):
        # Ignore routine underflows (normal during streaming synthesis pauses)
        if status and status.output_underflow:
            pass
        elif status:
            self._emit_error(f"Audio output status: {status}")

        if self._interrupted:
            outdata.fill(0)
            return

        try:
            chunk = self._buffer.get_nowait()
            if len(chunk) >= frames:
                outdata[:] = chunk[:frames].reshape(-1, 1)
            else:
                outdata[:len(chunk)] = chunk.reshape(-1, 1)
                outdata[len(chunk):] = 0
        except queue.Empty:
            outdata.fill(0)
            if self._playing and not self._interrupted and self._draining:
                # Buffer underrun after producer is finished writing -> playback complete
                self._playing = False
                self._draining = False
                # Close the stream so next turn starts with a fresh hardware clock
                if self._stream:
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                if self.on_finished:
                    self.on_finished()

    def _emit_error(self, msg: str):
        if self.on_error:
            self.on_error(msg)

    def close(self):
        """Clean up resources."""
        self.stop()
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None