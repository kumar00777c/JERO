"""
Voice State Machine - Manages the voice conversation states.

States:
- IDLE: Voice window closed or not initialized
- LISTENING: Waiting for user speech (VAD active, STT ready)
- HEARING: User is actively speaking (VAD detected speech)
- THINKING: STT complete, waiting for LLM first token
- SPEAKING: TTS is playing JERO's response
- STANDBY: Only listening for wake phrase, ignores other speech
"""

from enum import Enum, auto
from typing import Callable, Optional
from PySide6.QtCore import QObject, Signal, QTimer
import time


class VoiceState(Enum):
    """Voice conversation states."""
    IDLE = auto()
    LISTENING = auto()
    HEARING = auto()
    THINKING = auto()
    SPEAKING = auto()
    STANDBY = auto()

    @property
    def label(self) -> str:
        """Human-readable state label for UI."""
        return {
            VoiceState.IDLE: "Idle",
            VoiceState.LISTENING: "Listening",
            VoiceState.HEARING: "Hearing",
            VoiceState.THINKING: "Thinking",
            VoiceState.SPEAKING: "Speaking",
            VoiceState.STANDBY: "Standby",
        }[self]


class VoiceStateMachine(QObject):
    """
    State machine for voice conversation flow.

    Emits signals on state transitions for UI updates.
    All state changes go through this machine to ensure valid transitions.
    """

    # Signals
    state_changed = Signal(VoiceState, VoiceState)  # old_state, new_state
    error_occurred = Signal(str)                     # error message

    # Valid transitions: current_state -> set of allowed next states
    VALID_TRANSITIONS = {
        VoiceState.IDLE: {VoiceState.LISTENING, VoiceState.STANDBY},
        VoiceState.LISTENING: {VoiceState.HEARING, VoiceState.THINKING, VoiceState.STANDBY, VoiceState.IDLE},
        VoiceState.HEARING: {VoiceState.LISTENING, VoiceState.THINKING, VoiceState.STANDBY, VoiceState.IDLE},
        VoiceState.THINKING: {VoiceState.SPEAKING, VoiceState.LISTENING, VoiceState.STANDBY, VoiceState.IDLE},
        VoiceState.SPEAKING: {VoiceState.LISTENING, VoiceState.STANDBY, VoiceState.IDLE},
        VoiceState.STANDBY: {VoiceState.LISTENING, VoiceState.IDLE},
    }

    def __init__(self, initial_state: VoiceState = VoiceState.IDLE):
        super().__init__()
        self._state = initial_state
        self._previous_state = None
        self._state_enter_time = time.monotonic()
        self._callbacks: dict[VoiceState, list[Callable]] = {s: [] for s in VoiceState}

    @property
    def state(self) -> VoiceState:
        return self._state

    @property
    def previous_state(self) -> Optional[VoiceState]:
        return self._previous_state

    @property
    def time_in_state(self) -> float:
        """Seconds spent in current state."""
        return time.monotonic() - self._state_enter_time

    def can_transition(self, new_state: VoiceState) -> bool:
        """Check if transition is valid."""
        return new_state in self.VALID_TRANSITIONS.get(self._state, set())

    def transition(self, new_state: VoiceState, force: bool = False) -> bool:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            force: If True, allow invalid transitions (for error recovery)

        Returns:
            True if transition occurred, False if invalid
        """
        if not force and not self.can_transition(new_state):
            self.error_occurred.emit(
                f"Invalid state transition: {self._state.name} -> {new_state.name}"
            )
            return False

        if new_state == self._state:
            return True  # No-op

        old_state = self._state
        self._previous_state = old_state
        self._state = new_state
        self._state_enter_time = time.monotonic()

        # Execute enter callbacks
        for callback in self._callbacks.get(new_state, []):
            try:
                callback()
            except Exception as e:
                self.error_occurred.emit(f"State enter callback error: {e}")

        self.state_changed.emit(old_state, new_state)
        return True

    def on_enter(self, state: VoiceState, callback: Callable):
        """Register a callback to run when entering a state."""
        self._callbacks[state].append(callback)

    def reset(self):
        """Reset to IDLE state."""
        self.transition(VoiceState.IDLE, force=True)

    # Convenience methods for each transition
    def start_listening(self) -> bool:
        return self.transition(VoiceState.LISTENING)

    def start_hearing(self) -> bool:
        return self.transition(VoiceState.HEARING)

    def start_thinking(self) -> bool:
        return self.transition(VoiceState.THINKING)

    def start_speaking(self) -> bool:
        return self.transition(VoiceState.SPEAKING)

    def enter_standby(self) -> bool:
        return self.transition(VoiceState.STANDBY)

    def stop(self) -> bool:
        return self.transition(VoiceState.IDLE)

    def handle_interrupt(self) -> bool:
        """Handle user interrupt during speaking/thinking."""
        if self._state in (VoiceState.SPEAKING, VoiceState.THINKING):
            return self.transition(VoiceState.LISTENING)
        return False

    def handle_standby_phrase(self) -> bool:
        """Handle 'stay on standby' phrase."""
        if self._state in (VoiceState.LISTENING, VoiceState.HEARING, VoiceState.THINKING, VoiceState.SPEAKING):
            return self.transition(VoiceState.STANDBY)
        return False

    def handle_wake_phrase(self) -> bool:
        """Handle 'wake up jero' phrase."""
        if self._state == VoiceState.STANDBY:
            return self.transition(VoiceState.LISTENING)
        return False

    def handle_vad_speech_start(self) -> bool:
        """Handle VAD detecting speech start."""
        if self._state == VoiceState.LISTENING:
            return self.transition(VoiceState.HEARING)
        elif self._state == VoiceState.STANDBY:
            # In standby, VAD triggers wake detection, not hearing
            return False
        return False

    def handle_vad_speech_end(self) -> bool:
        """Handle VAD detecting speech end (silence)."""
        if self._state == VoiceState.HEARING:
            return self.transition(VoiceState.THINKING)
        return False

    def handle_stt_result(self, text: str) -> bool:
        """Handle STT producing text result."""
        if self._state in (VoiceState.HEARING, VoiceState.THINKING):
            if text.strip():
                return self.transition(VoiceState.THINKING)
            else:
                return self.transition(VoiceState.LISTENING)
        return False

    def handle_llm_first_token(self) -> bool:
        """Handle first token from LLM stream."""
        if self._state == VoiceState.THINKING:
            return self.transition(VoiceState.SPEAKING)
        return False

    def handle_tts_finished(self) -> bool:
        """Handle TTS playback complete."""
        if self._state == VoiceState.SPEAKING:
            return self.transition(VoiceState.LISTENING)
        return False