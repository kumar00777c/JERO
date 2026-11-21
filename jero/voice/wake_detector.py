"""
Wake Word / Standby Phrase Detector.

Detects phrases like "stay on standby" and "wake up jero" in transcribed text.
Uses fuzzy matching for robustness against transcription errors.
"""

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple
from enum import Enum

from jero.config import voice


class PhraseType(Enum):
    STANDBY = "standby"
    WAKE = "wake"
    NONE = "none"


class WakeDetector:
    """
    Detects standby and wake phrases in transcribed text.

    Features:
    - Case-insensitive matching
    - Fuzzy matching for transcription errors
    - Configurable phrases and threshold
    - Returns phrase type and matched phrase
    """

    def __init__(
        self,
        standby_phrases: List[str] = None,
        wake_phrases: List[str] = None,
        threshold: float = 0.8,
    ):
        self.standby_phrases = [p.lower().strip() for p in (standby_phrases or voice.STANDBY_PHRASES)]
        self.wake_phrases = [p.lower().strip() for p in (wake_phrases or voice.WAKE_PHRASES)]
        self.threshold = threshold

        # Compile regex patterns for exact matching (faster)
        self._standby_patterns = [re.compile(re.escape(p), re.IGNORECASE) for p in self.standby_phrases]
        self._wake_patterns = [re.compile(re.escape(p), re.IGNORECASE) for p in self.wake_phrases]

    def detect(self, text: str) -> Tuple[PhraseType, Optional[str], float]:
        """
        Detect standby/wake phrase in text.

        Args:
            text: Transcribed text

        Returns:
            (PhraseType, matched_phrase, confidence)
        """
        if not text:
            return PhraseType.NONE, None, 0.0

        text_lower = text.lower().strip()

        # 1. Try exact substring match first (fastest, most reliable)
        for i, pattern in enumerate(self._standby_patterns):
            if pattern.search(text_lower):
                return PhraseType.STANDBY, self.standby_phrases[i], 1.0

        for i, pattern in enumerate(self._wake_patterns):
            if pattern.search(text_lower):
                return PhraseType.WAKE, self.wake_phrases[i], 1.0

        # 2. Try fuzzy matching on the whole text
        # This catches transcription errors like "stay on stand by" vs "stay on standby"
        best_match = None
        best_score = 0.0
        best_type = PhraseType.NONE

        # Check standby phrases
        for phrase in self.standby_phrases:
            score = self._fuzzy_match(text_lower, phrase)
            if score > best_score:
                best_score = score
                best_match = phrase
                best_type = PhraseType.STANDBY

        # Check wake phrases
        for phrase in self.wake_phrases:
            score = self._fuzzy_match(text_lower, phrase)
            if score > best_score:
                best_score = score
                best_match = phrase
                best_type = PhraseType.WAKE

        if best_score >= self.threshold:
            return best_type, best_match, best_score

        return PhraseType.NONE, None, 0.0

    def _fuzzy_match(self, text: str, phrase: str) -> float:
        """
        Fuzzy match phrase against text using SequenceMatcher.

        Returns similarity ratio (0.0 to 1.0).
        """
        # Use SequenceMatcher for substring matching
        matcher = SequenceMatcher(None, text, phrase)
        # Find longest matching block
        match = matcher.find_longest_match(0, len(text), 0, len(phrase))
        if match.size == 0:
            return 0.0

        # Score based on matched length vs phrase length
        return match.size / len(phrase)

    def is_standby_phrase(self, text: str) -> bool:
        """Quick check if text contains standby phrase."""
        phrase_type, _, _ = self.detect(text)
        return phrase_type == PhraseType.STANDBY

    def is_wake_phrase(self, text: str) -> bool:
        """Quick check if text contains wake phrase."""
        phrase_type, _, _ = self.detect(text)
        return phrase_type == PhraseType.WAKE


# Convenience functions
def create_wake_detector() -> WakeDetector:
    """Create detector with default config."""
    return WakeDetector()