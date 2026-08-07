"""Continuous signing session: accepts recognized words, decides when a
sentence is finished, and produces the caption text.

Acceptance is deliberately conservative. Idle and signing motion overlap
heavily in this data, so a window that merely *looks* active can still reach
the model; the margin gate below is the main defense against a spurious word
landing in a sentence, since a genuine sign scores ~0.99 while an ambiguous
window spreads probability across candidates.
"""
import time

from sentence_engine import assemble_sentence

CONF_THRESHOLD = 0.80    # per CLAUDE.md
MARGIN         = 0.25    # top-1 must beat top-2 by this much
PAUSE_SECS     = 3.5     # silence after which the sentence is considered finished


class SignSession:
    def __init__(self, pause_secs: float = PAUSE_SECS,
                 conf_threshold: float = CONF_THRESHOLD, margin: float = MARGIN):
        self.pause_secs = pause_secs
        self.conf_threshold = conf_threshold
        self.margin = margin
        self.words: list[str] = []
        self.finalized: str = ""
        self.last_word_at: float = 0.0
        self.rejected: str = ""      # why the last candidate was dropped, for the UI

    def add_word(self, word: str, conf: float, runner_up_conf: float = 0.0) -> bool:
        """Offer a recognized word. Returns True if it was accepted into the
        current sentence."""
        if conf < self.conf_threshold:
            self.rejected = f"{word} {conf*100:.0f}% — low confidence"
            return False
        if conf - runner_up_conf < self.margin:
            self.rejected = f"{word} — ambiguous"
            return False
        if self.words and self.words[-1] == word:
            self.rejected = f"{word} — repeat ignored"
            return False

        self.words.append(word)
        self.last_word_at = time.time()
        self.finalized = ""          # a new word means the previous sentence is superseded
        self.rejected = ""
        return True

    def offer(self, results: list[tuple[str, float]]) -> tuple[bool, str]:
        """Offer SignPredictor's ranked output. Returns (accepted, message).
        Keeps the top-1/top-2 unpacking next to the gates that use it."""
        word, conf = results[0]
        runner_up = results[1][1] if len(results) > 1 else 0.0
        if self.add_word(word, conf, runner_up):
            return True, f"recognized: {word}"
        return False, self.rejected

    def tick(self) -> str | None:
        """Call periodically. Returns the sentence exactly once, at the moment
        a pause marks it complete."""
        if not self.words:
            return None
        if time.time() - self.last_word_at < self.pause_secs:
            return None
        sentence = assemble_sentence(self.words)
        self.finalized = sentence
        self.words = []
        return sentence

    @property
    def is_building(self) -> bool:
        return bool(self.words)

    @property
    def caption(self) -> str:
        """Text for the caption bar: the finished sentence, or a live preview
        of the sentence so far while still signing."""
        if self.words:
            return assemble_sentence(self.words)
        return self.finalized

    @property
    def subtitle(self) -> str:
        if self.words:
            return " · ".join(self.words)
        return ""

    def clear(self) -> None:
        self.words = []
        self.finalized = ""
        self.rejected = ""
        self.last_word_at = 0.0
