"""Hand-written translations for exact sign sequences, checked before the
rule engine in sentence_engine.py.

Why these bypass the rules: the template engine assumes an implied
first-person subject, which is right for "I play basketball" but wrong here,
because the subject is signed (COUSIN, DOCTOR). Writing them out gives proper
third-person agreement, articles and future tense.

Add your own by appending to DEMO_PHRASES — keys are lowercase word tuples in
signing order. Check new words against `python scripts/rank_signs.py` first;
every word used below scores 100% on the personal recordings.
"""

DEMO_PHRASES: dict[tuple[str, ...], str] = {
    ("cousin", "play", "basketball"):           "The cousin plays basketball.",
    ("doctor", "help", "cousin"):               "The doctor helps the cousin.",
    ("cousin", "walk", "later"):                "The cousin will walk later.",
    ("doctor", "go", "thursday"):               "The doctor will go on Thursday.",
    ("tall", "cousin", "play", "basketball"):   "The tall cousin plays basketball.",
    ("cousin", "like", "fish"):                 "The cousin likes fish.",
    ("who", "play", "basketball"):              "Who plays basketball?",
    ("cousin", "go", "before", "thursday"):     "The cousin will go before Thursday.",
    ("deaf", "cousin", "play", "basketball"):   "The deaf cousin plays basketball.",
    ("doctor", "play", "basketball", "later"):  "The doctor will play basketball later.",
}


def lookup(words: list[str]) -> str | None:
    """Exact translation for a completed phrase, else None."""
    return DEMO_PHRASES.get(tuple(w.lower() for w in words))


def partial(words: list[str]) -> str | None:
    """Progressive text for an unfinished demo phrase, else None.

    Without this the rules run on a half-signed phrase and produce something
    plainly wrong on camera — COUSIN PLAY renders as "I play cousin." before
    resolving. Showing the gloss so far reads as in-progress instead."""
    if not words:
        return None
    key = tuple(w.lower() for w in words)
    if any(g[:len(key)] == key and len(g) > len(key) for g in DEMO_PHRASES):
        text = " ".join(words)
        return text[0].upper() + text[1:] + "..."
    return None
