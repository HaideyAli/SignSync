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


_LENGTHS = sorted({len(g) for g in DEMO_PHRASES}, reverse=True)


def lookup(words: list[str]) -> str | None:
    """Translation for a demo phrase at the END of the signed sequence.

    Matching the tail rather than the whole list means a stray word earlier in
    the session — a misrecognised sign, or a leftover from a previous
    sentence — does not stop the phrase being recognised. Longest phrases are
    tried first so TALL COUSIN PLAY BASKETBALL wins over the shorter
    COUSIN PLAY BASKETBALL nested inside it."""
    key = tuple(w.lower() for w in words)
    for n in _LENGTHS:
        if len(key) >= n and key[-n:] in DEMO_PHRASES:
            return DEMO_PHRASES[key[-n:]]
    return None


def partial(words: list[str]) -> str | None:
    """Progressive text for an unfinished demo phrase, else None.

    Without this the rules run on a half-signed phrase and produce something
    plainly wrong on camera — COUSIN PLAY renders as "I play cousin." before
    resolving. Showing the gloss so far reads as in-progress instead. Also
    matched against the tail, for the same reason as lookup()."""
    key = tuple(w.lower() for w in words)
    # At least two words must be committed. A single trailing word starts
    # several phrases, so a 1-word match would hijack complete sentences the
    # rules handle fine — WHO DOCTOR would render "Doctor..." not
    # "Who is the doctor?".
    for n in range(min(len(key), max(_LENGTHS, default=0)), 1, -1):
        tail = key[-n:]
        if any(g[:n] == tail and len(g) > n for g in DEMO_PHRASES):
            text = " ".join(words[-n:])
            return text[0].upper() + text[1:] + "..."
    return None


if __name__ == "__main__":
    # A stray word earlier must not block the phrase; the longest phrase
    # wins; and a 1-word tail must not hijack sentences the rules own.
    assert lookup(["walk", "doctor", "help", "cousin"]) == \
        DEMO_PHRASES[("doctor", "help", "cousin")]
    assert lookup(["dog", "tall", "cousin", "play", "basketball"]) == \
        DEMO_PHRASES[("tall", "cousin", "play", "basketball")]
    assert lookup(["who", "doctor"]) is None
    assert partial(["doctor", "help"]) == "Doctor help..."
    assert partial(["junk", "doctor", "help"]) == "Doctor help..."
    assert partial(["doctor"]) is None            # 1-word tail must not fire
    assert partial(["who", "doctor"]) is None
    for gloss, text in DEMO_PHRASES.items():
        assert lookup(list(gloss)) == text
    print(f"demo_phrases: {len(DEMO_PHRASES)} phrases + tail-matching checks passed")
