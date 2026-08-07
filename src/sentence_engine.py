"""Deterministic ASL-gloss -> English sentence assembly for the 50-word
vocabulary in data/labels_50.json. No ML, no network — pure functions.

refine_fn is an unused hook so a future LLM-polishing pass can be dropped
in later without touching call sites; the project stays rule-based for now
by explicit choice (no API key / network dependency risk during a demo).

Deliberately simple, not a grammar engine: no article/possessive agreement,
and a bare single adjective with no noun reads a little oddly. Avoid
single-adjective-only sequences when picking demo sentences.
"""
from enum import Enum, auto
from typing import Callable


class Role(Enum):
    INTERJECTION = auto()   # yes, no
    QUESTION     = auto()   # what, who
    TIME         = auto()   # before, last, later, thursday, thanksgiving
    VERB         = auto()   # change, cheat, drink, give, go, graduate, help, like, play, walk
    ADJECTIVE    = auto()   # black, cool, dark, deaf, full, hot, many, short, tall, thin
    NOUN         = auto()   # accident, apple, basketball, bed, bird, bowling, candy, computer,
                             # corn, cousin, doctor, dog, family, fish, language, letter, man,
                             # mother, pizza, shirt, woman


WORD_ROLES: dict[str, Role] = {
    "yes": Role.INTERJECTION, "no": Role.INTERJECTION,
    "what": Role.QUESTION, "who": Role.QUESTION,
    "before": Role.TIME, "last": Role.TIME, "later": Role.TIME,
    "thursday": Role.TIME, "thanksgiving": Role.TIME,
    "change": Role.VERB, "cheat": Role.VERB, "drink": Role.VERB, "give": Role.VERB,
    "go": Role.VERB, "graduate": Role.VERB, "help": Role.VERB, "like": Role.VERB,
    "play": Role.VERB, "walk": Role.VERB,
    "black": Role.ADJECTIVE, "cool": Role.ADJECTIVE, "dark": Role.ADJECTIVE,
    "deaf": Role.ADJECTIVE, "full": Role.ADJECTIVE, "hot": Role.ADJECTIVE,
    "many": Role.ADJECTIVE, "short": Role.ADJECTIVE, "tall": Role.ADJECTIVE, "thin": Role.ADJECTIVE,
}

TIME_PHRASES = {"thursday": "on Thursday", "thanksgiving": "on Thanksgiving",
                "later": "later", "before": "before", "last": "last time"}


def _role(word: str) -> Role:
    return WORD_ROLES.get(word, Role.NOUN)


def _noun_phrase(adjs: list[str], nouns: list[str]) -> str:
    if nouns:
        return " ".join(adjs + nouns[:1]) if adjs else nouns[0]
    return adjs[0] if adjs else ""


def _build_core(question, verbs, nouns, adjs, negative: bool) -> str:
    phrase = _noun_phrase(adjs, nouns)
    if question == "what":
        return f"what {phrase} do you want" if phrase else "what do you want"
    if question == "who":
        return f"who is the {phrase}" if phrase else "who is that"

    want = "do not want" if negative else "want"
    if verbs and phrase:
        return f"I {want} to {verbs[0]} {phrase}"
    if verbs:
        return f"I {want} to {verbs[0]}"
    if phrase:
        return f"I {want} {phrase}"
    return f"I {want} that"


def assemble_sentence(words: list[str], refine_fn: Callable[[str], str] | None = None) -> str:
    """Ordered recognized words -> best-guess English sentence."""
    if not words:
        return ""

    remaining = list(words)
    lead     = remaining.pop(0) if remaining and _role(remaining[0]) is Role.INTERJECTION else None
    question = remaining.pop(0) if remaining and _role(remaining[0]) is Role.QUESTION else None

    times = [w for w in remaining if _role(w) is Role.TIME]
    adjs  = [w for w in remaining if _role(w) is Role.ADJECTIVE]
    verbs = [w for w in remaining if _role(w) is Role.VERB]
    nouns = [w for w in remaining if _role(w) not in (Role.TIME, Role.ADJECTIVE, Role.VERB)]

    core = _build_core(question, verbs, nouns, adjs, negative=(lead == "no"))
    if times:
        core += " " + " ".join(TIME_PHRASES.get(t, t) for t in times)

    prefix = {"yes": "Yes, ", "no": "No, "}.get(lead, "")
    sentence = (prefix + core.rstrip())
    sentence = sentence[0].upper() + sentence[1:]   # capitalise only the first char — "I" is
    sentence += "?" if question else "."            # already correctly capitalised in _build_core

    return refine_fn(sentence) if refine_fn else sentence


if __name__ == "__main__":
    cases = [
        (["yes", "pizza", "later"],       "Yes, I want pizza later."),
        (["no", "candy"],                 "No, I do not want candy."),
        (["give", "candy"],               "I want to give candy."),
        (["what", "pizza"],               "What pizza do you want?"),
        (["who", "doctor"],               "Who is the doctor?"),
        (["hot", "pizza", "thursday"],    "I want hot pizza on Thursday."),
        (["deaf", "cousin", "help"],      "I want to help deaf cousin."),
        ([], ""),
    ]
    for words, expected in cases:
        got = assemble_sentence(words)
        status = "OK" if got == expected else "FAIL"
        print(f"{status}: {words} -> {got!r}")
        assert got == expected, f"expected {expected!r}, got {got!r}"
    print(f"\nall {len(cases)} cases passed")
