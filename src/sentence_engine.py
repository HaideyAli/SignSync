"""Deterministic ASL-gloss -> English sentence assembly for the 50-word
vocabulary in data/labels_50.json. No ML, no network — pure functions.

Design rule: render only what was signed. An earlier version wrapped
everything in "I want to ...", which invented modality the signer never
expressed — there is no WANT sign in this vocabulary at all, so PLAY
BASKETBALL became "I want to play basketball" rather than "I play
basketball". For a tool whose purpose is faithful interpretation that is
the worst kind of bug: confident and wrong. First person is still implied
for verbs (standard when glossing ASL), and negation is only applied when
NO was actually signed.

Articles and possessives are omitted rather than guessed — ASL does not
mark them, so "I give letter" is left as-is instead of inventing "a"/"the".

refine_fn is an unused hook so a future LLM-polishing pass can be dropped
in later without touching call sites; the project stays rule-based for now
by explicit choice (no API key / network dependency risk during a demo).
"""
from enum import Enum, auto
from typing import Callable

from demo_phrases import DEMO_PHRASES, lookup, partial


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
    """Render the signed content only — no invented verbs or modality."""
    phrase = _noun_phrase(adjs, nouns)
    if question == "what":
        return f"what {phrase}" if phrase else "what"
    if question == "who":
        return f"who is the {phrase}" if phrase else "who"

    if verbs:
        # First person is implied — standard when glossing ASL to English
        neg = "do not " if negative else ""
        return f"I {neg}{verbs[0]} {phrase}".rstrip()
    return phrase          # topic only, e.g. "black shirt"


def assemble_sentence(words: list[str], refine_fn: Callable[[str], str] | None = None) -> str:
    """Ordered recognized words -> best-guess English sentence."""
    if not words:
        return ""

    # Hand-written demo glosses win outright; a phrase still being signed
    # shows its gloss so far rather than a wrong guess from the rules
    phrase = lookup(words) or partial(words)
    if phrase:
        return refine_fn(phrase) if refine_fn else phrase

    remaining = list(words)
    lead     = remaining.pop(0) if remaining and _role(remaining[0]) is Role.INTERJECTION else None
    question = remaining.pop(0) if remaining and _role(remaining[0]) is Role.QUESTION else None

    times = [w for w in remaining if _role(w) is Role.TIME]
    adjs  = [w for w in remaining if _role(w) is Role.ADJECTIVE]
    verbs = [w for w in remaining if _role(w) is Role.VERB]
    nouns = [w for w in remaining if _role(w) not in (Role.TIME, Role.ADJECTIVE, Role.VERB)]

    negative = lead == "no"
    core = _build_core(question, verbs, nouns, adjs, negative)
    if times:
        core = f"{core} {' '.join(TIME_PHRASES.get(t, t) for t in times)}".strip()

    if lead == "yes":
        sentence = f"Yes, {core}" if core else "Yes"
    elif negative:
        # "No, I do not give candy" when a verb carries the negation,
        # otherwise the bare "No candy"
        sentence = (f"No, {core}" if verbs else f"No {core}") if core else "No"
    else:
        sentence = core or ""

    if not sentence:
        return ""
    sentence = sentence[0].upper() + sentence[1:]   # only the first char — "I" is
    sentence += "?" if question else "."            # already capitalised in _build_core

    return refine_fn(sentence) if refine_fn else sentence


if __name__ == "__main__":
    # Nothing here may introduce a word the signer did not sign — in
    # particular no "want", which does not exist in this vocabulary.
    cases = [
        (["yes", "play", "basketball"],   "Yes, I play basketball."),
        (["go", "bowling", "thursday"],   "I go bowling on Thursday."),
        (["help", "mother", "thursday"],  "I help mother on Thursday."),
        (["give", "letter", "later"],     "I give letter later."),
        (["who", "doctor"],               "Who is the doctor?"),
        (["what", "computer"],            "What computer?"),
        (["black", "shirt"],              "Black shirt."),
        (["hot", "pizza", "thursday"],    "Hot pizza on Thursday."),
        (["deaf", "cousin", "help"],      "I help deaf cousin."),
        (["no", "candy"],                 "No candy."),
        (["no", "give", "candy"],         "No, I do not give candy."),
        (["yes"],                         "Yes."),
        (["no"],                          "No."),
        (["short"],                       "Short."),
        ([], ""),
    ]
    for words, expected in cases:
        got = assemble_sentence(words)
        status = "OK" if got == expected else "FAIL"
        print(f"{status}: {words} -> {got!r}")
        assert got == expected, f"expected {expected!r}, got {got!r}"

    for gloss, expected in DEMO_PHRASES.items():
        got = assemble_sentence(list(gloss))
        assert got == expected, f"{gloss}: expected {expected!r}, got {got!r}"
        print(f"OK: {' '.join(g.upper() for g in gloss)} -> {got!r}")

    print(f"\nall {len(cases)} rule cases + {len(DEMO_PHRASES)} demo phrases passed")
