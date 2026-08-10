# SignBridge — Demo Script

## One-time setup

1. **Install OBS Studio** (free, ~130MB) from <https://obsproject.com>.
   The installer registers the virtual-camera device. OBS itself does **not**
   need to be running — SignBridge writes to the device directly.
   *If SignBridge still reports "VIRTUAL CAMERA OFF", launch OBS once, click
   **Start Virtual Camera**, then close OBS. That completes registration.*

2. `pip install -r requirements.txt`

3. Have the checkpoint at `checkpoints/best_model_transformer_50_v10.pth`.

## Running

```
python src/gui_app.py --checkpoint checkpoints/best_model_transformer_50_v10.pth
```

The window shows the exact frame being sent to Zoom, captions included.
The header line reports the virtual camera status.

Useful flags:
- `--manual` — button-triggered capture instead of continuous (backup for retakes)
- `--no-vcam` — window only, no virtual camera
- `--camera 1` — pick a different webcam

## Connecting to Zoom

Zoom ▸ **Settings** ▸ **Video** ▸ **Camera** ▸ select **OBS Virtual Camera**.

You'll see yourself with the caption bar along the bottom. Everyone in the
meeting sees the same thing — no host permission and no per-meeting setup,
and it works in meetings you merely join.

## How it reads you — sign along to the rhythm

Press **Start signing** once, then follow the cue on screen. It repeats:

```
get ready... 2.0s     <- hands down, prepare the next sign
● SIGN NOW   4.0s     <- sign as soon as this appears
                         (word is recognized, then it loops)
```

Sign **immediately** when SIGN NOW appears. Your natural reaction delay is
what puts the sign about 18% into the window — exactly where it sat when the
training data was recorded. Don't wait for the countdown to run down.

Each word takes ~6 seconds (4s capture + 2s gap), so a three-word sentence is
about 18 seconds. Press **Stop** when the sentence is done.

Why a fixed rhythm rather than auto-detecting when you start: motion-based
triggering measured badly here. Its baseline spans ~6.6s of history, so it
climbs while you sign and fires as the sign *ends* — and `yes` (1.6s of
motion), `no` (1.5s) and `cool` (0.9s) were missed entirely. `--mode motion`
still exists if you want to try it, but the cycle is what works.

After **3.5 seconds** with no accepted word, the sentence is finalized and
stays on screen until you begin the next one.

## Demo sentences

Every word below scores **100% top-1 accuracy** on your own recordings
(`python scripts/rank_signs.py`). Sign them in the order shown.

### Scripted phrases (exact translations)

These ten have hand-written translations in `src/demo_phrases.py`, so they
produce proper English rather than generic rule output — the rules assume an
implied "I", which is wrong when the subject is signed (COUSIN, DOCTOR).
Part-way through, the caption shows the gloss so far ("Cousin play...") and
snaps to the full sentence on the final word.

| Sign in order | Caption produced |
|---|---|
| COUSIN → PLAY → BASKETBALL | The cousin plays basketball. |
| DOCTOR → HELP → COUSIN | The doctor helps the cousin. |
| COUSIN → WALK → LATER | The cousin will walk later. |
| DOCTOR → GO → THURSDAY | The doctor will go on Thursday. |
| TALL → COUSIN → PLAY → BASKETBALL | The tall cousin plays basketball. |
| COUSIN → LIKE → FISH | The cousin likes fish. |
| WHO → PLAY → BASKETBALL | Who plays basketball? |
| COUSIN → GO → BEFORE → THURSDAY | The cousin will go before Thursday. |
| DEAF → COUSIN → PLAY → BASKETBALL | The deaf cousin plays basketball. |
| DOCTOR → PLAY → BASKETBALL → LATER | The doctor will play basketball later. |

Add more by appending to `DEMO_PHRASES` in `src/demo_phrases.py` — keys are
lowercase word tuples in signing order.

### Anything else falls back to the rules

| Sign in order | Caption produced |
|---|---|
| yes → play → basketball | Yes, I play basketball. |
| go → bowling → thursday | I go bowling on Thursday. |
| help → mother → thursday | I help mother on Thursday. |
| who → doctor | Who is the doctor? |
| black → shirt | Black shirt. |

The engine renders **only what you signed**. First person is implied for
verbs (standard when glossing ASL), and articles are left out because ASL
does not mark them — "I give letter", not an invented "a letter".

### Avoid on camera

- **`no`, `pizza`, `bird`, `apple`** — the only four words below 100% (90% each).
  `yes` is safe at 100%; `no` is the weaker half of the pair.

## If a word is misread

The app is deliberately conservative: a word is only accepted at ≥80%
confidence *and* a clear margin over the runner-up, so ambiguous windows are
dropped rather than inserted. The subtitle line tells you when something was
rejected and why.

Just re-sign the word. **Clear** resets the sentence.

If something looks badly wrong, cross-check the same sign against the
known-good CLI: `python src/inference.py --checkpoint <path>`.

## Honest framing for the video

Worth stating plainly if you narrate it: **81% validation accuracy on 50
signs**, measured on a leakage-controlled split. The model is trained largely
on your own recordings, so it recognizes *your* signing well — a stranger
would score lower (~41% signer-disjoint). That's a normal limitation of a
small personal dataset, and saying so is more impressive than overclaiming.
