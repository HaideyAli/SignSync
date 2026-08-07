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

## How it reads you

Each sign is captured over a **4-second window**, because that is exactly how
the training data was recorded — the model only ever saw signs in that
geometry. So pace yourself: sign, pause, sign. Roughly one word per 4–5
seconds, meaning a three-word sentence takes about 15 seconds.

After **3.5 seconds** with no new sign, the sentence is finalized and stays on
screen until you start the next one.

## Demo sentences

Every word below scores **100% top-1 accuracy** on your own recordings
(`python scripts/rank_signs.py`). Sign them in the order shown.

| Sign in order | Caption produced |
|---|---|
| yes → play → basketball | Yes, I want to play basketball. |
| go → bowling → thursday | I want to go bowling on Thursday. |
| help → mother → thursday | I want to help mother on Thursday. |
| give → letter → later | I want to give letter later. |
| who → doctor | Who is the doctor? |

### Avoid on camera

- **`no`, `pizza`, `bird`, `apple`** — the only four words below 100% (90% each).
  `yes` is safe at 100%; `no` is the weaker half of the pair.
- **Adjective with no noun** — e.g. `hot → drink` renders as "I want to drink
  hot." The sentence engine is a template system, not a grammar engine.

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
