"""Extract {video_id: signer_id} from WLASL_v0.3.json into data/signers.json.

WLASL_v0.3.json lives under data/raw/ which is gitignored and never reaches
Colab, so the mapping is distilled into a small committed file instead.

Usage: python scripts/build_signer_map.py
"""
import json
from pathlib import Path
from collections import Counter

RAW_META  = Path("data/raw/WLASL_v0.3.json")
LANDMARKS = Path("data/landmarks")
OUT_PATH  = Path("data/signers.json")


def main():
    if not RAW_META.exists():
        print(f"{RAW_META} not found — nothing to do")
        return

    # Only keep clips we actually extracted, so the committed file stays small
    have = {f.stem.rsplit("_", 1)[-1] for f in LANDMARKS.glob("*.npy")
            if "_personal_" not in f.name}

    meta = json.load(open(RAW_META))
    signer_of: dict[str, int] = {}
    for entry in meta:
        for inst in entry["instances"]:
            sid = inst.get("signer_id")
            vid = str(inst["video_id"])
            if sid is not None and vid in have:
                signer_of[vid] = int(sid)

    OUT_PATH.write_text(json.dumps(signer_of, indent=0, sort_keys=True))
    counts = Counter(signer_of.values())
    print(f"Saved {len(signer_of)} video->signer entries "
          f"({len(counts)} distinct signers) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
