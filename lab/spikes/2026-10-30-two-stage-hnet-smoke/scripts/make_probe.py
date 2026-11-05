"""Write data/probe_paragraphs.json: 20 paragraphs, each with one made-up word repeated 6-8 times.
Deterministic (fixed seed); the output file is committed so the probe set never drifts."""
import json
import random
from pathlib import Path

rng = random.Random(20261030)
WORDS = ["glorbinfax", "snickwhistle", "trombulate", "quarvexil", "plimbostat", "dravenquip", "zorpleton",
         "mubblecrunch", "thrinkavel", "fladgewick", "ostrimble", "yendarphlox", "crumbazine", "wuffletonk",
         "pyrodexia", "blanterwick", "kroznitude", "sapphindle", "vorbulent", "jemmilard"]
SENT = [
    "The committee agreed that the {w} was far more important than anyone had expected.",
    "Every morning she checked the {w} before opening the shop on the corner.",
    "Nobody could explain why the {w} kept appearing in the old records.",
    "He wrote that a good {w} should be simple, reliable, and a little bit strange.",
    "After the storm, the {w} was the only thing left standing near the river.",
    "They called it a {w}, and the name stuck for the rest of the decade.",
    "In the second chapter the {w} returns, and this time it matters.",
    "A visiting teacher described the {w} as the heart of the whole village.",
    "Students were asked to draw a {w} from memory, and every drawing looked different.",
    "The manual says to clean the {w} twice a year, though most people forget.",
    "Prices for a {w} rose sharply that spring, surprising even the merchants.",
    "She still remembers the first time her grandfather showed her a real {w}.",
]
FILL = [
    "The town had been quiet for weeks, and people spoke softly in the market.",
    "Rain fell on the tin roofs while the lamps were lit one by one.",
    "It was an ordinary year in most ways, with the usual arguments about water and taxes.",
    "Maps of the region were rare, so travelers relied on stories and landmarks.",
]
paras = []
for w in WORDS:
    n = rng.randint(6, 8)
    sents = rng.sample(SENT, n)
    out = [rng.choice(FILL)]
    for s in sents:
        out.append(s.format(w=w))
        if rng.random() < 0.4:
            out.append(rng.choice(FILL))
    text = " ".join(out)
    assert text.count(w) >= 5 and len(text.encode()) <= 1536, (w, len(text))
    paras.append(dict(word=w, text=text))
Path(__file__).resolve().parents[1].joinpath("data/probe_paragraphs.json").write_text(json.dumps(paras, indent=1))
print(len(paras), "paragraphs; max bytes", max(len(p["text"].encode()) for p in paras))
