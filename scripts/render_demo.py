"""Render assets/demo.png from a REAL model run (no hand-edited numbers).

Usage: python scripts/render_demo.py   (needs pillow + cached weights)
"""

import time
import warnings
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from laya_windows import load

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
FONT = "C:/Windows/Fonts/consola.ttf"
BOLD = "C:/Windows/Fonts/consolab.ttf"

state = "We were billed twice for March. Please refund the duplicate today or we cancel."
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                     "sales": "pricing, contracts", "other": "everything else"},
    },
    "urgency": {"type": "score", "instructions": "How urgent is this request?",
                "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}

agent = load(local_files_only=True)
agent.predict(state, questions)  # warm-up
t = time.perf_counter()
answers = agent.predict(state, questions)["answers"]
ms = (time.perf_counter() - t) * 1000
device = "CUDA" if agent.device.type == "cuda" else "CPU"

C = {"bg": (13, 17, 23), "bar": (22, 27, 34), "fg": (230, 237, 243), "dim": (125, 133, 144),
     "green": (63, 185, 80), "blue": (88, 166, 255), "purple": (188, 140, 255),
     "orange": (255, 166, 87), "track": (33, 38, 45)}
W, PAD, LH = 1400, 48, 34
font, bold, small = (ImageFont.truetype(FONT, 22), ImageFont.truetype(BOLD, 22),
                     ImageFont.truetype(FONT, 18))
lines = []  # (segments, bar) ; segment = (text, color, font)


def line(*segs, bar=None):
    lines.append((segs, bar))


line(("PS> ", C["green"], bold), ("laya ", C["blue"], bold),
     ('--state "We were billed twice for March. Please refund..."', C["fg"], font))
line(("    --questions support.json", C["fg"], font))
line()
d = answers["department"]
line(("department  ", C["purple"], bold), ("choice  ", C["dim"], font),
     (f"→ {d['choice']}", C["green"], bold), (f"   confidence {d['confidence']:.2f}", C["dim"], font))
for label, p in d["probabilities"].items():
    line((f"   {label:<10}", C["fg"], font), (f"{p:6.1%}", C["dim"], font), bar=(p, C["green"]))
line()
u = answers["urgency"]
line(("urgency     ", C["purple"], bold), ("score   ", C["dim"], font),
     (f"→ {u['score']:.2f} / 2", C["orange"], bold))
for i, p in u["probabilities"].items():
    line((f"   {['not urgent', 'soon', 'critical'][int(i)]:<10}", C["fg"], font), (f"{p:6.1%}", C["dim"], font),
         bar=(p, C["orange"]))
line()
r = answers["refund"]
line(("refund      ", C["purple"], bold), ("yes/no  ", C["dim"], font),
     (f"→ {'YES' if r['noul'] >= 0.5 else 'NO'}", C["blue"], bold),
     (f"   p(yes) = {r['noul']:.3f}", C["dim"], font))
line()
line((f"» 3 decisions in {ms:.0f} ms on {device} · 0 generated tokens · offline", C["green"], font))

H = 44 + PAD + LH * len(lines) + PAD - 10
img = Image.new("RGB", (W, H), C["bg"])
g = ImageDraw.Draw(img)
g.rectangle([0, 0, W, 44], fill=C["bar"])
g.text((PAD - 28, 22), ">_  laya-windows — PowerShell", font=small, fill=C["dim"], anchor="lm")
for i, glyph in enumerate(["–", "□", "×"]):
    g.text((W - 150 + i * 50, 22), glyph, font=font, fill=C["dim"], anchor="mm")
y = 44 + PAD
for segs, bar in lines:
    x = PAD
    for text, col, f in segs:
        g.text((x, y), text, font=f, fill=col)
        x += g.textlength(text, font=f)
    if bar:
        p, col = bar
        bx, bw = PAD + 300, 520
        g.rounded_rectangle([bx, y + 6, bx + bw, y + 22], 6, fill=C["track"])
        if p > 0.004:
            g.rounded_rectangle([bx, y + 6, bx + max(12, int(bw * p)), y + 22], 6, fill=col)
    y += LH
out = ROOT / "assets" / "demo.png"
img.save(out)
print(out, f"{ms:.0f} ms", device)
