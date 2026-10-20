"""Creates the chat window images in static/ by cutting them out of design-references/design-main.png (the robot is not
redrawn - it is the image itself, as the design instructions require):

  static/robot.png        the big robot with the circle behind it and the decorations (brush stroke, two dots), without the
                          greeting card (the card is HTML and overlaps the bottom of the robot); soft edges, so the box
                          cannot be seen on the window background
  static/robot-icon.png   the robot head and chest in a circle - the header icon and the icon next to the bot's answers
  static/send-icon.svg    the ready send button, copied as is

Run:  .venv\\Scripts\\python design-references\\make_assets.py
"""
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "design-references" / "design-main.png"
OUT = ROOT / "static"
OUT.mkdir(exist_ok=True)

im = Image.open(SRC).convert("RGB")

# --- the big robot: with the decorations and the circle behind it, down to where the circle ends (y ~530); the area of the
# greeting card (x 398-838 from y 452) is cleared - the card is HTML, drawn over it, and the circle shows at its sides ---
BOX = (270, 142, 920, 530)  # the header bar ends at y=132 (its shadow a few pixels below): the cut starts under it
CARD = (398 - BOX[0], 452 - BOX[1], 838 - BOX[0], BOX[3] - BOX[1])
robot = im.crop(BOX).convert("RGBA")
# the window interior of the reference is (249.5, 249.8, 250.5) and the window color of the instructions is #F7F8FB: a gain per
# channel, so that the box of the image cannot be seen on the window background
gain = np.array([247 / 249.5, 248 / 249.8, 251 / 250.5, 1.0])
robot = Image.fromarray(np.clip(np.asarray(robot, dtype=float) * gain, 0, 255).astype("uint8"), "RGBA")
w, h = robot.size
SIDE_FEATHER, TOP_FEATHER = 30, 6  # a narrow top edge: the crest of the robot is 13px below it
mask = Image.new("L", (w, h), 0)
d = ImageDraw.Draw(mask)
d.rectangle((SIDE_FEATHER, TOP_FEATHER, w - SIDE_FEATHER, h - SIDE_FEATHER), fill=255)
mask = mask.filter(ImageFilter.GaussianBlur(SIDE_FEATHER / 2))
ImageDraw.Draw(mask).rectangle(CARD, fill=0)
robot.putalpha(mask)
robot.save(OUT / "robot.png", optimize=True)

# --- the icon: the head and the top of the chest, in a circle (a square cut, then a round mask) ---
SIDE = 300
cx, cy = 612, 300
icon = im.crop((cx - SIDE // 2, cy - SIDE // 2, cx + SIDE // 2, cy + SIDE // 2)).convert("RGBA")
icon = icon.resize((256, 256), Image.LANCZOS)
circle = Image.new("L", (256 * 4, 256 * 4), 0)
ImageDraw.Draw(circle).ellipse((0, 0, 256 * 4 - 1, 256 * 4 - 1), fill=255)
icon.putalpha(circle.resize((256, 256), Image.LANCZOS))
icon.save(OUT / "robot-icon.png", optimize=True)
icon.resize((128, 128), Image.LANCZOS).save(OUT / "robot-icon-small.png", optimize=True)  # embedded in the page CSS (the icon repeats in every message)

shutil.copy(ROOT / "design-references" / "send-icon.svg", OUT / "send-icon.svg")
print("static/:", sorted(p.name for p in OUT.iterdir()), "| robot", robot.size, "| icon", icon.size)
