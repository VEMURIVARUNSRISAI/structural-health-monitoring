"""
scripts/verify_dataset.py — draw converted YOLO labels back onto images.
Run:  python scripts/verify_dataset.py
Then open data/verify/ and check every box by eye before training.
"""
import random
from pathlib import Path

import cv2

CLASSES = ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]
COLORS = [(0, 0, 255), (0, 128, 255), (255, 255, 0), (255, 0, 255), (0, 165, 255)]
SPLIT, N = "train", 12

img_dir = Path(f"data/processed/images/{SPLIT}")
out_dir = Path("data/verify"); out_dir.mkdir(exist_ok=True)

imgs = list(img_dir.glob("*.*"))
for img_path in random.sample(imgs, min(N, len(imgs))):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    label = Path(f"data/processed/labels/{SPLIT}/{img_path.stem}.txt")
    for line in label.read_text().splitlines():
        cid, cx, cy, bw, bh = line.split()
        cid = int(cid)
        cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[cid], 3)
        cv2.putText(img, CLASSES[cid], (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[cid], 2)
    cv2.imwrite(str(out_dir / img_path.name), img)

print(f"Annotated samples written to {out_dir}/ — inspect them by eye.")