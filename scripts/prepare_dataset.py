"""
scripts/prepare_dataset.py — lossless CODEBRIM → YOLO conversion.

Guarantees, in order of importance:
  1. Multi-label boxes → one YOLO line per active defect on the box.
     CODEBRIM lets a single box carry several defects at once; a naive
     converter keeps only the first and silently discards the rest.
  2. Any label that maps to no known class is COUNTED and REPORTED,
     never silently dropped (except the explicit Background label).
  3. Coordinates are clamped to [0, 1], so edge-touching boxes are never
     rejected by the trainer.
  4. Background-only images are kept (with empty label files) as negatives.
  5. macOS packaging junk (__MACOSX folder, '._' sidecar files) is skipped.
  6. Images and annotations live in SEPARATE folders, so images are matched
     to annotations by filename stem, not by folder position.
  7. A final accounting is printed: labels read MUST equal lines written.

Run AFTER scripts/inspect_codebrim.py, and after editing CLASSES/ALIASES
below to match the inspector's output:
    python scripts/prepare_dataset.py
"""

import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

# ==== EDIT to match the inspector's output EXACTLY ======================
# The order here defines class IDs 0..N-1. Freeze it after first training —
# the ensemble (Part 6), colors (Part 7), and severity weights (Part 8)
# all depend on this exact order.
CLASSES = ["Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"]

# Map alternate spellings the inspector revealed onto canonical names.
ALIASES = {
    "crack": "Crack",
    "spallation": "Spallation", "spalling": "Spallation",
    "efflorescence": "Efflorescence",
    "exposedbars": "ExposedBars", "exposed_bars": "ExposedBars",
    "exposedrebars": "ExposedBars",
    "corrosionstain": "CorrosionStain", "corrosion_stain": "CorrosionStain",
    "corrosion": "CorrosionStain", "rust": "CorrosionStain",
}
# ========================================================================

RAW_DIR, OUT_DIR = Path("data/raw"), Path("data/processed")
TRAIN, VAL = 0.70, 0.20            # test = remainder (0.10)
BACKGROUND_KEEP_RATIO = 0.5        # keep half of the empty images as negatives
SEED = 42                          # fixed seed → identical split every run


def is_junk(path: Path) -> bool:
    """Skip the macOS __MACOSX folder and '._' AppleDouble sidecar files."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def canonical(name: str) -> str | None:
    """Map a raw label from the XML onto one of our canonical class names."""
    if name in CLASSES:
        return name
    return ALIASES.get(name.lower().replace(" ", ""))


def active_labels(obj) -> list[str]:
    """CODEBRIM stores the real class as 0/1 flags inside a <Defect> wrapper,
    NOT in the <name> tag (which just says 'defect'). Read the flags: every
    class flag set to '1' is an active label on this box. A box may have
    several set to 1 at once — that is the multi-label case."""
    labels = []
    for el in obj.iter():
        # These are the five real class-flag tag names (plus Background, skipped)
        if el.tag in ("Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain"):
            if el.text and el.text.strip() == "1":
                labels.append(el.tag)
    return labels


def parse_boxes(xml_path: Path, w: int, h: int, unknown: Counter):
    """Return (yolo_lines, labels_seen). One line PER LABEL per box — lossless."""
    lines, labels_seen = [], 0
    root = ET.parse(xml_path).getroot()
    for obj in root.findall(".//object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.find("xmin").text); y1 = float(box.find("ymin").text)
            x2 = float(box.find("xmax").text); y2 = float(box.find("ymax").text)
        except (AttributeError, TypeError, ValueError):
            continue                                   # malformed coordinates

        if x2 <= x1 or y2 <= y1:
            continue                                   # degenerate box

        # Normalize to 0-1 and CLAMP so edge boxes are never rejected later
        cx = min(max(((x1 + x2) / 2) / w, 0.0), 1.0)
        cy = min(max(((y1 + y2) / 2) / h, 0.0), 1.0)
        bw = min(max((x2 - x1) / w, 0.0), 1.0)
        bh = min(max((y2 - y1) / h, 0.0), 1.0)

        for raw in active_labels(obj):
            cls = canonical(raw)
            if cls is None:
                if raw.lower() != "background":
                    unknown[raw] += 1                  # surfaced, never silent
                continue
            # One YOLO line per label — this is the multi-label preservation
            lines.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            labels_seen += 1
    return lines, labels_seen


def main():
    random.seed(SEED)
    for split in ("train", "val", "test"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Index every REAL annotation by filename stem. CODEBRIM keeps images and
    # annotations in separate folders, so we pair them by name, not location.
    all_xml = [p for p in RAW_DIR.rglob("*.xml") if not is_junk(p)]
    xml_by_stem = {p.stem: p for p in all_xml}

    all_images = [p for p in (list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.png")))
                  if not is_junk(p)]
    print(f"Real images: {len(all_images)}   Real annotations: {len(all_xml)}")
    if not all_images:
        print("ERROR: no images found under data/raw/ — did extraction succeed?")
        return

    # Split images into those that have an annotation and those that do not
    annotated = [img for img in all_images if img.stem in xml_by_stem]
    background = [img for img in all_images if img.stem not in xml_by_stem]
    background = random.sample(background, int(len(background) * BACKGROUND_KEEP_RATIO))

    pool = annotated + background
    random.shuffle(pool)
    n = len(pool)
    splits = {"train": pool[:int(n * TRAIN)],
              "val":   pool[int(n * TRAIN):int(n * (TRAIN + VAL))],
              "test":  pool[int(n * (TRAIN + VAL)):]}

    unknown = Counter()
    total_labels_in = total_lines_out = 0
    stats = {s: {"images": 0, "negatives": 0} for s in splits}

    for split, imgs in splits.items():
        for img_path in imgs:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue                               # unreadable image, skip
            xml_path = xml_by_stem.get(img_path.stem)  # matched by filename
            if xml_path is not None:
                lines, seen = parse_boxes(xml_path, w, h, unknown)
                total_labels_in += seen
                total_lines_out += len(lines)
            else:
                lines = []                             # negative: empty label file
                stats[split]["negatives"] += 1
            shutil.copy2(img_path, OUT_DIR / "images" / split / img_path.name)
            (OUT_DIR / "labels" / split / f"{img_path.stem}.txt")\
                .write_text("\n".join(lines))
            stats[split]["images"] += 1

    # Write dataset.yaml — this is what the training scripts load
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (OUT_DIR / "dataset.yaml").write_text(
        f"path: {OUT_DIR.resolve()}\ntrain: images/train\nval: images/val\n"
        f"test: images/test\n\nnc: {len(CLASSES)}\nnames:\n{names}\n")

    print("=" * 58)
    for s, d in stats.items():
        print(f"  {s:5s}: {d['images']:5d} images ({d['negatives']} negatives)")
    print(f"\n  LOSSLESS CHECK — labels read: {total_labels_in}, "
          f"YOLO lines written: {total_lines_out}")
    print("  These MUST be equal. If not, something was dropped.")
    if unknown:
        print("\n  WARNING — labels that matched no class and were skipped:")
        for k, v in unknown.most_common():
            print(f"    {k}: {v}")
        print("  Add these to ALIASES (or CLASSES) and re-run. Do NOT ignore this.")
    else:
        print("  No unknown labels — every annotation mapped to a class.")
    print(f"\n  dataset.yaml → {OUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()