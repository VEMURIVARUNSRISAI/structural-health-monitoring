"""
scripts/inspect_codebrim.py
Read-only audit of the raw CODEBRIM annotations.
Run BEFORE converting:  python scripts/inspect_codebrim.py
"""
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

RAW_DIR = Path("data/raw")


def is_junk(path: Path) -> bool:
    """macOS zips add a __MACOSX folder full of AppleDouble '._' sidecar files.
    Those are NOT real images or annotations — skip them everywhere."""
    return "__MACOSX" in path.parts or path.name.startswith("._")


def active_labels(obj) -> list[str]:
    """Class labels on one <object>, robust to both annotation styles:
      - VOC style: one or more <name> tags
      - CODEBRIM style: class flags set to '1' (often nested inside <Defect>)
    Searches all descendants so nesting depth does not matter."""
    names = [n.text.strip() for n in obj.iter("name") if n.text and n.text.strip()]
    if names:
        return names
    flags = []
    for el in obj.iter():
        tag = el.tag.lower()
        if tag in ("object", "bndbox", "xmin", "ymin", "xmax", "ymax", "defect"):
            continue                       # skip container and coordinate tags
        if el.text and el.text.strip() == "1":
            flags.append(el.tag)
    return flags


# Collect REAL files only (exclude the __MACOSX junk)
all_xml = [p for p in RAW_DIR.rglob("*.xml") if not is_junk(p)]
all_img = [p for p in (list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.png")))
           if not is_junk(p)]
print(f"Real XML annotation files (excluding __MACOSX): {len(all_xml)}")
print(f"Real image files (excluding __MACOSX):          {len(all_img)}")

# Match images to annotations BY FILENAME STEM, because CODEBRIM stores them
# in SEPARATE folders (original_dataset/images and original_dataset/annotations).
xml_by_stem = {p.stem: p for p in all_xml}
with_ann = sum(1 for img in all_img if img.stem in xml_by_stem)
print(f"Images WITH a matching annotation:              {with_ann}")
print(f"Images with NO annotation (background/negative): {len(all_img) - with_ann}")

class_counter = Counter()      # how often each class name/flag appears
labels_per_box = Counter()     # 1 label per box? 2? 3?
boxes_per_image = Counter()
tag_names_seen = Counter()     # every descendant tag inside <object>, for schema discovery
parse_errors = 0

for xf in all_xml:
    try:
        root = ET.parse(xf).getroot()
    except ET.ParseError:
        parse_errors += 1
        continue
    objects = root.findall(".//object")
    boxes_per_image[len(objects)] += 1
    for obj in objects:
        for el in obj.iter():
            if el is not obj:
                tag_names_seen[el.tag] += 1
        active = active_labels(obj)
        for a in active:
            class_counter[a] += 1
        labels_per_box[len(active)] += 1

print(f"\nXML parse errors: {parse_errors}")
print("\nClass names / flags found and their counts:")
for name, count in class_counter.most_common():
    print(f"  {name:20s} {count}")

print("\nLabels per bounding box (KEY multi-label check):")
for n, count in sorted(labels_per_box.items()):
    print(f"  {n} label(s): {count} boxes")

print("\nBoxes per image distribution (0 = background-only image):")
for n, count in sorted(boxes_per_image.items())[:10]:
    print(f"  {n} boxes: {count} images")

print("\nAll descendant tags seen inside <object> (schema discovery):")
for tag, count in tag_names_seen.most_common():
    print(f"  <{tag}>: {count}")