"""
label_boxes_with_preview.py
============================
CLI labeling tool with automatic image preview.

Shows each box_RR_CC.png in Preview, you type the letter it represents in Terminal.
Progress is saved after every label so you can quit and resume.

Output
  Test_Data/side_mount/labels.csv   — columns: image_path, csv_path, label

Usage
  python3 label_boxes_with_preview.py
  python3 label_boxes_with_preview.py --img-dir "path/to/split_images"   # override default
"""

import argparse
import csv
import sys
import subprocess
import time
from pathlib import Path


# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR.parent.parent / "Test_Data" / "side_mount"
IMG_DIR    = DATA_ROOT / "split_images"
CSV_DIR    = DATA_ROOT / "split_csvs"
OUT_CSV    = DATA_ROOT / "labels.csv"


# ── Collect all box images ────────────────────────────────────────────────────

def collect_pairs(img_dir: Path, csv_dir: Path) -> list:
    """Return list of {img_path, csv_path} for every box that has both files."""
    pairs = []
    if not img_dir.exists():
        print(f"Warning: Image directory not found: {img_dir}")
        return pairs
    
    for img_folder in sorted(img_dir.iterdir()):
        if not img_folder.is_dir():
            continue
        csv_folder = csv_dir / img_folder.name
        for img_path in sorted(img_folder.glob("box_*.png")):
            csv_path = csv_folder / img_path.with_suffix(".csv").name
            pairs.append({
                "img_path": str(img_path.relative_to(DATA_ROOT)),
                "csv_path": str(csv_path.relative_to(DATA_ROOT)) if csv_path.exists() else "",
            })
    return pairs


def load_existing(out_csv: Path) -> set:
    """Return set of already-labeled image paths (including skipped)."""
    done = set()
    if out_csv.exists():
        try:
            with open(out_csv, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    done.add(row["image_path"])
        except Exception as e:
            print(f"Warning reading existing labels: {e}")
    return done


def save_label(out_csv: Path, img_path: str, csv_path: str, label: str) -> None:
    """Save a single label to the CSV file."""
    try:
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([img_path, csv_path, label])
    except Exception as e:
        print(f"Error saving label: {e}")


def open_image_in_preview(img_path: Path) -> None:
    """Open image in macOS Preview."""
    try:
        subprocess.Popen(["open", "-a", "Preview", str(img_path)])
        time.sleep(0.5)  # Give Preview time to open
    except Exception as e:
        print(f"Error opening image in Preview: {e}")
        print(f"Image path: {img_path}")


def close_preview() -> None:
    """Close all Preview windows."""
    try:
        subprocess.run(["killall", "Preview"], stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label boxes with automatic image preview."
    )
    parser.add_argument("--img-dir", default=str(IMG_DIR))
    parser.add_argument("--csv-dir", default=str(CSV_DIR))
    parser.add_argument("--out",     default=str(OUT_CSV))
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    csv_dir = Path(args.csv_dir)
    out_csv = Path(args.out)

    print("Collecting box pairs...")
    all_pairs = collect_pairs(img_dir, csv_dir)
    print(f"  Found {len(all_pairs)} total boxes")

    done = load_existing(out_csv)
    pairs = [p for p in all_pairs if p["img_path"] not in done]
    print(f"  {len(done)} already labeled — {len(pairs)} remaining\n")

    if not pairs:
        print("Nothing left to label.")
        return

    # Ensure output CSV has a header
    if not out_csv.exists():
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["image_path", "csv_path", "label"])

    # Main labeling loop
    for idx, pair in enumerate(pairs):
        total = len(pairs)
        img_path = pair["img_path"]
        csv_path = pair["csv_path"]
        
        full_img_path = DATA_ROOT / img_path
        
        print(f"\n[{idx + 1}/{total}] Opening: {img_path}")
        
        # Open image in Preview
        open_image_in_preview(full_img_path)
        
        print("Image opened in Preview. Enter a letter (A-Z) or press Enter to skip, or 'q' to quit:")
        
        while True:
            try:
                raw = input("> ").strip().upper()
            except KeyboardInterrupt:
                print("\n\nQuitting...")
                close_preview()
                print_summary(out_csv)
                return
            except EOFError:
                print("\n\nQuitting...")
                close_preview()
                print_summary(out_csv)
                return
            
            if raw == 'Q':
                print("\nQuitting...")
                close_preview()
                print_summary(out_csv)
                return
            elif raw == '':
                # Skip
                save_label(out_csv, img_path, csv_path, "SKIP")
                print("Skipped.")
                close_preview()
                break
            elif len(raw) == 1 and raw.isalpha():
                # Valid letter
                save_label(out_csv, img_path, csv_path, raw)
                print(f"Labeled as: {raw}")
                close_preview()
                break
            else:
                print("Invalid input. Please enter a single letter (A-Z) or press Enter to skip.")

    print("\nAll done!")
    close_preview()
    print_summary(out_csv)


def print_summary(out_csv: Path) -> None:
    """Print a summary of labels."""
    if out_csv.exists():
        try:
            with open(out_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            labeled = sum(1 for r in rows if r["label"] != "SKIP")
            skipped = sum(1 for r in rows if r["label"] == "SKIP")
            print(f"\nLabels saved to: {out_csv}")
            print(f"  Labeled: {labeled}   Skipped: {skipped}   Total: {len(rows)}")
        except Exception as e:
            print(f"Error generating summary: {e}")


if __name__ == "__main__":
    main()
