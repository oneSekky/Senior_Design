"""
label_boxes.py
==============
Simple labeling tool for side-mount box images.

Shows each box_RR_CC.png, you type the letter it represents (or leave blank
and hit Skip).  Progress is saved after every label so you can quit and resume.

Output
  Test_Data/side_mount/labels.csv   — columns: image_path, csv_path, label

Usage
  python label_boxes.py
  python label_boxes.py --img-dir "path/to/split_images"   # override default
"""

import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from PIL import Image, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_ROOT  = SCRIPT_DIR.parent.parent / "Test_Data" / "side_mount"
IMG_DIR    = DATA_ROOT / "split_images"
CSV_DIR    = DATA_ROOT / "split_csvs"
OUT_CSV    = DATA_ROOT / "labels.csv"

DISPLAY_SIZE = 400   # px — image shown at this size


# ── Collect all box images ────────────────────────────────────────────────────

def collect_pairs(img_dir: Path, csv_dir: Path) -> list[dict]:
    """Return list of {img_path, csv_path} for every box that has both files."""
    pairs = []
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


def load_existing(out_csv: Path) -> set[str]:
    """Return set of already-labeled image paths (including skipped)."""
    done = set()
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["image_path"])
    return done


# ── Main app ──────────────────────────────────────────────────────────────────

class LabelApp:
    def __init__(self, root: tk.Tk, pairs: list[dict], out_csv: Path) -> None:
        self.root     = root
        self.pairs    = pairs
        self.out_csv  = out_csv
        self.idx      = 0

        root.title("Box Labeler")
        root.resizable(False, False)
        root.configure(bg="#111")
        root.bind("<Return>", self._submit)
        root.bind("<Escape>", self._skip)

        big_font   = tkfont.Font(family="Helvetica", size=18, weight="bold")
        small_font = tkfont.Font(family="Helvetica", size=12)
        mono_font  = tkfont.Font(family="Courier",   size=14)

        # Progress label
        self._prog_var = tk.StringVar()
        tk.Label(root, textvariable=self._prog_var,
                 bg="#111", fg="#888", font=small_font).pack(pady=(10, 2))

        # Image display
        self._img_label = tk.Label(root, bg="#111")
        self._img_label.pack(padx=20, pady=4)

        # Filename hint
        self._name_var = tk.StringVar()
        tk.Label(root, textvariable=self._name_var,
                 bg="#111", fg="#555", font=small_font).pack()

        # Letter entry
        frame = tk.Frame(root, bg="#111")
        frame.pack(pady=10)
        tk.Label(frame, text="Letter:", bg="#111", fg="white",
                 font=small_font).pack(side=tk.LEFT, padx=(0, 6))
        self._entry = tk.Entry(frame, width=4, font=big_font,
                                justify="center", bg="#222", fg="white",
                                insertbackground="white", relief="flat")
        self._entry.pack(side=tk.LEFT)
        self._entry.focus_set()

        # Buttons
        btn_frame = tk.Frame(root, bg="#111")
        btn_frame.pack(pady=(4, 14))

        btn_cfg = dict(width=10, font=small_font, relief="flat", cursor="hand2")
        tk.Button(btn_frame, text="Label  [↵]", bg="#2a6", fg="white",
                  command=self._submit, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Skip  [Esc]", bg="#444", fg="#aaa",
                  command=self._skip, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Quit", bg="#611", fg="#aaa",
                  command=root.destroy, **btn_cfg).pack(side=tk.LEFT, padx=6)

        # Ensure output CSV has a header
        if not out_csv.exists():
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["image_path", "csv_path", "label"])

        self._show_current()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_current(self) -> None:
        if self.idx >= len(self.pairs):
            self._prog_var.set("All done!")
            self._img_label.configure(image="", text="✓ finished",
                                      fg="white", font=tkfont.Font(size=24))
            self._entry.configure(state="disabled")
            return

        pair = self.pairs[self.idx]
        total = len(self.pairs)
        self._prog_var.set(f"{self.idx + 1} / {total}")
        self._name_var.set(pair["img_path"])
        self._entry.delete(0, tk.END)
        self._entry.focus_set()

        img_path = DATA_ROOT / pair["img_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self._img_label.configure(image=self._photo, text="")
        except Exception:
            self._img_label.configure(image="", text="[image error]", fg="red")

    def _save(self, label: str) -> None:
        pair = self.pairs[self.idx]
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([pair["img_path"], pair["csv_path"], label])

    def _submit(self, _event=None) -> None:
        raw = self._entry.get().strip()
        if not raw:
            return                          # don't save empty entry
        # Accept only a single letter A-Z / a-z
        if len(raw) == 1 and raw.isalpha():
            self._save(raw)
            self.idx += 1
            self._show_current()
        else:
            self._entry.configure(bg="#511")
            self.root.after(300, lambda: self._entry.configure(bg="#222"))

    def _skip(self, _event=None) -> None:
        self._save("SKIP")
        self.idx += 1
        self._show_current()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
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
    print(f"  {len(done)} already labeled — {len(pairs)} remaining")

    if not pairs:
        print("Nothing left to label.")
        return

    root = tk.Tk()
    app  = LabelApp(root, pairs, out_csv)
    root.mainloop()

    # Summary
    if out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        labeled = sum(1 for r in rows if r["label"] != "SKIP")
        skipped = sum(1 for r in rows if r["label"] == "SKIP")
        print(f"\nLabels saved to: {out_csv}")
        print(f"  Labeled: {labeled}   Skipped: {skipped}   Total: {len(rows)}")


if __name__ == "__main__":
    main()
