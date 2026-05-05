"""
review_labels.py
================
Review already-labeled box images and delete bad ones.

Shows each labeled image with its assigned letter.  Press:
  Enter / A   — Approve (keep, move to next)
  Delete / D  — Delete (remove from labels.csv, move to next)
  Backspace   — Go back one image
  Escape      — Quit

Deletions are written to labels.csv immediately so progress is never lost.
Only rows with a real letter label are shown (SKIP rows are ignored).

Usage
  cd Signal_Processing_Algorithm/scripts
  python review_labels.py
  python review_labels.py --only A    # review only images labelled A
"""

import argparse
import csv
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

from PIL import Image, ImageTk

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
DATA_ROOT    = SCRIPT_DIR.parent.parent / "Test_Data" / "side_mount"
LABELS_CSV   = DATA_ROOT / "labels.csv"
DISPLAY_SIZE = 400


# ── Load rows to review ───────────────────────────────────────────────────────

def load_rows(only_letter: str | None) -> list[dict]:
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"labels.csv not found: {LABELS_CSV}")
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        label = r["label"].strip().upper()
        if label == "SKIP" or not label or len(label) != 1 or not label.isalpha():
            continue
        if only_letter and label != only_letter.upper():
            continue
        out.append(r)
    return out


def save_csv(all_rows: list[dict]) -> None:
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "csv_path", "label"])
        w.writeheader()
        w.writerows(all_rows)


def load_all_rows() -> list[dict]:
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── App ───────────────────────────────────────────────────────────────────────

class ReviewApp:
    def __init__(self, root: tk.Tk, review_rows: list[dict]) -> None:
        self.root        = root
        self.review_rows = review_rows   # subset being reviewed
        self.idx         = 0
        self.deleted     = 0

        root.title("Label Reviewer")
        root.resizable(False, False)
        root.configure(bg="#111")
        root.bind("<Return>",    self._approve)
        root.bind("a",           self._approve)
        root.bind("A",           self._approve)
        root.bind("<Delete>",    self._delete)
        root.bind("d",           self._delete)
        root.bind("D",           self._delete)
        root.bind("<BackSpace>", self._back)
        root.bind("<Escape>",    lambda _e: root.destroy())

        big_font   = tkfont.Font(family="Helvetica", size=22, weight="bold")
        small_font = tkfont.Font(family="Helvetica", size=12)

        self._prog_var = tk.StringVar()
        tk.Label(root, textvariable=self._prog_var,
                 bg="#111", fg="#888", font=small_font).pack(pady=(10, 2))

        self._img_label = tk.Label(root, bg="#111")
        self._img_label.pack(padx=20, pady=4)

        self._name_var = tk.StringVar()
        tk.Label(root, textvariable=self._name_var,
                 bg="#111", fg="#555", font=small_font).pack()

        self._letter_var = tk.StringVar()
        tk.Label(root, textvariable=self._letter_var,
                 bg="#111", fg="white", font=big_font).pack(pady=(4, 2))

        btn_frame = tk.Frame(root, bg="#111")
        btn_frame.pack(pady=(6, 14))
        btn_cfg = dict(width=12, font=small_font, relief="flat", cursor="hand2")
        tk.Button(btn_frame, text="Approve  [↵]", bg="#2a6", fg="white",
                  command=self._approve, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Delete  [Del]", bg="#a22", fg="white",
                  command=self._delete, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Back  [⌫]", bg="#444", fg="#aaa",
                  command=self._back, **btn_cfg).pack(side=tk.LEFT, padx=6)
        tk.Button(btn_frame, text="Quit", bg="#333", fg="#888",
                  command=root.destroy, **btn_cfg).pack(side=tk.LEFT, padx=6)

        self._show_current()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _show_current(self) -> None:
        if self.idx >= len(self.review_rows):
            self._prog_var.set(f"All done!  Deleted {self.deleted} images.")
            self._img_label.configure(image="", text="Finished",
                                      fg="white", font=tkfont.Font(size=20))
            self._name_var.set("")
            self._letter_var.set("")
            return

        row   = self.review_rows[self.idx]
        total = len(self.review_rows)
        self._prog_var.set(
            f"{self.idx + 1} / {total}   |   deleted this session: {self.deleted}"
        )
        self._name_var.set(row["image_path"])
        self._letter_var.set(f"Label: {row['label'].upper()}")

        img_path = DATA_ROOT / row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.NEAREST)
            self._photo = ImageTk.PhotoImage(img)
            self._img_label.configure(image=self._photo, text="")
        except Exception:
            self._img_label.configure(image="", text="[image missing]", fg="red")

    def _approve(self, _e=None) -> None:
        self.idx += 1
        self._show_current()

    def _delete(self, _e=None) -> None:
        if self.idx >= len(self.review_rows):
            return
        row = self.review_rows[self.idx]
        img_path = row["image_path"]

        # Remove from the review list
        self.review_rows.pop(self.idx)
        self.deleted += 1

        # Remove from labels.csv (reload → filter → save)
        all_rows = load_all_rows()
        all_rows = [r for r in all_rows if r["image_path"] != img_path]
        save_csv(all_rows)

        self._show_current()

    def _back(self, _e=None) -> None:
        if self.idx > 0:
            self.idx -= 1
            self._show_current()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Review only images labelled this letter, e.g. --only A")
    args = parser.parse_args()

    rows = load_rows(args.only)
    if not rows:
        print("No labeled rows to review.")
        return

    filter_msg = f" (letter={args.only.upper()})" if args.only else ""
    print(f"Reviewing {len(rows)} labeled images{filter_msg}")

    root = tk.Tk()
    ReviewApp(root, rows)
    root.mainloop()

    print("Done.")


if __name__ == "__main__":
    main()
