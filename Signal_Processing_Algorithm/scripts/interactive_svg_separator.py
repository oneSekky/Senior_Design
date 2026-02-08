import math
import re
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import messagebox, ttk


class SVGSeparatorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive SVG Letter Separator")
        self.root.geometry("1200x800")

        # Data
        self.base_dir = Path(__file__).parent
        self.input_dir = self.base_dir / "test_data" / "alphabet_write"
        self.output_dir = self.input_dir / "separated_letters"
        self.output_dir.mkdir(exist_ok=True)

        self.svg_files = [
            self.input_dir / "a1-11.svg",
            self.input_dir / "a12-35.svg",
            self.input_dir / "a36-55.svg",
        ]
        self.current_file_idx = 0
        self.current_letter_number = 1

        # Canvas state
        self.paths = []
        self.canvas_scale = 2.0
        self.selection_rect = None
        self.start_x = None
        self.start_y = None
        self.rect_id = None

        # Setup UI
        self.setup_ui()
        self.load_current_svg()

    def setup_ui(self):
        # Top control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.file_label = ttk.Label(control_frame, text="", font=("Arial", 12, "bold"))
        self.file_label.pack(side=tk.LEFT, padx=10)

        self.letter_label = ttk.Label(
            control_frame, text="Next: a_write_1", font=("Arial", 12)
        )
        self.letter_label.pack(side=tk.LEFT, padx=10)

        ttk.Button(control_frame, text="Next File →", command=self.next_file).pack(
            side=tk.RIGHT, padx=5
        )
        ttk.Button(control_frame, text="← Prev File", command=self.prev_file).pack(
            side=tk.RIGHT, padx=5
        )

        # Instructions
        inst_frame = ttk.Frame(self.root)
        inst_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        instructions = ttk.Label(
            inst_frame,
            text="Instructions: Click and drag to select a letter. Click 'Save Selected' to save it as the next letter. Ignore labels!",
            font=("Arial", 10),
            foreground="blue",
        )
        instructions.pack()

        # Canvas for SVG display
        canvas_frame = ttk.Frame(self.root)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg="white",
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set,
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Bottom button panel
        button_frame = ttk.Frame(self.root)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame, text="Clear Selection", command=self.clear_selection
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="Save Selected Letter",
            command=self.save_selection,
            style="Accent.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame, text="Skip Letter Number", command=self.skip_letter
        ).pack(side=tk.LEFT, padx=5)

        # Status
        self.status_label = ttk.Label(button_frame, text="Ready", foreground="green")
        self.status_label.pack(side=tk.RIGHT, padx=10)

    def load_current_svg(self):
        """Load and display the current SVG file."""
        if self.current_file_idx >= len(self.svg_files):
            messagebox.showinfo("Complete", "All SVG files processed!")
            return

        svg_file = self.svg_files[self.current_file_idx]
        self.file_label.config(text=f"File: {svg_file.name}")

        # Parse SVG
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Get viewBox or dimensions
        viewbox = root.get("viewBox", "0 0 210 297")
        vb_parts = viewbox.split()
        self.vb_min_x = float(vb_parts[0])
        self.vb_min_y = float(vb_parts[1])
        self.vb_width = float(vb_parts[2])
        self.vb_height = float(vb_parts[3])

        # Find all paths
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", namespace)
        if not paths:
            paths = root.findall(".//path")

        self.paths = []
        for path in paths:
            d = path.get("d")
            style = path.get("style", "")
            if d:
                self.paths.append({"d": d, "style": style, "element": path})

        # Clear and redraw canvas
        self.canvas.delete("all")
        self.draw_svg()

        # Update canvas scrollregion
        canvas_width = int(self.vb_width * self.canvas_scale)
        canvas_height = int(self.vb_height * self.canvas_scale)
        self.canvas.config(scrollregion=(0, 0, canvas_width, canvas_height))

        self.status_label.config(
            text=f"Loaded {len(self.paths)} paths from {svg_file.name}"
        )

    def draw_svg(self):
        """Draw all SVG paths on the canvas."""
        for path_data in self.paths:
            self.draw_path(path_data["d"], fill="", outline="black", width=2)

    def draw_path(self, d, fill="", outline="black", width=2):
        """Draw an SVG path on the canvas."""
        # Parse path data and convert to canvas coordinates
        # Simplified: just draw lines between points
        coords = re.findall(r"-?\d+\.?\d*", d)
        if len(coords) < 2:
            return

        floats = [float(c) for c in coords]
        points = []

        for i in range(0, len(floats) - 1, 2):
            x = floats[i]
            y = floats[i + 1]
            # Convert from SVG coordinates to canvas coordinates
            canvas_x = (x - self.vb_min_x) * self.canvas_scale
            canvas_y = (y - self.vb_min_y) * self.canvas_scale
            points.extend([canvas_x, canvas_y])

        if len(points) >= 4:
            self.canvas.create_line(points, fill=outline, width=width, smooth=True)

    def svg_to_canvas(self, x, y):
        """Convert SVG coordinates to canvas coordinates."""
        canvas_x = (x - self.vb_min_x) * self.canvas_scale
        canvas_y = (y - self.vb_min_y) * self.canvas_scale
        return canvas_x, canvas_y

    def canvas_to_svg(self, canvas_x, canvas_y):
        """Convert canvas coordinates to SVG coordinates."""
        x = canvas_x / self.canvas_scale + self.vb_min_x
        y = canvas_y / self.canvas_scale + self.vb_min_y
        return x, y

    def on_mouse_down(self, event):
        """Start drawing selection rectangle."""
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.rect_id:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=2,
            dash=(5, 5),
        )

    def on_mouse_drag(self, event):
        """Update selection rectangle."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_mouse_up(self, event):
        """Finish selection rectangle."""
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)

        # Convert to SVG coordinates
        svg_x1, svg_y1 = self.canvas_to_svg(
            min(self.start_x, cur_x), min(self.start_y, cur_y)
        )
        svg_x2, svg_y2 = self.canvas_to_svg(
            max(self.start_x, cur_x), max(self.start_y, cur_y)
        )

        self.selection_rect = (svg_x1, svg_y1, svg_x2, svg_y2)

        # Highlight selected paths
        selected_count = self.count_selected_paths()
        self.status_label.config(text=f"Selected {selected_count} paths")

    def count_selected_paths(self):
        """Count how many paths are within the selection."""
        if not self.selection_rect:
            return 0

        x1, y1, x2, y2 = self.selection_rect
        count = 0

        for path_data in self.paths:
            if self.path_in_selection(path_data["d"], x1, y1, x2, y2):
                count += 1

        return count

    def path_in_selection(self, d, x1, y1, x2, y2):
        """Check if a path is within the selection rectangle."""
        coords = re.findall(r"-?\d+\.?\d*", d)
        if len(coords) < 2:
            return False

        floats = [float(c) for c in coords]

        # Get path bounding box
        x_coords = floats[0::2]
        y_coords = floats[1::2]

        path_min_x = min(x_coords)
        path_max_x = max(x_coords)
        path_min_y = min(y_coords)
        path_max_y = max(y_coords)

        # Check if path is at least partially within selection
        # (path intersects with selection rectangle)
        return not (
            path_max_x < x1 or path_min_x > x2 or path_max_y < y1 or path_min_y > y2
        )

    def clear_selection(self):
        """Clear the current selection."""
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.selection_rect = None
        self.status_label.config(text="Selection cleared")

    def save_selection(self):
        """Save the selected paths as a new letter SVG."""
        if not self.selection_rect:
            messagebox.showwarning(
                "No Selection",
                "Please select a letter first by dragging a box around it.",
            )
            return

        x1, y1, x2, y2 = self.selection_rect

        # Collect selected paths
        selected_paths = []
        for path_data in self.paths:
            if self.path_in_selection(path_data["d"], x1, y1, x2, y2):
                selected_paths.append(path_data)

        if not selected_paths:
            messagebox.showwarning("No Paths", "No paths found in selection.")
            return

        # Calculate bounding box of selected paths
        all_x = []
        all_y = []

        for path_data in selected_paths:
            coords = re.findall(r"-?\d+\.?\d*", path_data["d"])
            floats = [float(c) for c in coords]
            all_x.extend(floats[0::2])
            all_y.extend(floats[1::2])

        padding = 5
        min_x = min(all_x) - padding
        max_x = max(all_x) + padding
        min_y = min(all_y) - padding
        max_y = max(all_y) + padding

        width = max_x - min_x
        height = max_y - min_y

        # Create new SVG
        svg_content = f'''<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="{width}mm" height="{height}mm"
 viewBox="{min_x} {min_y} {width} {height}"
 xmlns="http://www.w3.org/2000/svg" version="1.1"
 xmlns:xlink="http://www.w3.org/1999/xlink">
<desc>Letter a_{self.current_letter_number} - manually separated</desc>

<g id="letter">
'''

        for path_data in selected_paths:
            svg_content += (
                f'<path\nstyle="{path_data["style"]}"\n d="{path_data["d"]}"\n/>\n'
            )

        svg_content += """</g>
</svg>"""

        # Save file
        output_file = self.output_dir / f"a_write_{self.current_letter_number}.svg"
        with open(output_file, "w") as f:
            f.write(svg_content)

        messagebox.showinfo("Saved", f"Saved as {output_file.name}")

        self.current_letter_number += 1
        self.letter_label.config(text=f"Next: a_write_{self.current_letter_number}")
        self.clear_selection()
        self.status_label.config(
            text=f"Saved! Ready for next letter", foreground="green"
        )
        self.status_label.config(text=f"Saved! Ready for next letter", foreground='green')

    def skip_letter(self):
        """Skip the current letter number."""
        self.current_letter_number += 1
        self.letter_label.config(text=f"Next: a_write_{self.current_letter_number}")
        self.status_label.config(text=f"Skipped to letter {self.current_letter_number}")

    def next_file(self):
        """Load the next SVG file."""
        if self.current_file_idx < len(self.svg_files) - 1:
            self.current_file_idx += 1
            self.load_current_svg()
        else:
            messagebox.showinfo("End", "This is the last file!")

    def prev_file(self):
        """Load the previous SVG file."""
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.load_current_svg()
        else:
            messagebox.showinfo("Start", "This is the first file!")


def main():
    root = tk.Tk()
    app = SVGSeparatorUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
