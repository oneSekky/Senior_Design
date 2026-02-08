"""
Convert SVG files to PNG images manually.
This avoids needing Cairo or other complex dependencies.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

def parse_svg_path(d):
    """Extract points from SVG path data."""
    # Simple parser - extracts all coordinate pairs
    coords = re.findall(r"-?\d+\.?\d*", d)
    if len(coords) < 2:
        return []

    floats = [float(c) for c in coords]
    points = []
    for i in range(0, len(floats) - 1, 2):
        points.append((floats[i], floats[i + 1]))

    return points


def svg_to_png_simple(svg_path, output_path, image_size=64):
    """Convert SVG to PNG by drawing the paths."""
    try:
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Get viewBox
        viewbox = root.get("viewBox", "0 0 210 297")
        vb_parts = viewbox.split()
        vb_min_x = float(vb_parts[0])
        vb_min_y = float(vb_parts[1])
        vb_width = float(vb_parts[2])
        vb_height = float(vb_parts[3])

        # Find all paths
        namespace = {"svg": "http://www.w3.org/2000/svg"}
        paths = root.findall(".//svg:path", namespace)
        if not paths:
            paths = root.findall(".//path")

        # Create blank image (white background)
        image = Image.new("L", (image_size, image_size), color=255)
        draw = ImageDraw.Draw(image)

        # Draw all paths
        for path in paths:
            d = path.get("d")
            if not d:
                continue

            points = parse_svg_path(d)
            if len(points) < 2:
                continue

            # Convert SVG coordinates to image coordinates
            img_points = []
            for x, y in points:
                # Normalize to viewBox
                norm_x = (x - vb_min_x) / vb_width
                norm_y = (y - vb_min_y) / vb_height

                # Scale to image size
                img_x = norm_x * image_size
                img_y = norm_y * image_size

                img_points.append((img_x, img_y))

            # Draw lines connecting the points
            if len(img_points) >= 2:
                draw.line(img_points, fill=0, width=2)

        # Save as PNG
        image.save(output_path)
        return True

    except Exception as e:
        print(f"Error converting {svg_path.name}: {e}")
        return False


def main():
    base_dir = Path(__file__).parent
    svg_dir = base_dir / "test_data" / "alphabet_write" / "separated_letters"
    png_dir = base_dir / "test_data" / "alphabet_write" / "letter_images"

    # Create output directory
    png_dir.mkdir(exist_ok=True)

    # Get all SVG files
    svg_files = sorted(
        svg_dir.glob("a_write_*.svg"), key=lambda x: int(x.stem.split("_")[-1])
    )

    print(f"Converting {len(svg_files)} SVG files to PNG...")
    print(f"Output directory: {png_dir}\n")

    success_count = 0
    for svg_file in svg_files:
        output_file = png_dir / f"{svg_file.stem}.png"

        if svg_to_png_simple(svg_file, output_file, image_size=64):
            print(f"✓ {svg_file.name} → {output_file.name}")
            success_count += 1
        else:
            print(f"✗ Failed: {svg_file.name}")

    print(f"\n{'=' * 60}")
    print(f"✓ Converted {success_count}/{len(svg_files)} files")
    print(f"PNG files saved to: {png_dir}")
    print(f"{'=' * 60}")

    print(f"PNG files saved to: {png_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
