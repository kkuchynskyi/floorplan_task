from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pipeline_panel(array, label, size=(400, 400)):
    if array.dtype == bool:
        array = array.astype(np.uint8) * 255
    elif array.ndim == 2 and array.max() <= 1:
        array = array.astype(np.uint8) * 255

    image = Image.fromarray(array.astype(np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size)

    image = image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("Arial.ttf", 25)
    padding = 8
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_position = (padding, size[1] - (text_bbox[3] - text_bbox[1]) - padding)
    text_bbox = draw.textbbox(text_position, label, font=font)
    background_bbox = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)
    draw.rectangle(background_bbox, fill=(0, 0, 0, 120))
    draw.text(text_position, label, font=font, fill=(0, 255, 0))
    return Image.alpha_composite(image, overlay).convert("RGB")


def save_pipeline_image(output_path, steps, panel_size=(400, 400), columns=2):
    rows = int(np.ceil(len(steps) / columns))
    pipeline = Image.new("RGB", (panel_size[0] * columns, panel_size[1] * rows), "white")
    for index, (label, panel) in enumerate(steps):
        x = (index % columns) * panel_size[0]
        y = (index // columns) * panel_size[1]
        pipeline.paste(pipeline_panel(panel, label, panel_size), (x, y))
    pipeline.save(output_path)


def main():
    input_dir = Path("data/inputs")
    output_dir = Path("data/output")

    for image_path in sorted(input_dir.glob("*.webp")):
        # 1. Load image and convert to HSV
        rgb = np.array(Image.open(image_path).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # 2. Detect foreground
        foreground = ((val < 247) | (sat > 10)).astype(np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # 3. Find and fill the floorplan region
        n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        floorplan = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        floorplan = cv2.morphologyEx(
            floorplan.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
        ).astype(bool)
        padded = np.pad(floorplan.astype(np.uint8), 1)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 1)
        floorplan = floorplan | (flood[1:-1, 1:-1] == 0)

        # 4. Extract outer floorplan boundary
        outer = floorplan & ~cv2.erode(
            floorplan.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        ).astype(bool)

        # 5. Detect wall candidates
        wall_candidates = ((val > 218) & (sat < 48) & floorplan).astype(np.uint8)
        wall_candidates = cv2.morphologyEx(
            wall_candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        # Use wall candidates directly
        walls = wall_candidates.astype(bool)

        # 6. Build final border mask
        borders = (walls | outer).astype(np.uint8)
        borders = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (43, 5)))
        borders = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 43)))
        borders = cv2.dilate(borders, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        borders = (borders > 0) & floorplan

        # 7. Overlay final border mask
        border_overlay = rgb.copy()
        border_overlay[borders] = (255, 0, 0)

        # Save border mask
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        save_pipeline_image(
            image_output_dir / "pipeline.png",
            [
                ("1. Load image and convert to HSV", rgb),
                ("2. Detect foreground", foreground),
                ("3. Find and fill floorplan", floorplan),
                ("4. Extract outer boundary", outer),
                ("5. Detect wall candidates", wall_candidates),
                ("6. Build final border mask", borders),
                ("7. Overlay final border mask", border_overlay),
            ],
            columns=3,
        )
        Image.fromarray(borders.astype(np.uint8) * 255).save(image_output_dir / "border_mask.png")
        print(f"wrote {image_output_dir / 'border_mask.png'}")


if __name__ == "__main__":
    main()
