#!/usr/bin/env python3
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    input_dir = Path("data/inputs")
    output_dir = Path("data/output")

    for image_path in sorted(input_dir.glob("*.webp")):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        foreground = ((val < 247) | (sat > 10)).astype(np.uint8)
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
        floorplan = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        floorplan = cv2.morphologyEx(
            floorplan.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        ).astype(bool)

        padded = np.pad(floorplan.astype(np.uint8), 1)
        flood = padded.copy()
        cv2.floodFill(flood, None, (0, 0), 1)
        floorplan = floorplan | (flood[1:-1, 1:-1] == 0)

        outer = floorplan & ~cv2.erode(
            floorplan.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
        ).astype(bool)

        wall_candidates = ((val > 218) & (sat < 48) & floorplan).astype(np.uint8)
        wall_candidates = cv2.morphologyEx(
            wall_candidates,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )

        walls = np.zeros(wall_candidates.shape, dtype=bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(wall_candidates, 8)
        for label in range(1, n):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            min_side = max(1, min(w, h))
            max_side = max(w, h)
            density = area / (w * h)
            if min_side <= 72 or max_side / min_side >= 2.2 or density < 0.7:
                walls[labels == label] = True

        borders = (walls | outer).astype(np.uint8)
        borders = cv2.morphologyEx(
            borders,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (43, 5)),
        )
        borders = cv2.morphologyEx(
            borders,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 43)),
        )
        borders = cv2.dilate(
            borders,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )
        borders = (borders > 0) & floorplan

        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(borders.astype(np.uint8) * 255).save(
            image_output_dir / "border_mask.png"
        )
        print(f"wrote {image_output_dir / 'border_mask.png'}")


if __name__ == "__main__":
    main()
