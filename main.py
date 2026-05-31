import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils import describe_rooms, draw_room_area_labels, save_pipeline_image


def label_rooms(floorplan, wall_lines):
    wall_mask = cv2.dilate(wall_lines.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))).astype(bool)

    room_space = floorplan & ~wall_mask
    _, labels = cv2.connectedComponents(room_space.astype(np.uint8), 8)

    outer_edge = floorplan & ~cv2.erode(
        floorplan.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ).astype(bool)

    edge_labels = np.unique(labels[outer_edge])
    for label in edge_labels:
        labels[labels == label] = 0

    return labels


def color_regions(labels, wall_lines=None):
    rng = np.random.default_rng(7)
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in range(1, labels.max() + 1):
        colored[labels == label] = rng.integers(40, 256, size=3, dtype=np.uint8)
    return colored


def overlay_regions(rgb, labels, alpha=0.45):
    overlay = rgb.copy()
    region_colors = color_regions(labels)
    region_mask = labels > 0
    overlay[region_mask] = (
        rgb[region_mask].astype(np.float32) * (1 - alpha) + region_colors[region_mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def fill_holes(mask):
    padded = np.pad(mask.astype(np.uint8), 1)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = flood[1:-1, 1:-1] == 0
    return mask | holes


def close_region_gaps(labels, kernel_size=9):
    closed_labels = labels.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Close and fill each room separately so small gaps become part of the room.
    region_ids = np.unique(labels[labels > 0])
    for region_id in region_ids:
        region = labels == region_id
        closed = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        filled = fill_holes(closed)

        # Only claim unlabeled pixels; do not overwrite another room.
        closed_labels[(closed_labels == 0) & filled] = region_id

    return closed_labels


def merge_small_regions(labels, min_area=50):
    merged = labels.copy()

    # Split rooms into small regions that should be merged and large target regions.
    region_ids, areas = np.unique(labels[labels > 0], return_counts=True)
    area_by_id = dict(zip(region_ids, areas))
    large_ids = set(region_ids[areas >= min_area])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for region_id, area in zip(region_ids, areas):
        if area >= min_area:
            continue

        region_mask = labels == region_id
        search = region_mask.astype(np.uint8)

        # Grow the small region until it touches a large neighboring region.
        touched_ids = set()
        while not touched_ids:
            search = cv2.dilate(search, kernel)
            touched_ids = set(np.unique(labels[search > 0])) & large_ids

        # Merge into the largest touched neighbor.
        target_id = max(touched_ids, key=lambda label: area_by_id[label])
        merged[region_mask | ((search > 0) & (labels == 0))] = target_id

    return merged


def main():
    input_dir = Path("data/inputs")
    output_dir = Path("data/output")

    for image_path in sorted(input_dir.glob("*.webp")):
        ### 1. Load image and convert to HSV
        rgb = np.array(Image.open(image_path).convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        ### 2. Isolate the drawing from the white background
        # The constants were chosen empirically
        drawing_mask = ((val < 247) | (sat > 10)).astype(np.uint8)

        ### 3. Find and fill the floorplan region
        # Connected components split the drawing into separate blobs.
        _, labels, stats, _ = cv2.connectedComponentsWithStats(drawing_mask.astype(np.uint8), 8)
        # We keep the largest blob because it is the main floorplan footprint.
        largest_component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        floorplan = fill_holes(largest_component)

        ### 4. Extract outer floorplan boundary
        # Erosion shrinks the floorplan
        outer = floorplan & ~cv2.erode(
            floorplan.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        ).astype(bool)

        ### 5. Build final border mask
        # Use light interior wall pixels plus the extracted outer boundary.
        # The constants were chosen empirically
        borders = (((val > 200) & (sat < 48) & floorplan) | outer) & floorplan

        ### 6. Extract largest border component
        n, labels, stats, _ = cv2.connectedComponentsWithStats(borders.astype(np.uint8), 8)
        border_component = np.zeros(borders.shape, dtype=bool)
        if n > 1:
            border_component = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))

        ### 7. Skeletonize inner and outer walls
        line_mask = cv2.ximgproc.thinning(border_component.astype(np.uint8) * 255) > 0

        ### 8. Color inner room regions
        room_labels = label_rooms(floorplan, line_mask)
        room_regions = color_regions(room_labels, line_mask)

        ### 9. Merge tiny room regions
        merged_room_labels = merge_small_regions(room_labels, min_area=2600)
        merged_room_regions = color_regions(merged_room_labels)

        ### 10. Close room gaps
        closed_room_labels = close_region_gaps(merged_room_labels, kernel_size=9)
        closed_room_regions = color_regions(closed_room_labels)

        ### 11. Overlay
        overlay = overlay_regions(rgb, closed_room_labels)
        rooms = describe_rooms(closed_room_labels)
        overlay = draw_room_area_labels(overlay, rooms)

        # Save final outputs
        image_output_dir = output_dir / image_path.stem
        image_output_dir.mkdir(parents=True, exist_ok=True)
        save_pipeline_image(
            image_output_dir / "pipeline.png",
            [
                ("1. Load image and convert to HSV", rgb),
                ("2. Isolate the drawing from the white background", drawing_mask),
                ("3. Find and fill floorplan", floorplan),
                ("4. Extract outer boundary", outer),
                ("5. Build final border mask", borders),
                ("6. Extract largest border component", border_component),
                ("7. Skeletonize walls", line_mask),
                ("8. Color inner regions", room_regions),
                ("9. Merge tiny regions", merged_room_regions),
                ("10. Close room gaps", closed_room_regions),
                ("11. Overlay", overlay),
            ],
            columns=3,
        )
        Image.fromarray(overlay).save(image_output_dir / "overlay.png")
        with (image_output_dir / "rooms.json").open("w") as file:
            json.dump(
                {
                    "source": str(image_path),
                    "image_width": rgb.shape[1],
                    "image_height": rgb.shape[0],
                    "room_count": len(rooms),
                    "rooms": rooms,
                },
                file,
                indent=2,
            )
        print(f"wrote {image_output_dir}")


if __name__ == "__main__":
    main()
