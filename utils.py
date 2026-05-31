import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def pipeline_panel(array, label, size=(400, 400), padding=8):
    if array.dtype == bool or (array.ndim == 2 and array.max() <= 1):
        array = array.astype(np.uint8) * 255

    image = Image.fromarray(array.astype(np.uint8))
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(size)

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", 22)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_position = (padding, size[1] - (text_bbox[3] - text_bbox[1]) - padding)
    draw.text(text_position, label, font=font, fill=(0, 255, 0), stroke_width=2, stroke_fill=(0, 0, 0))
    return image


def save_pipeline_image(output_path, steps, panel_size=(400, 400), columns=2):
    rows = int(np.ceil(len(steps) / columns))
    pipeline = Image.new("RGB", (panel_size[0] * columns, panel_size[1] * rows), "white")
    for index, (label, panel) in enumerate(steps):
        x = (index % columns) * panel_size[0]
        y = (index // columns) * panel_size[1]
        pipeline.paste(pipeline_panel(panel, label, panel_size), (x, y))
    pipeline.save(output_path)


def describe_rooms(labels):
    rooms = []
    total_area = int((labels > 0).sum())

    for index, label in enumerate(np.unique(labels[labels > 0]), start=1):
        room_mask = (labels == label).astype(np.uint8)
        area = int(room_mask.sum())

        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(int).tolist()

        moments = cv2.moments(room_mask)
        center = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]

        rooms.append(
            {
                "index": index,
                "area": area,
                "relative_area": area / total_area,
                "area_percent": round(area / total_area * 100, 2),
                "center": center,
                "polygon": polygon,
            }
        )

    return rooms


def draw_room_area_labels(overlay, rooms):
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Arial.ttf", 32)

    for room in rooms:
        label = f"{room['area_percent']:.1f}%"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        x = room["center"][0] - text_width // 2
        y = room["center"][1] - text_height // 2
        draw.text((x, y), label, font=font, fill=(0, 255, 0), stroke_width=3, stroke_fill=(0, 0, 0))

    return np.array(image)
