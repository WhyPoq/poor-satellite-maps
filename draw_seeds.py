import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def draw_seeds(image_path, output, radius=3):
    """Paint road seeds with the left button, background with the right."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if radius < 0:
        raise ValueError("Brush radius must be nonnegative.")
    labels = np.zeros(image.shape[:2], np.uint8)
    if Path(output).exists():
        seeds = json.loads(Path(output).read_text(encoding="utf-8"))
        for name, label in (("fg", 1), ("bg", 2)):
            points = np.asarray(seeds[name], int).reshape(-1, 2)
            if np.any(points < 0) or np.any(points >= np.array(image.shape[1::-1])):
                raise ValueError("Saved seeds do not fit this image.")
            labels[points[:, 1], points[:, 0]] = label
    brush = 0

    def mouse(event, x, y, flags, _):
        nonlocal brush
        if event == cv2.EVENT_LBUTTONDOWN:
            brush = 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            brush = 2
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            brush = 0
        if brush and (flags & (cv2.EVENT_FLAG_LBUTTON | cv2.EVENT_FLAG_RBUTTON)):
            cv2.circle(labels, (x, y), radius, brush, -1)

    title = "Left: road | Right: background | C: clear | S: save | Esc: cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            view = image.copy()
            view[labels == 1] = (0, 0, 255)
            view[labels == 2] = (255, 0, 0)
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 255
            if key == ord("c"):
                labels[:] = 0
            elif key == ord("s"):
                if not (np.any(labels == 1) and np.any(labels == 2)):
                    print("Draw both road and background seeds before saving.")
                    continue
                seeds = {
                    name: np.argwhere(labels == label)[:, ::-1].tolist()
                    for name, label in (("fg", 1), ("bg", 2))
                }
                path = Path(output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(seeds), encoding="utf-8")
                return seeds
            elif key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw road and background seeds.")
    parser.add_argument("image")
    parser.add_argument("output", nargs="?", default="seeds.json")
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()
    draw_seeds(args.image, args.output, args.radius)
