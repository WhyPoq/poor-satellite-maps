"""Street-map styled rendering of extracted centrelines (no labels).

Roads are rasterized from centreline vectors at their estimated widths, smoothed,
and composited over cream city blocks with white kerbs, as in common web maps.
Wide road areas that straight centrelines cannot describe, such as plazas,
roundabouts and boulevards, come from the cleaned road mask. Vegetation from the
input image becomes park green on blocks and median strips on roads.
"""
import cv2
import numpy as np

from road_network import line_opening

# BGR palette.
BLOCK = (220, 242, 250)
KERB = (255, 255, 255)
ROAD = (230, 220, 211)
PARK = (200, 235, 200)
PARK_EDGE = (180, 222, 182)

SUPERSAMPLE = 3


def _disc(radius):
    size = 2 * max(1, int(round(radius))) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _smooth(mask, sigma):
    """Remove scallops from overlapping caps and widths: blur, then threshold at half."""
    blurred = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return blurred > 0.5


def _large(mask, min_area):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def vegetation_mask(original, shape, scale):
    """Smooth vegetation regions from the BGR input, at the supersampled shape.

    scale is the typical road width in supersampled pixels; it sets how much
    detail survives, so single street trees vanish and parks stay.
    """
    image = original.astype(np.float32)
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    excess_green = (2 * g - r - b) / np.maximum(r + g + b, 1)
    excess_green = cv2.resize(excess_green, shape[::-1], interpolation=cv2.INTER_LINEAR)
    green = cv2.GaussianBlur(excess_green, (0, 0), 0.2 * scale) > 0.05
    green = cv2.morphologyEx(green.astype(np.uint8), cv2.MORPH_OPEN, _disc(0.25 * scale))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, _disc(0.25 * scale))
    green = _smooth(green, 0.15 * scale)
    return _large(green, scale ** 2)


def road_classes(widths, count=3):
    """Snap widths to the median of their quantile class, so a street keeps one width."""
    widths = np.asarray(widths, float)
    if widths.size < count:
        return widths
    classes = np.digitize(widths, np.quantile(widths, np.arange(1, count) / count))
    return np.array([np.median(widths[classes == c]) for c in classes])


def render_road_map(lines, original, road_area=None):
    """lines: [{'points': [[x, y], ...], 'width': px}] in processed-image pixels.

    road_area, optional, is the cleaned road mask at the processed size. Only its
    parts wider than about 2.5 typical road widths are drawn, so its noisy narrow
    fringes never reach the map.
    """
    height, width = original.shape[:2]
    s = SUPERSAMPLE
    shape = (height * s, width * s)
    typical = float(np.median([line['width'] for line in lines])) * s if lines else 10.0 * s
    roads = np.zeros(shape, np.uint8)
    for line, line_width in zip(lines, road_classes([line['width'] for line in lines])):
        points = np.rint(np.asarray(line['points']) * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(roads, [points], False, 1, max(1, round(line_width * s)), cv2.LINE_8)
    if road_area is not None:
        area = cv2.resize(road_area.astype(np.uint8), (width * s, height * s), interpolation=cv2.INTER_NEAREST)
        area = cv2.morphologyEx(area, cv2.MORPH_OPEN, _disc(1.25 * typical))
        roads |= _smooth(area, 0.5 * typical).astype(np.uint8)
    roads = _smooth(roads, 0.2 * typical)
    # Closing fills concave road corners, which rounds the corners of the blocks.
    roads = cv2.morphologyEx(roads.astype(np.uint8), cv2.MORPH_CLOSE, _disc(0.3 * typical)).astype(bool)
    blocks = ~roads
    kerb = cv2.dilate(roads.astype(np.uint8), _disc(1.5 * s)).astype(bool) & blocks
    green = vegetation_mask(original, shape, typical)
    # Green on a road reads as a median strip only when it stays clear of the kerbs
    # and is long and straight; line openings turn tree crowns into strips.
    medians = green & cv2.erode(roads.astype(np.uint8), _disc(0.12 * typical)).astype(bool)
    medians = _smooth(line_opening(medians, 2 * typical, 12), 0.1 * typical)
    parks = _large(green & ~roads, 6 * typical ** 2)
    # Closing bridges the kerb-clearance gap where a median meets a park.
    green = cv2.morphologyEx((medians | parks).astype(np.uint8), cv2.MORPH_CLOSE, _disc(0.15 * typical)).astype(bool)
    green_edge = green & ~cv2.erode(green.astype(np.uint8), _disc(0.5 * s)).astype(bool)

    canvas = np.empty((*shape, 3), np.uint8)
    canvas[:] = BLOCK
    canvas[roads] = ROAD
    canvas[kerb] = KERB
    canvas[green] = PARK
    canvas[green_edge] = PARK_EDGE
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)

