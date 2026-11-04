"""Street-map styled rendering of extracted centrelines (no labels).

Roads are rasterized from centreline vectors at their estimated widths, smoothed,
and composited over cream city blocks with white kerbs, as in common web maps.
Wide road areas that straight centrelines cannot describe, such as plazas,
roundabouts and boulevards, come from the cleaned road mask. Vegetation from the
input image becomes park green on blocks and median strips on roads.
"""
import cv2
import numpy as np
import shapely
from shapely.geometry import LineString

from road_network import line_opening

# BGR palette.
BLOCK = (220, 242, 250)
KERB = (255, 255, 255)
ROAD = (230, 220, 211)
PARK = (200, 235, 200)
PARK_EDGE = (180, 222, 182)

SUPERSAMPLE = 3
# Enclosed regions in roads and parks smaller than this many squared typical road widths are filled.
HOLE_AREA = 10


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


def _touching(mask, anchors):
    """Components of mask that overlap anchors or touch the image border."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[anchors & mask])] = True
    keep[np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))] = True
    keep[0] = False
    return keep[labels]


def fill_holes(mask, max_area):
    """Fill background regions enclosed by mask, not touching the border, below max_area."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=4)
    height, width = mask.shape
    x, y, w, h, area = stats.T
    inside = (x > 0) & (y > 0) & (x + w < width) & (y + h < height)
    fill = inside & (area < max_area)
    fill[0] = False
    return mask | fill[labels]


def connected_lines(lines, shape):
    """Keep only roads in a connected group that reaches the image border.

    Two roads touch when their footprints, the centrelines widened to their
    widths, overlap. A group of touching roads floating inside the image, even
    one of several roads, is dropped.
    """
    height, width = shape[:2]
    if not lines:
        return []
    footprints = [LineString(line["points"]).buffer(line["width"] / 2) for line in lines]
    parent = list(range(len(lines)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = shapely.STRtree(footprints)
    for i, j in zip(*tree.query(footprints, predicate="intersects")):
        parent[root(int(i))] = root(int(j))
    border = set()
    for i, footprint in enumerate(footprints):
        x0, y0, x1, y1 = footprint.bounds
        if x0 <= 0 or y0 <= 0 or x1 >= width - 1 or y1 >= height - 1:
            border.add(root(i))
    return [line for i, line in enumerate(lines) if root(i) in border]


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
    lines = connected_lines(lines, original.shape)
    s = SUPERSAMPLE
    shape = (height * s, width * s)
    typical = float(np.median([line['width'] for line in lines])) * s if lines else 10.0 * s
    roads = np.zeros(shape, np.uint8)
    for line, line_width in zip(lines, road_classes([line['width'] for line in lines])):
        points = np.rint(np.asarray(line['points']) * s).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(roads, [points], False, 1, max(1, round(line_width * s)), cv2.LINE_8)
    centrelines = roads.astype(bool)
    if road_area is not None:
        area = cv2.resize(road_area.astype(np.uint8), (width * s, height * s), interpolation=cv2.INTER_NEAREST)
        area = cv2.morphologyEx(area, cv2.MORPH_OPEN, _disc(1.25 * typical))
        roads |= _smooth(area, 0.5 * typical).astype(np.uint8)
    roads = _smooth(roads, 0.2 * typical)
    # Closing fills concave road corners, which rounds the corners of the blocks.
    roads = cv2.morphologyEx(roads.astype(np.uint8), cv2.MORPH_CLOSE, _disc(0.3 * typical)).astype(bool)
    # Small enclosed islands are mask noise, cars or road markings, not city blocks.
    roads = fill_holes(roads, HOLE_AREA * typical ** 2)
    # Road areas count as roads too: drop those reaching no kept road and no border.
    roads = _touching(roads, centrelines)
    blocks = ~roads
    kerb = cv2.dilate(roads.astype(np.uint8), _disc(1.5 * s)).astype(bool) & blocks
    green = vegetation_mask(original, shape, typical)
    # Green on a road reads as a median strip only when it stays clear of the kerbs
    # and is long and straight; line openings turn tree crowns into strips.
    medians = green & cv2.erode(roads.astype(np.uint8), _disc(0.12 * typical)).astype(bool)
    medians = _smooth(line_opening(medians, 2 * typical, 12), 0.1 * typical)
    parks = fill_holes(_large(green & ~roads, 6 * typical ** 2), HOLE_AREA * typical ** 2) & ~roads
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


# Schematic palette (BGR): paper, road casing and fills for minor, middle and major classes.
PAPER = (248, 248, 248)
CASING = (90, 90, 90)
CLASS_FILLS = ((255, 255, 255), (120, 215, 250), (70, 150, 240))
CLASS_STROKES = (3, 5, 7)


def render_schematic_map(lines, shape):
    """Diagram of the network from centreline vectors alone, on a plain background.

    Each centreline is simplified to a few straight segments and drawn with a fixed
    stroke for its width class (minor, middle, major), so the map shows network
    structure and hierarchy rather than measured road footprints.
    """
    height, width = shape[:2]
    lines = connected_lines(lines, shape)
    s = SUPERSAMPLE
    canvas = np.empty((height * s, width * s, 3), np.uint8)
    canvas[:] = PAPER
    if not lines:
        return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    widths = np.array([line["width"] for line in lines], float)
    classes = np.zeros(len(lines), int)
    if len(lines) >= 3:
        classes = np.digitize(widths, np.quantile(widths, [1 / 3, 2 / 3]))
    tolerance = 0.25 * float(np.median(widths)) * s
    paths = []
    for line, road_class in zip(lines, classes):
        points = (np.asarray(line["points"], np.float32) * s).reshape(-1, 1, 2)
        points = cv2.approxPolyDP(points, tolerance, False)
        paths.append((road_class, np.rint(points).astype(np.int32)))
    # Casings first, then fills by ascending class, so major roads pass over minor ones.
    for road_class, points in paths:
        stroke = (CLASS_STROKES[road_class] + 2) * s
        cv2.polylines(canvas, [points], False, CASING, stroke, cv2.LINE_AA)
    for road_class, points in sorted(paths, key=lambda path: path[0]):
        cv2.polylines(canvas, [points], False, CLASS_FILLS[road_class], CLASS_STROKES[road_class] * s, cv2.LINE_AA)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
