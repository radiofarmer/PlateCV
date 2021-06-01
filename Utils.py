import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Polygon


def extract(image, bbox, rois=None):
    bbox = [int(b) for b in bbox]
    cropped_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    if rois is not None:
        new_rois = [(r[0] - bbox[0], r[1] - bbox[1], r[2] - bbox[0], r[3] - bbox[1]) for r in rois]
        return cropped_image, new_rois
    else:
        return cropped_image


def count_regions(img):
    return max(len(np.unique(img)) - 1, 0)


def get_box_center(bbox):
    center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
    return [int(c) for c in center]


def new_bbox_at(coords, bbox, img, mode="center"):
    img_rmax, img_cmax = img.shape
    if mode == "center":
        # Get a bounding box with the same dimensions, centered at the new coordinates
        row_shift = coords[0] - get_box_center(bbox)[0]
        col_shift = coords[1] - get_box_center(bbox)[1]
    elif mode == "shift":
        # Shift the bounding box by the amount specified in coords (i.e. coords is a directional vector)
        row_shift, col_shift = coords
    elif mode == "shift_center":
        # Move the center of the bounding box by the specified amount
        row_shift = coords[0] - (get_box_center(bbox)[0] - bbox[0])
        col_shift = coords[1] - (get_box_center(bbox)[1] - bbox[1])
    else:
        row_shift = coords[0] - bbox[0]
        col_shift = coords[1] - bbox[1]
    bbox = (max(bbox[0] + row_shift, 0),
            max(bbox[1] + col_shift, 0),
            max(min(bbox[2] + row_shift, img_rmax - 1), 0),
            max(min(bbox[3] + col_shift, img_cmax - 1), 0))
    return [int(b) for b in bbox]


def plot_rois(img, rois, show=False, title=None):
    fig, ax = plt.subplots()
    if title:
        fig.suptitle(title)
    ax.imshow(img)
    for r in rois:
        ymin, xmin, ymax, xmax = r
        rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    if show:
        plt.show()
        return None, None
    else:
        return fig, ax


def add_rois(axes, rois):
    for r in rois:
        ymin, xmin, ymax, xmax = r
        rect = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red', linewidth=1)
        axes.add_patch(rect)
    return axes


def is_in_bbox(point, bbox):
    return (bbox[0] <= point[0] <= bbox[2]) and (bbox[1] <= point[1] <= bbox[3])


def check_intersection(rect1, rect2):
    rect1_corners = ((rect1[0], rect1[1]),
                     (rect1[0], rect1[3]),
                     (rect1[2], rect1[3]),
                     (rect1[2], rect1[1]))
    rect2_corners = ((rect2[0], rect2[1]),
                     (rect2[0], rect2[3]),
                     (rect2[2], rect2[3]),
                     (rect2[2], rect2[1]))
    a = Polygon(rect1_corners)
    b = Polygon(rect2_corners)
    return a.intersects(b)


def bbox_area(bbox):
    return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
