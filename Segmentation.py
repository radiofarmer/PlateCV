from skimage import morphology, filters
from skimage.feature import canny
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import probabilistic_hough_line, rotate
import matplotlib.pyplot as plt
from PlateCV.Utils import *


def straighten_image(img, plot=False, **kwargs):
    hzn_edges = filters.sobel_h(img)
    hzn_edges = canny(hzn_edges)
    hzn_edges_binary = morphology.closing(hzn_edges > filters.threshold_otsu(hzn_edges), morphology.square(3))
    lines = probabilistic_hough_line(hzn_edges_binary, **kwargs)

    argands = [np.arctan(np.abs(p1[1] - p0[1]) / np.abs(p1[0] - p0[0]))
               for p0, p1 in [line for line in lines] if p0[0] != p1[1]]

    theta_avg = np.mean(argands)
    rot_degrees = theta_avg / np.pi * 180.

    if plot:
        plt.imshow(img, cmap='Greys')
        for line in lines:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        plt.title(f"Rotation = {rot_degrees:.2f} degrees")
        plt.show()
    print(f"Image rotated {-rot_degrees} degrees")
    return rotate(img, angle=rot_degrees)


def find_plate_region(img, canny_sigma=1, report=False):
    # Find regions separated by edges
    edges = morphology.closing(canny(img, sigma=canny_sigma), morphology.square(5))
    try:
        thresh = filters.threshold_otsu(edges)
    except ValueError:
        plt.imshow(img)
        plt.show()
        return img
    labeled, num_regions = label(edges < thresh, return_num=True, connectivity=1)
    if report:
        print("Number of regions:", num_regions)
    # Find the area that represents the interior of the plate
    # Procedure:
    #   1.  Assume the two largest areas are the agar surface and the area outside the plate. Find these regions.
    #   2.  Recognizing that the area outside the plate has a very large whole in it, assume that it's convex
    #       area will differ substantially from its true area
    #   3.  Based on the assumption in (2), take the region with the lower convex-area-to-area ratio to be as
    #       the plate interior
    regions = regionprops(labeled)
    regions_max = sorted(regions, key=lambda r: r.area, reverse=True)[:2]
    plate_bbox = sorted(regions_max, key=lambda r: r.convex_area / r.area)[0].bbox

    return img[plate_bbox[0]:plate_bbox[2], plate_bbox[1]:plate_bbox[3]]


def find_spots(img, radius, design):
    # Rows and columns (of constructs), spots per series (i.e. per construct), horizontal orientation
    if isinstance(design, tuple):
        rows, cols, spc, hzn = design
    else:
        rows = design['rows']
        cols = design['cols']
        spc = design['spots_per_construct']
        hzn = design['horizontal']
    n = rows * cols
    # Smooth out the background, leaving only the colonies
    spots = morphology.white_tophat(img, morphology.disk(radius))
    # Get the size of the spots based on the largest ones visible
    #   First consolidate the spots into solid regions
    spot_thresh = filters.threshold_li(spots)  # Li's method chosen empirically
    spot_edges = morphology.closing(spots > spot_thresh, morphology.disk(6))  # Turn large colonies into solid regions
    spot_edges = morphology.opening(spot_edges, morphology.disk(radius * 0.1))  # Get rid of individual colonies
    spot_regions = label(spot_edges)
    #   Then get the n * sps largest regions, where n = number of constructs and sps = spots per series, to serve as
    #       templates
    spot_regionprops = [r for r in regionprops(spot_regions) if r.eccentricity < 0.5]
    # Sort by area
    start_spots = sorted(spot_regionprops, key=lambda r: r.area, reverse=True)[:n * spc]
    # Sort by difference between filled and real area (increasing)
    start_spots.sort(key=lambda r: r.filled_area - r.area)
    spot_diam = np.mean([r.equivalent_diameter for r in start_spots[:n]])
    # Filter out small spots
    diam_cutoff = 0.5 * spot_diam
    try:
        start_spots = [s for s in start_spots if s.equivalent_diameter > diam_cutoff]
    except IndexError:
        print("No spots found on plate!")
        return start_spots, []
    sorted_by_coord = sorted([s for s in start_spots if s.equivalent_diameter > diam_cutoff],
                             key=lambda r: np.sum(r.bbox[:2]))
    if sorted_by_coord:
        top_left = sorted_by_coord[0]
    else:
        print("No spots found on plate!")
        return start_spots, []

    # If there are multiple rows (if vertical) or columns (if horizontal) of constructs, find the span between them
    if hzn and cols > 1 or not hzn and rows > 1:
        idx = int(hzn)
        test_point = top_left.centroid[idx] + spc * spot_diam
        beyond_point = [r for r in start_spots if r.centroid[idx] > test_point and r.equivalent_diameter > diam_cutoff]
        try:
            next_group_start = sorted(beyond_point, key=lambda r: r.centroid[idx] - test_point)[0]
            span = next_group_start.centroid[idx] - top_left.centroid[idx]
        except IndexError:
            # This error occurs when `beyond_point` is an empty list, indicating that constructs are only visible
            # in one row or column
            span = 0
            if hzn:
                cols = 1
            else:
                rows = 0
    else:
        span = 0
    step_size = span / spc  # Space between spots of the same construct
    # Get bounding boxes for all regions
    bboxes = []
    spots_binary = spots > spot_thresh
    for reg in start_spots:
        for group in range(cols if hzn else rows):
            dir_idx = int(hzn)
            # Starting with the spot in the top-left corner, find the other high-concentration spots
            # by searching for regions located less than a spot-radius away in the x-direction (for horizontal
            # dilutions) or in the y-direction (for vertical dilutions)
            if abs(reg.centroid[dir_idx] - top_left.centroid[dir_idx] - span * group) < spot_diam / 2:
                spot_box = new_bbox_at([reg.centroid[0], reg.centroid[1]], reg.bbox, spots, mode="center")
                bboxes.append(spot_box)
                spot_box = refine_spot(spots_binary, spot_box, threshold="binary")
                # Move right (or down, as needed), creating bounding boxes for the rest of the dilution series
                for i in range(1, spc):
                    if hzn:
                        x_new = spot_box[1] + i * step_size
                        adj_spot_box = refine_spot(spots_binary,
                                                   (spot_box[0], x_new, spot_box[2], x_new + spot_diam),
                                                   threshold="binary")
                        bboxes.append(adj_spot_box)
                    else:
                        y_new = spot_box[0] + i * step_size
                        adj_spot_box = refine_spot(spots_binary,
                                                   (y_new, spot_box[1], y_new + spot_diam, spot_box[3]),
                                                   threshold="binary")
                        bboxes.append(adj_spot_box)
    bboxes = [expand_bounds(spot_regions, b) for b in bboxes]
    return spots, bboxes


def extract_and_label(img, bbox, threshold, tolerance):
    spot = extract(img, bbox)
    if threshold == 'auto':
        threshold = filters.threshold_li(spot)
        spot, n_regions = label(spot > threshold, return_num=True, connectivity=tolerance)
    elif threshold == 'binary':
        threshold = 0
        spot, n_regions = label(spot, return_num=True, connectivity=tolerance)
    elif threshold == 'labeled':
        threshold = 0
        n_regions = count_regions(spot)
    else:
        spot, n_regions = label(spot > threshold, return_num=True, connectivity=tolerance)
    return spot, n_regions, threshold


def refine_spot(img, bbox, threshold='binary', tolerance=1, max_shift=0.25, max_iter=20):
    '''
    Iteratively search for the center of a spot which the initial bounding box overlaps.
    :param img: Image array containing all spots
    :param bbox: The pixel data of the initial search region
    :param threshold: Either a value between zero and one to serve as the cutoff threshold, or a string specifying
    :param max_shift: Maximum distance, as a fraction of the dynamically-measured spot size, that a bounding box
    may shift in search of more colonies.
    :param max_iter: Maximum number o
    the threshold calculation method to be used.
    Possible string values:
        auto - Finds a threshold by Li's method
        binary - Indicates that the image is already threshold. Labeling is applied as-is.
        labeled - For pre-labeled images; assumes the background to be zero
    :param tolerance: The value to be passed to `connectivity` in skimage.measure.label
    :return: The optimized bounding box for the spot, hopefully centered directly on the region to which sample
    was initially applied
    '''
    bbox = refine_spot_recursive(0, img, bbox, "num_regions", threshold, tolerance, max_iter)
    return refine_spot_recursive(0, img, bbox, "area", threshold, tolerance, max_iter)


def refine_spot_recursive(curr_iter, img, bbox, metric="num_regions", threshold='binary',
                          tolerance=1, max_shift=0.25, max_iter=20):
    # Get the initial number of regions in the bounding box
    spot = extract(img, bbox)
    if threshold == 'auto':
        threshold = filters.threshold_li(spot)
        spot, n_regions = label(spot > threshold, return_num=True, connectivity=tolerance)
    elif threshold == 'binary':
        threshold = 0
        spot, n_regions = label(spot, return_num=True, connectivity=tolerance)
    elif threshold == 'labeled':
        threshold = 0
        n_regions = count_regions(spot)
    else:
        spot, n_regions = label(spot > threshold, return_num=True, connectivity=tolerance)

    # Get the centroids and areas of all regions
    spot_data = regionprops_table(spot, properties=('label', 'centroid', 'area'))
    spot_center = np.mean(np.row_stack([spot_data["centroid-0"], spot_data["centroid-1"]]), axis=1)
    # Check for NaNs (i.e. lack of points)
    if np.isnan(spot_center).any():
        print("No spot identified")
        return bbox
    curr_area = np.sum(spot_data['area'])
    # Make a new bounding box centered on the average of the centroids, and extract new regions
    bbox_new = new_bbox_at(spot_center, bbox, img, mode="shift_center")
    spot = extract(img, bbox_new) > threshold
    spot, n_regions_new = label(spot, return_num=True, connectivity=tolerance)
    new_area = np.sum(regionprops_table(spot, properties=('area',))['area'])
    # Recursively evaluate as long as the new bounding box contains more regions without decreasing the area
    if metric == "num_regions":
        if curr_iter < max_iter and n_regions_new > n_regions and new_area >= curr_area:
            return refine_spot_recursive(curr_iter + 1, img, bbox_new, "num_regions", threshold, tolerance)
        else:
            return bbox
    elif metric == "area":
        if curr_iter < max_iter and n_regions_new >= n_regions and new_area > curr_area:
            return refine_spot_recursive(curr_iter + 1, img, bbox_new, "num_regions", threshold, tolerance)
        else:
            return bbox


def expand_bounds(labeled_img, bbox):
    labels_in_spot = np.unique(extract(labeled_img, bbox))
    label_boxes = regionprops_table(labeled_img, properties=('label', 'bbox'))
    # Get maximum bounds of the regions contained within the spot bounding box
    t, l, b, r = bbox
    for bnd in [b for b, lab in zip(label_boxes['bbox-0'], label_boxes['label']) if lab in labels_in_spot]:
        t = np.min((t, bnd))
    for bnd in [b for b, lab in zip(label_boxes['bbox-1'], label_boxes['label']) if lab in labels_in_spot]:
        l = np.min((l, bnd))
    for bnd in [b for b, lab in zip(label_boxes['bbox-2'], label_boxes['label']) if lab in labels_in_spot]:
        b = np.max((b, bnd))
    for bnd in [b for b, lab in zip(label_boxes['bbox-3'], label_boxes['label']) if lab in labels_in_spot]:
        r = np.max((r, bnd))

    # zipped_data = zip(zip(label_boxes['bbox-0'],
    #                       label_boxes['bbox-1'],
    #                       label_boxes['bbox-2'],
    #                       label_boxes['bbox-3']),
    #                   label_boxes['label'])
    # for box_t, box_l, box_b, box_r in [b for b, lab in zipped_data if lab in labels_in_spot]:
    #     t = np.min((t, box_t))
    #     l = np.min((l, box_l))
    #     b = np.max((b, box_b))
    #     r = np.max((r, box_r))

    return t, l, b, r
