from skimage import morphology, filters
from skimage.feature import canny
from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage as ndi
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
    return rotate(img, angle=rot_degrees, mode='edge')


def find_plate_region(img, canny_sigma=1, tolerance=20, report=False):
    # Find regions separated by edges
    edges = morphology.dilation(canny(img, sigma=canny_sigma), morphology.square(tolerance))
    try:
        thresh = filters.threshold_otsu(edges)
    except ValueError:
        print("Error computing threshold")
        plt.imshow(img)
        plt.show()
        return img
    labeled, num_regions = label(edges < thresh, return_num=True, connectivity=1, background=-1)
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


def find_spots(img, radius_large, radius_small, design, show_plot=False, save_plot=None, ecc_cutoff=0.5, **kwargs):
    """

    :param img:
    :param radius_large:
    :param radius_small:
    :param design:
    :param show_plot:
    :param save_plot:
    :param ecc_cutoff:
    :return:
    """
    # Rows and columns (of constructs), spots per series (i.e. per construct), horizontal orientation
    if isinstance(design, tuple):
        rows, cols, spc, hzn = design
    else:
        rows = design['rows']
        cols = design['cols']
        spc = design['spots_per_construct']
        hzn = design['horizontal']
    n = rows * cols
    # Identify edges to get rid of smooth gradients in the background
    spots = img
    spot_edges = morphology.closing(
        filters.sobel(img),
        morphology.square(5))
    spot_edges_smoothed = filters.gaussian(spot_edges, sigma=2)
    spot_edges = filters.apply_hysteresis_threshold(spot_edges,
                                                    filters.threshold_minimum(spot_edges_smoothed),
                                                    np.mean(spot_edges_smoothed[spot_edges > np.mean(spot_edges)]))
    spots_filled = ndi.binary_fill_holes(spot_edges)
    # Tinker with this to adjust the spot size limit:
    spots_filled = morphology.closing(spots_filled, morphology.disk(radius_large / 4))
    # Make contiguous regions from clusters of small spots
    spots_labeled = label(spots_filled)
    small_spot_labels = [r.label for r in regionprops(spots_labeled) if r.equivalent_diameter <= radius_small]
    small_spots = np.any([spots_labeled == i for i in small_spot_labels], axis=0).astype(np.uint8)
    small_spots = morphology.dilation(small_spots, morphology.disk((radius_large + radius_small) / 2))
    ##########
    # plt.imshow(small_spots)
    # plt.show()
    ##########
    spot_regions = label(spots_filled)
    #   Then get the n * sps largest regions, where n = number of constructs and sps = spots per series, to serve as
    #       templates
    spot_regionprops = [r for r in regionprops(spot_regions)]
    # Remove highly-eccentric regions, in case the edges of the plate get labeled
    spot_regionprops = [r for r in spot_regionprops if r.eccentricity < ecc_cutoff]
    # Sort by area
    start_spots = sorted(spot_regionprops, key=lambda r: r.area, reverse=True)[:n * spc]
    # Save image with bounding boxes, if a filepath is provided
    if isinstance(save_plot, str):
        fig, ax = plot_rois(img, [s.bbox for s in spot_regionprops])
        plt.savefig(save_plot)
    if show_plot:
        plot_rois(img, [s.bbox for s in spot_regionprops], show=True)
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
        return spots, []

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
    spots_binary = spots > filters.threshold_li(np.where(spots_filled, spots, 0))
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
                                                   threshold="binary", **kwargs)
                        # TODO: Only add the new bounding box if it doesn't overlap with the previous one
                        bboxes.append(adj_spot_box)
                    else:
                        y_new = spot_box[0] + i * step_size
                        adj_spot_box = refine_spot(spots_binary,
                                                   (y_new, spot_box[1], y_new + spot_diam, spot_box[3]),
                                                   threshold="binary", **kwargs)
                        bboxes.append(adj_spot_box)
    bboxes = [expand_bounds(spot_regions, b) for b in bboxes]
    # Check for bounding boxes that encompass most of the image area
    bboxes = [b for b in bboxes
              if radius_small < b[2] - b[0] < 0.5 * img.shape[0]
              and radius_small < b[3] - b[1] < 0.5 * img.shape[1]]
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


def refine_spot(img, bbox, threshold='binary', tolerance=1, max_shift=0.15, max_iter=20, **kwargs):
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
    bbox = refine_spot_recursive(0, img, bbox, get_box_center(bbox), "num_regions", threshold, tolerance,
                                 max_shift * (bbox[2] - bbox[0]), max_iter, **kwargs)
    return refine_spot_recursive(0, img, bbox, get_box_center(bbox), "area", threshold, tolerance,
                                 max_shift * (bbox[2] - bbox[0]), max_iter, **kwargs)


def refine_spot_recursive(curr_iter, img, bbox, start_coords, metric, threshold,
                          tolerance, max_shift, max_iter, **kwargs):
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

    # Normalize spot size by adding unaltered spots under the radius value to eroded spots above the radius value
    radius = kwargs['radius'] if 'radius' in kwargs else 10
    spot = morphology.white_tophat(spot, morphology.disk(radius)) + morphology.erosion(spot, morphology.disk(radius))
    # Get the centroids and areas of all regions
    spot_data = regionprops_table(spot, properties=('label', 'centroid', 'area'))
    spot_center = np.mean(np.row_stack([spot_data["centroid-0"], spot_data["centroid-1"]]), axis=1)
    # Check for NaNs (i.e. lack of points)
    if np.isnan(spot_center).any():
        print("Empty spotting region detected")
        return bbox
    curr_area = np.sum(spot_data['area'])
    # Make a new bounding box centered on the average of the centroids, and extract new regions
    bbox_new = new_bbox_at(spot_center, bbox, img, mode="shift_center")
    spot = extract(img, bbox_new) > threshold
    spot, n_regions_new = label(spot, return_num=True, connectivity=tolerance)
    new_area = np.sum(regionprops_table(spot, properties=('area',))['area'])
    # Recursively evaluate as long as the new bounding box contains more regions without decreasing the area
    growth = max((bbox_new[3] - bbox_new[1]) / (bbox[3] - bbox[1]), (bbox_new[2] - bbox_new[0]) / (bbox[2] - bbox[0]))
    max_growth = kwargs['max_growth'] if 'max_growth' in kwargs else 2
    if metric == "num_regions":
        displacement = np.sqrt((get_box_center(bbox_new)[0] - start_coords[0]) ** 2 +
                               (get_box_center(bbox_new)[1] - start_coords[1]) ** 2)
        if curr_iter < max_iter and n_regions_new > n_regions \
                and new_area >= curr_area \
                and displacement <= max_shift \
                and growth <= max_growth:
            return refine_spot_recursive(curr_iter + 1, img, bbox_new, start_coords, "num_regions", threshold,
                                         tolerance, max_shift, max_iter)
        else:
            return bbox
    elif metric == "area":
        if curr_iter < max_iter and n_regions_new >= n_regions \
                and new_area > curr_area \
                and growth <= max_growth:
            return refine_spot_recursive(curr_iter + 1, img, bbox_new, start_coords, "area",
                                         threshold, tolerance, max_shift, max_iter)
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
