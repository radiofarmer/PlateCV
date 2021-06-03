from PlateCV.Structures import ROI
from skimage import morphology, measure, io, img_as_uint
import numpy as np
import matplotlib.pyplot as plt


def measure_coverage(roi: ROI):
    if roi is None:
        return 0.
    else:
        return np.sum(roi.mask) / roi.area


def measure_optical_density(roi: ROI, return_image=False, save_as=None, **kwargs):
    if roi is None:
        return (0., None) if return_image else 0.
    # Remove background
    mask = roi.mask
    mask_labeled = measure.label(mask, **kwargs)
    colonies = np.where(mask_labeled != 0, roi.data, 0)
    print(roi.threshold)
    if isinstance(save_as, str):
        io.imsave(save_as, img_as_uint(colonies))

    # Calculate the OD for a theoretically all-white image
    max_od = colonies.size * 2**(8 if colonies.dtype == np.uint8 else 16)
    if return_image:
        return np.sum(colonies) / max_od if colonies.size != 0 else 0., colonies
    else:
        return np.sum(colonies) / max_od if colonies.size != 0 else 0


def count_colonies(roi: ROI, filter_func=None, return_image=False, **kwargs):
    """
    Returns the number of distinct regions in an ROI
    :param roi: The region of interest to analyze (ROI object)
    :param filter_func: A function which takes 1) the raw image data minus the background and 2) the image with
    labeled regions, and returns a filtered image
    :param return_image: Return the labels along with the count?
    :param kwargs: Parameters to be passed to skimage.measure.label
    :return:
    """
    if roi is None:
        return (0., None) if return_image else 0.
    mask = roi.mask
    mask_labeled = measure.label(mask, **kwargs)
    if filter_func is not None:
        colonies = np.where(mask_labeled != 0, roi.data, 0)
        mask_labeled = filter_func(colonies, mask_labeled)
    count = np.unique(mask_labeled).shape[0]
    if return_image:
        return count, mask_labeled
    else:
        return count


def average_colony_size(roi: ROI, filter_func=None, return_image=False, save_as=None, **kwargs):
    if roi is None:
        return (0., None) if return_image else 0.
    count, labels = count_colonies(roi, filter_func=filter_func, return_image=True, **kwargs)

    if isinstance(save_as, str):
        io.imsave(save_as, img_as_uint(labels))

    return (np.sum(roi.mask / count), labels) if return_image else np.sum(roi.mask) / count
