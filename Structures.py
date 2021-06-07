from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, io, img_as_uint, morphology
from skimage.feature import canny
from scipy import spatial
from PlateCV.Utils import *

Thresholds = {
    "li": filters.threshold_li,
    "otsu": filters.threshold_otsu,
    "yen": filters.threshold_yen
}


class ROI:
    def __init__(self, img_data: Union[np.ndarray, list], label=None, **kwargs):
        self._multichannel = type(img_data) == list
        self._label = label
        self._data = img_data
        self._threshold = None

        if self._multichannel:
            self._channel_names = kwargs['channels'] if 'channels' in kwargs else None
        else:
            self._channel_names = None

    def set_threshold(self, val):
        if isinstance(val, int) or isinstance(val, float):
            self._threshold = float(val)
        else:
            self._threshold = Thresholds[val](self.data)

    @property
    def area(self):
        return self.data.shape[0] * self.data.shape[1]

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    @property
    def mask(self):
        return (self.data > self._threshold).astype(np.int8)

    @property
    def threshold(self):
        return self._threshold


class Construct:

    def __init__(self, label: str, num_dilutions: int, data=None, dilution_amt=10, **properties):
        self._label = label if label is not None else ""
        self._num_dilutions = num_dilutions
        self._props = properties
        self._rois = {lab: d for lab, d in enumerate(data)} if data is not None else []
        self._roi_bounds = []
        self._threshold = None

    def __getitem__(self, key):
        return self._rois[key] if key in self._rois else None

    def __bool__(self):
        return len(self._rois) > 0

    def set_rois(self, img, bboxes, threshold: Union[str, int] = 'li'):
        if bboxes:
            self._roi_bounds = bboxes
            self._rois = {i: ROI(extract(img, box), self._label + f"_D{i}")
                          for i, box in enumerate(self._roi_bounds)}
        else:
            self._rois = {}
            return
        # Calculate global construct threshold
        all_pixels = np.concatenate([itm[1].data.ravel() for itm in self._rois.items()])
        if isinstance(threshold, str):
            self._threshold = Thresholds[threshold](all_pixels)
        else:
            self._threshold = threshold
        # Set the threshold of all ROIs to the global threshold value
        for k in self._rois.keys():
            self._rois[k].set_threshold(self._threshold)

    def plot_rois(self, func=None, save_as=None):
        if not self._rois:
            print("No data to display")
            return
        elif func:
            # Apply a processing function to the ROI (remember that Measure functions take ROI objects)
            img_data = [func(r[1]) for r in self._rois.items()]
        else:
            img_data = [r[1].data for r in self._rois.items()]

        if len(self._rois) > 1:
            fig, axes = plt.subplots(1, len(img_data))
            for a, roi in zip(axes, img_data):
                a.imshow(roi)
                a.xaxis.set_ticks([])
                a.yaxis.set_ticks([])
            fig.suptitle(self._label)
        else:
            plt.imshow(img_data[0])
            plt.xticks([])
            plt.yticks([])
            plt.title(self._label)

        if save_as is not None:
            plt.savefig(save_as + (".tif" if ".tif" not in save_as else ""))
        else:
            plt.show()

    @property
    def properties(self):
        return self._props

    @property
    def label(self):
        return self._label

    @property
    def roi_bboxes(self):
        return self._roi_bounds


class Plate:
    def __init__(self, layout, spots_per_construct, condition, group=None, orientation='horizontal'):
        # Experimental Parameters
        self._layout = layout if isinstance(layout, np.ndarray) else np.array(layout)
        self._construct_names = [c for c in self._layout.ravel()]
        self._num_dilutions = spots_per_construct
        self._condition = condition
        self._group = group
        self._horizontal = True if orientation.lower() == 'horizontal' else False

        # Construct data
        self._img_cropped = None
        self._img_full = None
        self._spot_region = None
        self._threshold = 0
        self._constructs = {}
        self._sectors = []

        # List for logging potential issues discovered in data
        self._log = []

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._constructs[key]
        elif isinstance(key, int):
            return self._constructs[self._construct_names[key]]

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self):
        if self._n < len(self._constructs):
            constr = self._constructs[self.labels[self._n]]
            self._n += 1
            return constr
        else:
            raise StopIteration

    def read_spotting_data(self, img, spots, manual_range=False, save_rois=None, save_threshold=None,
                           show_plot=False, threshold='li'):
        """
        Generate construct objects from a segmented plate image
        :param img: The image data of the whole relevant area of the plate, as a numpy array
        :param spots: Coordinates of bounding boxes of all spots
        :param manual_range: Specifies whether to ask for manual input of a new range if fewer rows or columns are
        found than expected.
        :param show_plot: Plot the extracted region, with spot and sector bounding boxes
        :param threshold: The method for measuring the plate global threshold value
        :return: None
        """
        # First, check whether any spots are present at all
        if not len(spots):
            for const in self.labels:
                # Cut out only the area containing spots and adjust the spot coordinates accordingly
                self._img_cropped = img
                plt.imshow(img)
                if save_rois is not None:
                    plt.title(f"Plate {self._group}, {self._condition}")
                    plt.savefig(save_rois)
                    plt.close()
                else:
                    plt.show()
                self._constructs[const] = Construct(const, self._num_dilutions)

                # Create empty sectors
                self._sectors = [(0, 0, 0, 0) for _ in range(len(self._construct_names))]
            return

        # List to hold the regions that (hopefully) contain spots derived from the same construct
        sectors = []
        # Crop to area containing bounding boxes
        bounds = [
            np.min([b[0] for b in spots]),
            np.min([b[1] for b in spots]),
            np.max([b[2] for b in spots]),
            np.max([b[3] for b in spots])
        ]
        self._spot_region = bounds
        # Save the full image
        self._img_full = img
        # Cut out only the area containing spots and adjust the spot coordinates accordingly
        img, spots = extract(img, bounds, spots)
        self._img_cropped = img

        # Measure a global threshold value based on the spot ROIs on a reconstructed image
        # self._threshold = Thresholds[threshold](np.concatenate([extract(img, s).ravel() for s in spots]))
        self._threshold = Thresholds[threshold](img)
        if isinstance(save_threshold, str):
            io.imsave(save_threshold + (".tif" if ".tif" not in save_threshold else ""),
                      img_as_uint(img > self._threshold))

        # Prepare ROI plot objects
        fig, ax = plt.subplots()
        ax.imshow(img)

        # Check for missing rows or columns by comparing the size of the image to the predicted size based
        # on the space between spots
        img_h, img_w = img.shape
        nrows, ncols = self._layout.shape
        row_start, row_end = 0, nrows
        col_start, col_end = 0, ncols
        spot_diam = np.mean(np.concatenate(([b[2] - b[0] for b in spots], [b[3] - b[1] for b in spots])))
        centers = [get_box_center(b) for b in spots]
        # Measure the distances between adjacent spots
        all_dists = spatial.distance_matrix(centers, centers)
        adjacent_dists = [d for d in all_dists.ravel() if
                          spot_diam < d < 1.5 * spot_diam]
        # If there are no adjacent spots, see if separate columns can be identified
        if len(adjacent_dists) == 0 and len(spots) > 1:
            idx = np.unravel_index(np.argmax(all_dists), all_dists.shape)
            b0, b1 = (spots[i] for i in idx)
            sep = int(input(f"How many {'columns' if self._horizontal else 'rows'} have visible spots?"))
            if sep == 0:
                padding = 0
            else:
                if self._horizontal:
                    padding = (get_box_center(b1)[1] - get_box_center(b0)[1]) / sep - spot_diam * self._num_dilutions
                    padding /= self._num_dilutions - 1
                else:
                    padding = (get_box_center(b1)[0] - get_box_center(b0)[0]) / sep - spot_diam * self._num_dilutions
                    padding /= self._num_dilutions - 1
        elif len(spots) == 1:
            padding = 0
        else:
            padding = np.mean(adjacent_dists) - spot_diam
        if self._horizontal:
            while nrows > 1 and spot_diam * nrows + (nrows - 1) * padding > img_h + spot_diam/2:
                print(f"Found missing row on plate {self._group}")
                if manual_range:
                    row_start = int(input(f"Specify row start offset (currently {row_start}):"))
                    row_end = int(input(f"Specify zero-indexed final row number (currently {row_end}):"))
                    nrows = row_end + 1 - row_start
                else:
                    # Assume missing row is the last one
                    nrows -= 1
                    row_end = nrows
            while ncols > 1 and spot_diam * ncols * self._num_dilutions + \
                    (ncols * self._num_dilutions - 1) * padding > img_w + spot_diam/2:
                print(f"Found missing column on plate {self._group}")
                if manual_range:
                    col_start = int(input(f"Specify col start offset (currently {col_start}):"))
                    col_end = int(input(f"Specify zero-indexed final col number (currently {col_end}:"))
                    ncols = col_end + 1 - col_start
                else:
                    ncols -= 1
                    col_end = ncols
            for r in range(nrows):
                for c in range(ncols):
                    sectors.append(((r * (spot_diam + padding),
                                     c * (spot_diam * self._num_dilutions + padding * (self._num_dilutions - 1)),
                                     (r + 1) * (spot_diam + padding),
                                     (c + 1) * (spot_diam * self._num_dilutions + padding * (
                                             self._num_dilutions - 1)))))
                    ax.axhline(sectors[-1][0])
                    ax.axvline(sectors[-1][1])
            for i, s in enumerate(sectors):
                ax.text(get_box_center(s)[1], get_box_center(s)[0], self._construct_names[i], color='white')
        else:
            # TODO: Vertical layout
            while nrows > 1 and spot_diam * nrows * self._num_dilutions + (
                    nrows * self._num_dilutions - 1) * padding > img_h:
                nrows -= 1
                print(f"Found missing row in plate {self._group}")
            while ncols > 1 and spot_diam * ncols + (ncols - 1) * padding > img_h:
                ncols -= 1
                print(f"Found missing row in plate {self._group}")
            for r in range(nrows):
                for c in range(ncols):
                    pass

        # Add missing sectors
        if len(sectors) < len(self._construct_names):
            sector_height = np.mean([s[2] - s[0] for s in sectors])
            sector_width = np.mean([s[3] - s[1] for s in sectors])
            nrows, ncols = self._layout.shape  # Get expected number of rows and columns again
            if self._horizontal:
                missing_rows = [r_ for r_ in range(nrows) if r_ < row_start or r_ >= row_end]
                for r in missing_rows:
                    for c in range(col_start, col_end):
                        row_offset = -row_start * sector_height if r < row_start else (r-row_end)*sector_height
                        col_offset = -col_start * sector_height if c < col_start else (c-col_end)*sector_height
                        new_sector = (
                            r * sector_height + row_offset,
                            c * sector_width + col_offset,
                            (r+1) * sector_height + row_offset,
                            (c+1) * sector_width + col_offset
                        )
                        if r < row_start:
                            sectors.insert(r + c, new_sector)
                        else:
                            sectors.append(new_sector)
                missing_cols = [c_ for c_ in range(ncols) if c_ < col_start or c_ > col_end]
                for c in missing_cols:
                    for r in range(nrows):
                        row_offset = -row_start * sector_height if r < row_end else (r - row_end) * sector_height
                        col_offset = -col_start * sector_height if c < col_end else (c - col_end) * sector_height
                        new_sector = (
                            r * sector_height + row_offset,
                            c * sector_width + col_offset,
                            (r+1) * sector_height + row_offset,
                            (c+1) * sector_width + col_offset
                        )
                        if c < col_start:
                            sectors.insert(r + c, new_sector)
                        else:
                            sectors.append(new_sector)

        # Construct name checklist
        missing_constructs = [c for c in self._construct_names if c is not None]
        # Create construct objects (iterate through `_construct_names` rather than `labels`, since
        # the former contains NoneType labels, while the latter does not)
        for s, c in zip(sectors, self._construct_names):
            # Skip sectors without constructs
            if c is None:
                continue
            # Get the bounding boxes whose center point is located within the sector bounding box
            spots_in_sector = sorted([b for b in spots if is_in_bbox(get_box_center(b), s)],
                                     key=lambda b: b[1] if self._horizontal else b[0])
            for i in range(1, len(spots_in_sector)):
                prev_spot = spots_in_sector[i-1]
                spot = spots_in_sector[i]
                if self._horizontal and spot[1] < prev_spot[3]:
                    spots_in_sector[i - 1] = (
                        spot[0], prev_spot[3] + 1, spot[2], spot[3]
                    )
                elif not self._horizontal and spot[0] < prev_spot[2]:
                    spots_in_sector[i - 1] = (
                        prev_spot[2] + 1, *spot[1:]
                    )
            add_rois(ax, spots_in_sector)

            print(f"Found {len(spots_in_sector)} spots in construct '{c}'")
            constr = Construct(c, self._num_dilutions) if c is not None else None
            constr.set_rois(img, spots_in_sector, threshold=self.threshold)
            self._constructs[c] = constr
            # Remove construct from 'missing' list if ROIs are found
            if spots_in_sector:
                missing_constructs.remove(c)
        # Add empty construct objects for spots that were not found on the plate
        for c in missing_constructs:
            self._constructs[c] = Construct(c, self._num_dilutions)
        self._sectors = sectors
        # Save or show the labeled image
        if save_rois is not None:
            ax.set_title(f"Plate {self._group}, {self._condition}")
            plt.savefig(save_rois)
        if show_plot:
            plt.show()
        else:
            plt.close()

    def make_figure(self, layout=None, **kwargs):
        """
        Returns a plot of all the figures on the plate
        :param layout: The arrangement of the constructs in the figure. None uses the experimental layout.
        :return: A matplotlib.pyplot figure
        """
        if layout is None:
            layout = self._layout.shape
        fig, ax = plt.subplots(*layout)
        for r in range(layout[0]):
            for c in range(layout[1]):
                if self._construct_names[r + c*layout[0]] is None:
                    ax[r, c].axis('off')
                    continue
                try:
                    ax[r, c].imshow(extract(self._img_cropped, self._sectors[r + c * layout[0]]), **kwargs)
                except IndexError:
                    ax[r, c].axis('off')
                ax[r, c].xaxis.set_ticks([])
                ax[r, c].yaxis.set_ticks([])
                ax[r, c].set_ylabel(self._construct_names[r + c*layout[0]].replace("_", " "))
        return fig, ax

    def get_sector(self, lab):
        if lab is None:
            return np.array([[]])
        const = self._constructs[lab]
        try:
            region = (
                min([b[0] for b in const.roi_bboxes]),
                min([b[1] for b in const.roi_bboxes]),
                max([b[2] for b in const.roi_bboxes]),
                max([b[3] for b in const.roi_bboxes]),
            )
            return extract(self._img_cropped, region)
        except ValueError:
            print("Empty sector found in get_sector")
            idx = self._construct_names.index(lab)
            sector = self._sectors[idx]
            if self._spot_region is not None:
                y_offset, x_offset = self._spot_region[:2]
            else:
                return np.array([[]])
            region = (
                sector[0] + y_offset,
                sector[1] + x_offset,
                sector[2] + y_offset,
                sector[3] + x_offset
            )
            return extract(self._img_full, region)

    @property
    def constructs(self):
        return (self._constructs[lab] for lab in self.labels)

    @property
    def labels(self):
        return tuple(c for c in self._construct_names if c is not None)

    @property
    def layout(self):
        return self._layout

    @property
    def threshold(self):
        return self._threshold

class Experiment:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name


class SpottingExperiment(Experiment):
    def __init__(self, name: str,
                 conditions: list,
                 constructs: list,
                 num_dilutions: int,
                 layout: list,
                 dilution_direction='horizontal',
                 plates_per_condition=1):
        super().__init__(name)
        self._conditions = conditions
        self._constructs = constructs
        self._num_dilutions = num_dilutions
        self._layout = layout


def assign_constructs(img, bboxes):
    pass
