from typing import Union
import numpy as np
from skimage import filters
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
        self._threshold = None

    def __getitem__(self, key):
        return self._rois[key] if key in self._rois else None

    def __bool__(self):
        return len(self._rois) > 0

    def set_rois(self, img, bboxes, threshold='li'):
        if bboxes:
            self._rois = {i: ROI(extract(img, box), self._label + f"_D{i}")
                          for i, box in enumerate(bboxes)}
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

    def plot_rois(self, func=None):
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
            plt.axes.xaxis.set_ticks([])
            plt.axes.yaxis.set_ticks([])
            plt.title(self._label)
        plt.show()

    @property
    def properties(self):
        return self._props

    @property
    def label(self):
        return self._label


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
        self._img = None
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

    def read_spotting_data(self, img, spots, manual_range=False, save_path=None, show_plot=False, threshold='li'):
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
                self._constructs[const] = Construct(const, self._num_dilutions)
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
        # Cut out only the area containing spots and adjust the spot coordinates accordingly
        img, spots = extract(img, bounds, spots)
        self._img = img
        # Measure a global threshold value
        self._threshold = Thresholds[threshold](img)
        fig, ax = plot_rois(img, spots, False)

        # Check for missing rows or columns
        img_h, img_w = img.shape
        nrows, ncols = self._layout.shape
        spot_diam = np.mean(np.concatenate(([b[2] - b[0] for b in spots], [b[3] - b[1] for b in spots])))
        centers = [get_box_center(b) for b in spots]
        adjacent_dists = [d for d in spatial.distance_matrix(centers, centers).ravel() if
                          spot_diam < d < 1.5 * spot_diam]
        padding = np.mean(adjacent_dists) - spot_diam
        if self._horizontal:
            row_start, row_end = 0, nrows
            col_start, col_end = 0, ncols
            while nrows > 1 and spot_diam * nrows + (nrows - 1) * padding > img_h:
                nrows -= 1
                print(f"Found missing row in plate {self._group}")
                if manual_range:
                    row_start = input(f"Specify row start offset (currently {row_start}):")
                    row_end = input(f"Specify final row number (currently {row_end}:")
                else:
                    row_end = nrows
            while ncols > 1 and spot_diam * ncols * self._num_dilutions + (
                    ncols * self._num_dilutions - 1) * padding > img_w:
                ncols -= 1
                print(f"Found missing column in plate {self._group}")
                if manual_range:
                    col_start = input(f"Specify col start offset (currently {col_start}):")
                    col_end = input(f"Specify final col number (currently {col_end}:")
                else:
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
        else:
            # TODO: Vertical layout
            while nrows > 1 and spot_diam * nrows * self._num_dilutions + (
                    nrows * self._num_dilutions - 1) * padding > img_h:
                nrows -= 1
                print(f"Found missing row in plate {self._group}")
            while ncols > 1 and spot_diam * ncols + (ncols - 1) * padding > img_h:
                ncols -= 1
                print(f"Found missing row in plate {self._group}")

        if save_path is not None:
            ax.set_title(f"Plate {self._group}, {self._condition}")
            plt.savefig(save_path)
        if show_plot:
            plt.show()
        else:
            plt.close()
        # Construct name checklist
        missing_constructs = [c for c in self._construct_names if c is not None]
        # Create construct objects
        for s, c in zip(sectors, self._construct_names):
            # Get the bounding boxes whose center point is located within the sector bounding box
            spots_in_sector = [b for b in spots if is_in_bbox(get_box_center(b), s)]
            print(f"Found {len(spots_in_sector)} spots in construct '{c}'")
            constr = Construct(c, self._num_dilutions)
            constr.set_rois(img, spots_in_sector, threshold=self.threshold)
            self._constructs[c] = constr
            # Remove construct from 'missing' list if ROIs are found
            if spots_in_sector:
                missing_constructs.remove(c)
        # Add empty construct objects for spots that were not found on the plate
        for c in missing_constructs:
            self._constructs[c] = Construct(c, self._num_dilutions)
        self._sectors = sectors

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
                    ax[r, c].imshow(extract(self._img, self._sectors[r + c*layout[0]]), **kwargs)
                except IndexError:
                    ax[r, c].axis('off')
                ax[r, c].xaxis.set_ticks([])
                ax[r, c].yaxis.set_ticks([])
                ax[r, c].set_ylabel(self._construct_names[r + c*layout[0]].replace("_", " "))
        return fig, ax

    @property
    def labels(self):
        return [c for c in self._construct_names if c is not None]

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
