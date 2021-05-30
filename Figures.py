from PlateCV.Structures import Plate
from typing import List
import numpy as np


def make_composite_figure(plates: List[Plate], layout=None):
    if layout is None:
        all_layouts = [p.layout for p in plates]
        layout = np.row_stack(all_layouts).shape
    else:
        layout = np.array(layout).shape
    print(layout)