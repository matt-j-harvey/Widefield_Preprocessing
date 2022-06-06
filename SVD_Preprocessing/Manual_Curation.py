import pyqtgraph

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib.pyplot as plt
import os
import h5py
import numpy as np
import sys

sys.path.append("/home/matthew/Documents/Github_Code/Widefield_Preprocessing")
import Widefield_General_Functions


session_list = [#r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
                ]

for base_directory in session_list:

    # Load Data
    data_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    data_container = h5py.File(data_file, 'r')

    print("Data Container", data_container)
    print("Data container", data_container.keys())