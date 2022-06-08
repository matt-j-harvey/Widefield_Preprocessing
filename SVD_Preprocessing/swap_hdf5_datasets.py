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



pyqtgraph.setConfigOptions(imageAxisOrder='row-major')



def load_generous_mask(home_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(home_directory + "/Generous_Mask.npy")

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width


class swapping_window(QWidget):

    def __init__(self, session_list):
        super().__init__()
        self.title = 'Swap Files'
        self.current_session = 0
        self.blue_image = None
        self.violet_image = None
        self.session_list = session_list
        self.base_directory = self.session_list[self.current_session]
        #self.setGeometry(0, 0, 500, 500)

        # Create Display View Widgets
        self.blue_display_view_widget = QWidget()
        self.blue_display_view_widget_layout = QGridLayout()
        self.blue_display_view = pyqtgraph.ImageView()
        self.blue_display_view.ui.histogram.hide()
        self.blue_display_view.ui.roiBtn.hide()
        self.blue_display_view.ui.menuBtn.hide()
        self.blue_display_view_widget_layout.addWidget(self.blue_display_view, 0, 0)
        self.blue_display_view_widget.setLayout(self.blue_display_view_widget_layout)
        self.blue_display_view_widget.setMinimumWidth(400)
        
        self.violet_display_view_widget = QWidget()
        self.violet_display_view_widget_layout = QGridLayout()
        self.violet_display_view = pyqtgraph.ImageView()
        self.violet_display_view.ui.histogram.hide()
        self.violet_display_view.ui.roiBtn.hide()
        self.violet_display_view.ui.menuBtn.hide()
        self.violet_display_view_widget_layout.addWidget(self.violet_display_view, 0, 0)
        self.violet_display_view_widget.setLayout(self.violet_display_view_widget_layout)
        self.violet_display_view_widget.setMinimumWidth(400)

        # Add Directory List
        self.session_list_widget = QListWidget()
        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.session_list_widget.addItem(session_name)
        self.session_list_widget.setCurrentRow(self.current_session)

        # Create Labels
        self.blue_label = QLabel("BLue Image")
        self.violet_label = QLabel("Violet Image")

        # Create Buttons
        self.swap_button = QPushButton("Swap Files")
        self.dont_swap_button = QPushButton("Dont Swap Files")
        self.swap_button.clicked.connect(self.swap_contents)
        self.dont_swap_button.clicked.connect(self.set_next_session)

        # Create Layout
        self.layout = QGridLayout()

        self.layout.addWidget(self.blue_label,                  0, 0, 1, 1)
        self.layout.addWidget(self.blue_display_view_widget,    1, 0, 1, 1)

        self.layout.addWidget(self.violet_label,                0, 1, 1, 1)
        self.layout.addWidget(self.violet_display_view_widget,  1, 1, 1, 1)

        self.layout.addWidget(self.session_list_widget,         0, 2, 3, 1)

        self.layout.addWidget(self.swap_button,                 3, 0, 1, 1)
        self.layout.addWidget(self.dont_swap_button,            3, 1, 1, 1)

        self.setLayout(self.layout)

        # Load First Session
        #self.load_data()


    def load_data(self):
        blue_image = np.load(os.path.join(self.base_directory, "Blue_Image_Sample.npy"))
        violet_image = np.load(os.path.join(self.base_directory, "Violet_Image_Sample.npy"))
        self.blue_display_view.setImage(blue_image)
        self.violet_display_view.setImage(violet_image)


    def set_next_session(self):
        self.current_session += 1
        if self.current_session < len(self.session_list):
            self.base_directory = self.session_list[self.current_session]
            self.session_list_widget.setCurrentRow(self.current_session)
            self.load_data()


    def swap_contents(self):

        # Load Data
        data_file = os.path.join(self.base_directory, "Motion_Corrected_Mask_Data.hdf5")
        data_container = h5py.File(data_file, 'r+')


        data_container['New_Violet_Data'] = data_container['Blue_Data']
        data_container['New_Blue_Data'] = data_container['Violet_Data']

        del data_container['Blue_Data']
        del data_container['Violet_Data']

        data_container['Violet_Data'] = data_container['New_Violet_Data']
        data_container['Blue_Data'] = data_container['New_Blue_Data']


        del data_container['New_Blue_Data']
        del data_container['New_Violet_Data']

        print("Swapped")
        self.set_next_session()


def get_image_samples(base_directory):

    # Load Data
    data_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    data_container = h5py.File(data_file, 'r')
    blue_array = data_container["Blue_Data"]
    violet_array = data_container["Violet_Data"]

    # Load Mask
    indicies, image_height, image_width = load_generous_mask(base_directory)
    print("Blue Array", blue_array)

    # Take Sample of Data
    blue_sample = np.array(blue_array[:, 1000:2000])
    violet_sample = np.array(violet_array[:, 1000:2000])
    print("Blue Sample", np.shape(blue_sample))

    # Get Max Projections
    blue_sample = np.max(blue_sample, axis=1)
    violet_sample = np.max(violet_sample, axis=1)

    # Reconstruct Images
    blue_image = Widefield_General_Functions.create_image_from_data(blue_sample, indicies, image_height, image_width)
    violet_image = Widefield_General_Functions.create_image_from_data(violet_sample, indicies, image_height, image_width)

    # Save Images
    np.save(os.path.join(base_directory, "Blue_Image_Sample.npy"), blue_image)
    np.save(os.path.join(base_directory, "Violet_Image_Sample.npy"), violet_image)


app = QApplication(sys.argv)

"""
            r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_15_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_16_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_17_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_19_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_23_Discrimination_Imaging",
                r"/media/matthew/Seagate Expansion Drive1/Longitudinal_Analysis/NRXN78.1D/2020_11_25_Discrimination_Imaging"

"""

session_list = [
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_02_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_04_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_06_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_08_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_10_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_12_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_14_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_16_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_18_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_23_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_25_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_02_27_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_01_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_03_Discrimination_Imaging",
r"/media/matthew/Seagate Expansion Drive1/NXAK4.1A/2021_03_05_Discrimination_Imaging",
]

for session in session_list:
    get_image_samples(session)

window = swapping_window(session_list)
window.show()

window.load_data()

app.exec_()