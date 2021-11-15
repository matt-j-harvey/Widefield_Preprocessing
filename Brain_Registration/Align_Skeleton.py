import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal, ndimage, stats
import os


from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph
import sys


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


def get_line(x1, y1, x2, y2):

    # Setup initial conditions
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    #swapped = False
    if swapped:
        points.reverse()
    return points




class session_matching_window(QWidget):

    def __init__(self, parent=None):
        super(session_matching_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Atlas Alignment")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.base_directory = None
        self.max_projection = None
        self.annonated_matrix = None
        self.skeleton_image = None
        self.midline_start_coords = [0, 0]
        self.midline_stop_coords  = [0, 0]
        self.coronal_start_coords = [0, 0]
        self.coronal_stop_coords  = [0, 0]

        # Create Figures
        self.display_view_widget = QWidget()
        self.display_view_widget_layout = QGridLayout()
        self.display_view = pyqtgraph.ImageView()
        self.display_view.ui.histogram.hide()
        self.display_view.ui.roiBtn.hide()
        self.display_view.ui.menuBtn.hide()
        self.display_view_widget_layout.addWidget(self.display_view, 0, 0)
        self.display_view_widget.setLayout(self.display_view_widget_layout)
        self.display_view_widget.setMinimumWidth(800)
        self.display_view_widget.setMinimumHeight(800)

        # Add ROIs
        self.midline_start_roi = pyqtgraph.ROI([0, 0],   [20, 20])
        self.midline_stop_roi  = pyqtgraph.ROI([50, 0],  [20, 20])
        self.coronal_start_roi = pyqtgraph.ROI([0, 50],  [20, 20])
        self.coronal_stop_roi  = pyqtgraph.ROI([50, 50], [20, 20])

        self.midline_start_roi.sigRegionChanged.connect(self.get_midline_start_roi_coords)
        self.midline_stop_roi.sigRegionChanged.connect(self.get_midline_stop_roi_coords)
        self.coronal_start_roi.sigRegionChanged.connect(self.get_coronal_start_roi_coords)
        self.coronal_stop_roi.sigRegionChanged.connect(self.get_coronal_stop_roi_coords)

        self.display_view.addItem(self.midline_start_roi)
        self.display_view.addItem(self.midline_stop_roi)
        self.display_view.addItem(self.coronal_start_roi)
        self.display_view.addItem(self.coronal_stop_roi)

        # Add Labels
        self.midline_start_coords_label = QLabel()
        self.midline_stop_coords_label  = QLabel()
        self.coronal_start_coords_label = QLabel()
        self.coronal_stop_coords_label  = QLabel()


        # Create Add Session Button
        self.load_max_projection_button = QPushButton("Select Brain Image")
        self.load_max_projection_button.clicked.connect(self.load_max_projection)
        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)


        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add List Widget
        self.layout.addWidget(self.load_max_projection_button,  0, 0, 1, 2)
        self.layout.addWidget(self.midline_start_coords_label,  1, 0, 1, 1)
        self.layout.addWidget(self.midline_stop_coords_label,   2, 0, 1, 1)
        self.layout.addWidget(self.coronal_start_coords_label,  3, 0, 1, 1)
        self.layout.addWidget(self.coronal_stop_coords_label,   4, 0, 1, 1)
        self.layout.addWidget(self.map_button,                  9, 0, 1, 2)
        self.layout.addWidget(self.display_view_widget,         0, 2, 10, 10)

        self.show()


    def get_midline_start_roi_coords(self):
            data2, xdata = self.midline_start_roi.getArrayRegion(self.max_projection, self.display_view.imageItem, returnMappedCoords=True)
            roi_center = xdata[:, 10, 10]
            self.midline_start_coords = roi_center
            self.midline_start_coords_label.setText(str(self.midline_start_coords))
            self.update_coordinates()

    def get_midline_stop_roi_coords(self):
            data2, xdata = self.midline_stop_roi.getArrayRegion(self.max_projection, self.display_view.imageItem, returnMappedCoords=True)
            roi_center = xdata[:, 10, 10]
            self.midline_stop_coords = roi_center
            self.midline_stop_coords_label.setText(str(self.midline_stop_coords))
            self.update_coordinates()

    def get_coronal_start_roi_coords(self):
            data2, xdata = self.coronal_start_roi.getArrayRegion(self.max_projection, self.display_view.imageItem, returnMappedCoords=True)
            roi_center = xdata[:, 10, 10]
            self.coronal_start_coords = roi_center
            self.coronal_start_coords_label.setText(str(self.coronal_start_coords))
            self.update_coordinates()

    def get_coronal_stop_roi_coords(self):
            data2, xdata = self.coronal_stop_roi.getArrayRegion(self.max_projection, self.display_view.imageItem, returnMappedCoords=True)
            roi_center = xdata[:, 10, 10]
            self.coronal_stop_coords = roi_center
            self.coronal_stop_coords_label.setText(str(self.coronal_stop_coords))
            self.update_coordinates()


    def update_coordinates(self):

        current_image = np.copy(self.max_projection)

        # Create Blank Skeleton Image
        image_height = np.shape(current_image)[0]
        image_width = np.shape(current_image)[1]
        skeleton_image = np.zeros((image_height, image_width))

        # Draw Midline Axis
        y1 = int(self.midline_start_coords[0])
        x1 = int(self.midline_start_coords[1])
        y2 = int(self.midline_stop_coords[0])
        x2 = int(self.midline_stop_coords[1])

        points = get_line(x1, y1, x2, y2)
        for point in points:
            current_image[point[1], point[0], [2]] = 1
            skeleton_image[point[1], point[0]] = 1

        # Draw Coronal Axis
        x1 = int(self.coronal_start_coords[1])
        y1 = int(self.coronal_start_coords[0])
        x2 = int(self.coronal_stop_coords[1])
        y2 = int(self.coronal_stop_coords[0])

        points = get_line(x1, y1, x2, y2)
        for point in points:
            current_image[point[1], point[0], [1]] = 1
            skeleton_image[point[1], point[0]] = 1

        self.display_view.setImage(current_image)
        self.skeleton_image = skeleton_image

    def load_max_projection(self):

        # Get File Location With Q Dialog
        max_projection_file_location = QFileDialog.getOpenFileName(self, "Select Max Projection")[0]
        print(max_projection_file_location)

        # Check Max Projection
        max_projection = np.load(max_projection_file_location, allow_pickle=True)
        max_projection = np.divide(max_projection, np.max(max_projection))
        max_projection = np.expand_dims(max_projection, axis=2)
        self.max_projection = np.repeat(a=max_projection, repeats=3, axis=2)
        self.display_view.setImage(self.max_projection)

        # Get New Directory + Session Name
        self.base_directory, max_projection_file_name = os.path.split(max_projection_file_location)

        # Get Max Projection Shape
        image_height = np.shape(self.max_projection)[0]
        image_width = np.shape(self.max_projection)[1]


        self.annonated_matrix = np.zeros((image_height, image_width, 3))


    def set_alignment(self):
        save_directory = self.base_directory + "/Skeleton.npy"
        np.save(save_directory, self.skeleton_image)
        print("Skeleton Saved")




if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = session_matching_window()
    window.show()
    app.exec_()


