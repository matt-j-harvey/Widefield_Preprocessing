import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class masking_window(QWidget):

    def __init__(self, session_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0}
        self.session_list = session_list
        self.current_template_index = 0
        self.current_matching_index = 0

        # Cross Display View
        self.cross_display_view_widget = QWidget()
        self.cross_display_view_widget_layout = QGridLayout()
        self.cross_display_view = pyqtgraph.ImageView()
        self.cross_display_view.ui.histogram.hide()
        self.cross_display_view.ui.roiBtn.hide()
        self.cross_display_view.ui.menuBtn.hide()
        self.cross_display_view_widget_layout.addWidget(self.cross_display_view, 0, 0)
        self.cross_display_view_widget.setLayout(self.cross_display_view_widget_layout)
        self.cross_display_view_widget.setFixedWidth(608)
        self.cross_display_view_widget.setFixedHeight(600)
        #self.cross_display_view.setImage(max_projection)

        # Regressor Display View
        self.regressor_display_view_widget = QWidget()
        self.regressor_display_view_widget_layout = QGridLayout()
        self.regressor_display_view = pyqtgraph.ImageView()
        self.regressor_display_view.ui.histogram.hide()
        self.regressor_display_view.ui.roiBtn.hide()
        self.regressor_display_view.ui.menuBtn.hide()
        self.regressor_display_view_widget_layout.addWidget(self.regressor_display_view, 0, 0)
        self.regressor_display_view_widget.setLayout(self.regressor_display_view_widget_layout)
        self.regressor_display_view_widget.setFixedWidth(608)
        self.regressor_display_view_widget.setFixedHeight(600)
        #self.regressor_display_view.setImage(max_projection)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)

        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.template_session_list_widget.addItem(session_name)
            self.matching_session_list_widget.addItem(session_name)

        self.matching_session_list_widget.setCurrentRow(self.current_matching_index)

        # Create Transformation Buttons
        self.left_button = QPushButton("Left")
        self.left_button.clicked.connect(self.move_left)

        self.right_button = QPushButton("Right")
        self.right_button.clicked.connect(self.move_right)

        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.move_up)

        self.down_button = QPushButton("Down")
        self.down_button.clicked.connect(self.move_down)

        self.rotate_clockwise_button = QPushButton("Rotate Clockwise")
        self.rotate_clockwise_button.clicked.connect(self.rotate_clockwise)

        self.rotate_counterclockwise_button = QPushButton("Rotate Counterclockwise")
        self.rotate_counterclockwise_button.clicked.connect(self.rotate_counterclockwise)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel("x: 0")
        self.y_label = QLabel("y: 0")
        self.angle_label = QLabel("angle: 0")

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.template_session_label,          0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label,          0, 2, 1, 2)

        # Add Display Views
        self.layout.addWidget(self.regressor_display_view_widget,   1, 1, 25, 1)
        self.layout.addWidget(self.cross_display_view_widget,       1, 3, 25, 1)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget,   1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget, 1, 2, 25, 1)

        # Add Transformation Controls
        self.layout.addWidget(self.left_button,                     2,  6, 1, 1)
        self.layout.addWidget(self.right_button,                    3,  6, 1, 1)
        self.layout.addWidget(self.up_button,                       4,  6, 1, 1)
        self.layout.addWidget(self.down_button,                     5,  6, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  6, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  6, 1, 1)
        self.layout.addWidget(self.x_label,                         8,  6, 1, 1)
        self.layout.addWidget(self.y_label,                         9,  6, 1, 1)
        self.layout.addWidget(self.angle_label,                     10, 6, 1, 1)
        self.layout.addWidget(self.map_button,                      11, 6, 1, 1)

        # Add ROI
        horizontal_line_coords = [[304, 500], [304, 25]]
        vertical_line_coords = [[202, 100], [402, 100]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)

        self.cross_display_view.addItem(self.horizontal_roi)
        self.cross_display_view.addItem(self.vertical_roi)

    def draw_images(self):
        # Rotate
        angle = self.variable_dictionary['rotation']
        x_shift = self.variable_dictionary['x_shift']
        y_shift = self.variable_dictionary['y_shift']

        transformed_max_projection = np.copy(self.max_projection)
        transformed_max_projection = ndimage.rotate(transformed_max_projection, angle, reshape=False)
        transformed_max_projection = np.roll(a=transformed_max_projection, axis=0, shift=y_shift)
        transformed_max_projection = np.roll(a=transformed_max_projection, axis=1, shift=x_shift)

        self.cross_display_view.setImage(transformed_max_projection)

    def move_left(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] - 1
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] + 1
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] - 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] + 1
        self.angle_label.setText("Angle: " + str(self.variable_dictionary['rotation']))
        self.draw_images()


    def set_alignment(self):
        # Get Save Directory
        save_directory = os.path.join(self.session_directory, "Static_Cross_Alignment_Dictionary.npy")
        print("Save Directory", save_directory)

        # Create Transformation Array
        transformation_dictionary = {}
        transformation_dictionary["rotation"] = self.variable_dictionary["rotation"]
        transformation_dictionary["y_shift"] = self.variable_dictionary["y_shift"]
        transformation_dictionary["x_shift"] = self.variable_dictionary["x_shift"]

        np.save(save_directory, transformation_dictionary)


    def load_matching_session(self):
        current_session_index = self.matching_session_list_widget.currentRow()
        print("Current session index", current_session_index)
        current_session = self.session_list[current_session_index]
        max_projection = np.load(os.path.join(current_session, "max_projection.npy"))
        self.max_projection = max_projection
        self.draw_images()
        #self.cross_display_view.setImage(max_projection)


def align_sessions(session_list):

    app = QApplication(sys.argv)

    window = masking_window(session_list)
    window.show()

    app.exec_()


session_list = [

    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_04_29_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_01_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_03_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_05_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_07_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK14.1A/2021_05_09_Discrimination_Imaging",

    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_04_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_06_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_08_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_10_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_12_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_14_Discrimination_Imaging",
    "/media/matthew/Expansion/Widefield_Analysis/NXAK4.1B/2021_02_22_Discrimination_Imaging"

]
align_sessions(session_list)
