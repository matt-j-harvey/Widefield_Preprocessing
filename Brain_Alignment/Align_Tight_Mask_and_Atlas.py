import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys


import Registration_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


class masking_window(QWidget):

    def __init__(self, session_list, mask, atlas_outline, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.mask = mask
        self.atlas_outline = atlas_outline
        self.number_of_sessions = len(self.session_list)
        self.current_template_index = 0
        self.current_matching_index = 0
        self.mask_variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'x_scale': 1, 'y_scale': 1}
        self.atlas_variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0, 'x_scale': 0.5, 'y_scale': 0.5}
        self.image_height = 600
        self.image_width = 608
        self.growth_increment = 0.01

        # Get List Of Max Projections
        self.max_projection_list = []
        for session in session_list:
            max_projection = np.load(os.path.join(session, "Blue_Example_Image.npy"))
            upper_bound = np.percentile(max_projection, 99)
            max_projection = np.divide(max_projection, upper_bound)
            max_projection = np.clip(max_projection, a_min=0, a_max=1)
            self.max_projection_list.append(max_projection)

        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):

            variable_dictionary_directory = os.path.join(self.session_list[session_index],"Cluster_Alignment_Dictionary.npy")
            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0}

            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        self.max_projection = self.max_projection_list[0]

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
        self.cross_display_view.setLevels(0, 1)
        #self.cross_display_view.setImage(max_projection)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.template_session_list_widget.currentRowChanged.connect(self.load_template_session)

        for session in self.session_list:
            session_name = session.split('/')[-1]
            self.matching_session_list_widget.addItem(session_name)

        self.matching_session_list_widget.setCurrentRow(self.current_matching_index)

        # Add Mask And Atlas To Template List Widget
        self.template_session_list_widget.addItem("Mask")
        self.template_session_list_widget.addItem("Atlas")
        self.template_session_list_widget.setCurrentRow(0)

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

        self.grow_x_button = QPushButton("Grow X")
        self.grow_x_button.clicked.connect(self.grow_x)

        self.shrink_x_button = QPushButton("Shrink X")
        self.shrink_x_button.clicked.connect(self.shrink_x)

        self.grow_y_button = QPushButton("Grow Y")
        self.grow_y_button.clicked.connect(self.grow_y)

        self.shrink_y_button = QPushButton("Shrink Y")
        self.shrink_y_button.clicked.connect(self.shrink_y)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)

        # Add Labels
        self.x_label = QLabel("x: 0")
        self.y_label = QLabel("y: 0")
        self.angle_label = QLabel("angle: 0")
        self.x_scale_label = QLabel("x_scale: 1")
        self.y_scale_label = QLabel("y_scale: 1")

        # Create and Set Layout]
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Labels
        self.layout.addWidget(self.template_session_label,          0, 0, 1, 2)
        self.layout.addWidget(self.matching_session_label,          0, 2, 1, 2)

        # Add Display Views
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
        self.layout.addWidget(self.grow_x_button,                   8, 6, 1, 1)
        self.layout.addWidget(self.shrink_x_button,                 9, 6, 1, 1)
        self.layout.addWidget(self.grow_y_button,                   10, 6, 1, 1)
        self.layout.addWidget(self.shrink_y_button,                 11, 6, 1, 1)

        self.layout.addWidget(self.x_label,                         12,  6, 1, 1)
        self.layout.addWidget(self.y_label,                         13,  6, 1, 1)
        self.layout.addWidget(self.angle_label,                     14, 6, 1, 1)
        self.layout.addWidget(self.x_scale_label,                   15, 6, 1, 1)
        self.layout.addWidget(self.y_scale_label,                   16, 6, 1, 1)
        self.layout.addWidget(self.map_button,                      17, 6, 1, 1)


        # Add ROI
        horizontal_line_coords = [[304, 500], [304, 25]]
        vertical_line_coords = [[202, 100], [402, 100]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)

        self.cross_display_view.addItem(self.horizontal_roi)
        self.cross_display_view.addItem(self.vertical_roi)


    def draw_images(self):

        # Draw Brain Image
        transformed_max_projection = self.transform_image(self.max_projection, self.variable_dictionary_list[self.current_matching_index])

        # Add Mask or Template
        if self.template_session_list_widget.currentItem().text() == "Mask":

            print("Transformed Max Projection Shape", np.shape(transformed_max_projection))
            transformed_mask = self.transform_mask_or_atlas(self.mask, self.mask_variable_dictionary)

            # Apply Mask
            transformed_max_projection = np.multiply(transformed_max_projection, transformed_mask)

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            transformed_atlas = self.transform_mask_or_atlas(self.atlas_outline, self.atlas_variable_dictionary)

            atlas_indicies = np.nonzero(transformed_atlas)
            transformed_max_projection[atlas_indicies] = 0

        self.cross_display_view.setImage(transformed_max_projection)



        
    def transform_image(self, image, variable_dictionary):
        
        # Rotate
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']

        transformed_image = np.copy(image)
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)
        transformed_image = np.roll(a=transformed_image, axis=0, shift=y_shift)
        transformed_image = np.roll(a=transformed_image, axis=1, shift=x_shift)

        return transformed_image


    def transform_mask_or_atlas(self, image, variable_dictionary):

        # Unpack Dictionary
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']
        x_scale = variable_dictionary['x_scale']
        y_scale = variable_dictionary['y_scale']

        # Copy
        transformed_image = np.copy(image)

        # Scale
        original_height, original_width = np.shape(transformed_image)
        new_height = int(original_height * y_scale)
        new_width = int(original_width * x_scale)
        transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)
        print("new image height", np.shape(transformed_image))

        # Rotate
        transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

        # Insert Into Background
        mask_height, mask_width = np.shape(transformed_image)
        centre_x = 200
        centre_y = 200
        background_array = np.zeros((1000, 1000))
        x_start = centre_x + x_shift
        x_stop = x_start + mask_width

        y_start = centre_y + y_shift
        y_stop = y_start + mask_height

        background_array[y_start:y_stop, x_start:x_stop] = transformed_image

        # Take Chunk
        transformed_image = background_array[centre_y:centre_y + self.image_height, centre_x:centre_x + self.image_width]

        # Rebinarize
        transformed_image = np.where(transformed_image > 0.5, 1, 0)



        return transformed_image


    def move_left(self):


        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['x_shift'] = self.mask_variable_dictionary['x_shift'] + 1
            self.x_label.setText("x: " + str(self.mask_variable_dictionary['x_shift']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['x_shift'] = self.atlas_variable_dictionary['x_shift'] + 1
            self.x_label.setText("x: " + str(self.atlas_variable_dictionary['x_shift']))

        self.draw_images()


    def move_right(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['x_shift'] = self.mask_variable_dictionary['x_shift'] - 1
            self.x_label.setText("x: " + str(self.mask_variable_dictionary['x_shift']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['x_shift'] = self.atlas_variable_dictionary['x_shift'] - 1
            self.x_label.setText("x: " + str(self.atlas_variable_dictionary['x_shift']))

        self.draw_images()


    def move_up(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['y_shift'] = self.mask_variable_dictionary['y_shift'] - 1
            self.y_label.setText("y: " + str(self.mask_variable_dictionary['y_shift']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['y_shift'] = self.atlas_variable_dictionary['y_shift'] - 1
            self.y_label.setText("y: " + str(self.atlas_variable_dictionary['y_shift']))

        self.draw_images()


    def move_down(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['y_shift'] = self.mask_variable_dictionary['y_shift'] + 1
            self.y_label.setText("y: " + str(self.mask_variable_dictionary['y_shift']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['y_shift'] = self.atlas_variable_dictionary['y_shift'] + 1
            self.y_label.setText("y: " + str(self.atlas_variable_dictionary['y_shift']))

        self.draw_images()


    def rotate_clockwise(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['rotation'] = self.mask_variable_dictionary['rotation'] - 1
            self.angle_label.setText("x: " + str(self.mask_variable_dictionary['rotation']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['rotation'] = self.atlas_variable_dictionary['rotation'] - 1
            self.angle_label.setText("x: " + str(self.atlas_variable_dictionary['rotation']))

        self.draw_images()


    def rotate_counterclockwise(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['rotation'] = self.mask_variable_dictionary['rotation'] + 1
            self.angle_label.setText("x: " + str(self.mask_variable_dictionary['rotation']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['rotation'] = self.atlas_variable_dictionary['rotation'] + 1
            self.angle_label.setText("x: " + str(self.atlas_variable_dictionary['rotation']))

        self.draw_images()


    def grow_x(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['x_scale'] = self.mask_variable_dictionary['x_scale'] + self.growth_increment
            self.x_scale_label.setText("x: " + str(self.mask_variable_dictionary['x_scale']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['x_scale'] = self.atlas_variable_dictionary['x_scale'] + self.growth_increment
            self.x_scale_label.setText("x: " + str(self.atlas_variable_dictionary['x_scale']))


        self.draw_images()


    def grow_y(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['y_scale'] = self.mask_variable_dictionary['y_scale'] + self.growth_increment
            self.y_scale_label.setText("y: " + str(self.mask_variable_dictionary['y_scale']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['y_scale'] = self.atlas_variable_dictionary['y_scale'] + self.growth_increment
            self.y_scale_label.setText("y: " + str(self.atlas_variable_dictionary['y_scale']))

        self.draw_images()


    def shrink_x(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['x_scale'] = self.mask_variable_dictionary['x_scale'] - self.growth_increment
            self.x_scale_label.setText("x: " + str(self.mask_variable_dictionary['x_scale']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['x_scale'] = self.atlas_variable_dictionary['x_scale'] - self.growth_increment
            self.x_scale_label.setText("x: " + str(self.atlas_variable_dictionary['x_scale']))

        self.draw_images()


    def shrink_y(self):

        if self.template_session_list_widget.currentItem().text() == "Mask":
            self.mask_variable_dictionary['y_scale'] = self.mask_variable_dictionary['y_scale'] - self.growth_increment
            self.y_scale_label.setText("y: " + str(self.mask_variable_dictionary['y_scale']))

        elif self.template_session_list_widget.currentItem().text() == "Atlas":
            self.atlas_variable_dictionary['y_scale'] = self.atlas_variable_dictionary['y_scale'] - self.growth_increment
            self.y_scale_label.setText("y: " + str(self.atlas_variable_dictionary['y_scale']))

        self.draw_images()



    def set_alignment(self):

        consensus_cluster_directory = r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering"

        # Save Dictionaries
        np.save(os.path.join(consensus_cluster_directory, "Consensus_Cluster_Atlas_Alignment_Dictionary.npy"), self.atlas_variable_dictionary)
        #np.save(os.path.join(consensus_cluster_directory, "Tight_Mask_Alignment_Dictionary.npy"), self.mask_variable_dictionary)


        pass


    def load_matching_session(self):
        
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        
        self.max_projection = self.max_projection_list[self.current_matching_index]
        self.draw_images()
        #self.cross_display_view.setImage(max_projection)


    def load_template_session(self):

        # Put Stuff In Here To Load Mask Or Load Allen Atlas
        self.current_template_index = self.template_session_list_widget.currentRow()
        self.draw_images()


def align_sessions(session_list, mask, atlas):

    app = QApplication(sys.argv)

    window = masking_window(session_list, mask, atlas)
    window.show()

    app.exec_()


mask_location = "/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy"
atlas_outline_location = "/home/matthew/Documents/Allen_Atlas_Templates/New_Outline.npy"

session_list = Registration_Utils.load_all_sessions()
mask = np.load(mask_location)
atlas_outline = np.load(atlas_outline_location)

align_sessions(session_list, mask, atlas_outline)
