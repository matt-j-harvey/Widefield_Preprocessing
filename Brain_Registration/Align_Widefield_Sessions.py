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





class session_matching_window(QWidget):

    def __init__(self, parent=None):
        super(session_matching_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Session Registration")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.template_directory = None
        self.template_name = None
        self.template_max_projection = None
        self.template_skeleton_image = None
        self.template_functional_image = None
        self.template_mask = None

        self.matching_directory = None
        self.matching_name = None
        self.matching_max_projection = None
        self.matching_skeleton_image = None
        self.matching_functional_image = None
        self.matching_mask = None

        self.variable_dictionary = self.create_variable_dictionary()


        # Add Session Buttons
        self.select_template_session_button = QPushButton("Select Template Session")
        self.select_matching_session_button = QPushButton("Select Matching Session")

        self.select_template_session_button.clicked.connect(self.select_template_session)
        self.select_matching_session_button.clicked.connect(self.select_matching_session)

        self.template_label = QLabel("Template Session: ")
        self.matching_label = QLabel("Matching Session: ")

        # Create Skeleton Figures
        self.skeleton_display_view_widget = QWidget()
        self.skeleton_display_view_widget_layout = QGridLayout()
        self.skeleton_display_view = pyqtgraph.ImageView()
        self.skeleton_display_view.ui.histogram.hide()
        self.skeleton_display_view.ui.roiBtn.hide()
        self.skeleton_display_view.ui.menuBtn.hide()
        self.skeleton_display_view_widget_layout.addWidget(self.skeleton_display_view, 0, 0)
        self.skeleton_display_view_widget.setLayout(self.skeleton_display_view_widget_layout)

        # Create Anatomy Figures
        self.anatomy_display_view_widget = QWidget()
        self.anatomy_display_view_widget_layout = QGridLayout()
        self.anatomy_display_view = pyqtgraph.ImageView()
        self.anatomy_display_view.ui.histogram.hide()
        self.anatomy_display_view.ui.roiBtn.hide()
        self.anatomy_display_view.ui.menuBtn.hide()
        self.anatomy_display_view_widget_layout.addWidget(self.anatomy_display_view, 0, 0)
        self.anatomy_display_view_widget.setLayout(self.anatomy_display_view_widget_layout)

        # Create Functional Figures
        self.functional_display_view_widget = QWidget()
        self.functional_display_view_widget_layout = QGridLayout()
        self.functional_display_view = pyqtgraph.ImageView()
        self.functional_display_view.ui.histogram.hide()
        self.functional_display_view.ui.roiBtn.hide()
        self.functional_display_view.ui.menuBtn.hide()
        self.functional_display_view_widget_layout.addWidget(self.functional_display_view, 0, 0)
        self.functional_display_view_widget.setLayout(self.functional_display_view_widget_layout)

        # Create Masked Figures
        self.masked_display_view_widget = QWidget()
        self.masked_display_view_widget_layout = QGridLayout()
        self.masked_display_view = pyqtgraph.ImageView()
        self.masked_display_view.ui.histogram.hide()
        self.masked_display_view.ui.roiBtn.hide()
        self.masked_display_view.ui.menuBtn.hide()
        self.masked_display_view_widget_layout.addWidget(self.masked_display_view, 0, 0)
        self.masked_display_view_widget.setLayout(self.masked_display_view_widget_layout)


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
        self.x_label = QLabel()
        self.y_label = QLabel()
        self.height_label = QLabel()
        self.width_label = QLabel()
        self.angle_label = QLabel()

        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Transformation Widgets
        self.layout.addWidget(self.template_label,                  0,  0,  1,  1)
        self.layout.addWidget(self.matching_label,                  1,  0,  1,  1)
        self.layout.addWidget(self.select_template_session_button,  2,  0,  1,  1)
        self.layout.addWidget(self.select_matching_session_button,  3,  0,  1,  1)
        self.layout.addWidget(self.left_button,                     4,  0,  1,  1)
        self.layout.addWidget(self.right_button,                    5,  0,  1,  1)
        self.layout.addWidget(self.up_button,                       6,  0,  1,  1)
        self.layout.addWidget(self.down_button,                     7,  0,  1,  1)
        self.layout.addWidget(self.rotate_clockwise_button,         8,  0,  1,  1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  9,  0,  1,  1)
        self.layout.addWidget(self.x_label,                         12, 0,  1,  1)
        self.layout.addWidget(self.y_label,                         13, 0,  1,  1)
        self.layout.addWidget(self.height_label,                    14, 0,  1,  1)
        self.layout.addWidget(self.width_label,                     15, 0,  1,  1)
        self.layout.addWidget(self.angle_label,                     16, 0,  1,  1)
        self.layout.addWidget(self.map_button,                      17, 0,  1,  1)

        # Add Display Widgets
        self.layout.addWidget(self.skeleton_display_view_widget,    0, 1, 9, 1)
        self.layout.addWidget(self.anatomy_display_view_widget,     0, 2, 9, 1)
        self.layout.addWidget(self.functional_display_view_widget,  9, 1, 9, 1)
        self.layout.addWidget(self.masked_display_view_widget,      9, 2, 9, 1)


        self.show()


    def create_variable_dictionary(self):

        # Transformation Attributes
        x_shift = 0
        y_shift = 0
        rotation = 0

        # Array Details
        background_size = 800
        bounding_size = 400
        background_array = np.zeros((background_size, background_size, 3))
        bounding_array = np.zeros((bounding_size, bounding_size))

        # Template Details
        template_x_start = 100
        template_y_start = 100
        template_width = 608
        template_height = 600

        # Create Dictionary
        variable_dictionary = {

            # Affine Atributes
            'x_shift': x_shift,
            'y_shift': y_shift,
            'rotation': rotation,

            # Template Deets
            'template_x_start': template_x_start,
            'template_y_start': template_y_start,
            'template_width': template_width,
            'template_height': template_height,

            # Arrays
            'background_array': background_array,
            'bounding_array': bounding_array
        }

        return variable_dictionary


    def select_template_session(self):

        # Get Template Directory
        print("Getting Template Session")
        new_session_directory = QFileDialog.getExistingDirectory(self, "Select Template Session Directory")
        self.template_directory = new_session_directory

        # Load Max Projection
        max_projection = np.load(new_session_directory + "/max_projection.npy")
        self.template_max_projection = max_projection

        # Load Skeleton
        skeleton_array = np.load(new_session_directory + "/Skeleton.npy")
        self.template_skeleton_image = skeleton_array

        # Load Retinotopy Contours
        retinotopy_array = np.load(new_session_directory + "/Registered_Contour_Array.npy")
        self.template_functional_image = retinotopy_array

        # Get New Directory + Session Name
        new_session_directory_split = new_session_directory.split("/")
        new_session_name = new_session_directory_split[-2] + "_" + new_session_directory_split[-1]

        # Add These To The Lists
        self.template_name = new_session_name
        self.template_label.setText("Template Session: " + new_session_name)


    def select_matching_session(self):

        # Get Template Directory
        new_session_directory = QFileDialog.getExistingDirectory(self, "Select Matching Session Directory")
        self.matching_directory = new_session_directory

        # Load Max Projection
        max_projection = np.load(new_session_directory + "/max_projection.npy")
        self.matching_max_projection = max_projection

        # Load Skeleton
        skeleton_array = np.load(new_session_directory + "/Skeleton.npy")
        self.matching_skeleton_image = skeleton_array

        # Load Retinotopy Contours
        if os.path.isfile(new_session_directory + "/Registered_Contour_Array.npy"):
            retinotopy_array = np.load(new_session_directory + "/Registered_Contour_Array.npy")
        else:
            retinotopy_array = np.zeros(np.shape(self.matching_skeleton_image))

        self.matching_functional_image = retinotopy_array

        # Get New Directory + Session Name
        new_session_directory_split = new_session_directory.split("/")
        new_session_name = new_session_directory_split[-2] + "_" + new_session_directory_split[-1]

        # Add These To The Lists
        self.matching_name = new_session_name
        self.matching_label.setText("Matching Session: " + new_session_name)


    def move_left(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 2
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 2
        self.x_label.setText("x: " + str(self.variable_dictionary['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] - 2
        self.y_label.setText("y: " + str(self.variable_dictionary['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] + 2
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


    def transform_array(self, template_image, matching_image, variable_dictionary):

        # Load Data
        template_x_start = variable_dictionary['template_x_start']
        template_y_start = variable_dictionary['template_y_start']
        template_width   = variable_dictionary['template_width']
        template_height  = variable_dictionary['template_height']
        x_shift          = variable_dictionary['x_shift']
        y_shift          = variable_dictionary['y_shift']
        background_array = np.copy(variable_dictionary["background_array"])

        # Rotate
        angle = variable_dictionary['rotation']
        matching_image = ndimage.rotate(matching_image, angle, reshape=False)

        # Translate
        matching_image = np.roll(a=matching_image, axis=0, shift=y_shift)
        matching_image = np.roll(a=matching_image, axis=1, shift=x_shift)

        # Scale Images
        template_image = np.divide(template_image.astype(np.float), np.percentile(template_image, 99))
        matching_image = np.divide(matching_image.astype(np.float), np.percentile(matching_image, 99))

        template_image = np.nan_to_num(template_image)
        matching_image = np.nan_to_num(matching_image)

        template_image = np.clip(template_image, a_min=0, a_max=1)
        matching_image = np.clip(matching_image, a_min=0, a_max=1)

        # Insert Images Into Background Array
        image_height = np.shape(matching_image)[0]
        image_width = np.shape(matching_image)[1]

        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 2] = template_image
        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 1] += 0.5 * template_image

        background_array[template_y_start:template_y_start + image_height, template_x_start:template_x_start + image_width, 0] = matching_image
        background_array[template_y_start:template_y_start + image_height, template_x_start:template_x_start + image_width, 1] += 0.5 * matching_image

        return background_array



    def draw_images(self):

        # Draw Anatomical Images
        anatomy_array = self.transform_array(self.template_max_projection, self.matching_max_projection, self.variable_dictionary)
        self.anatomy_display_view.setImage(anatomy_array)

        # Draw Skeleton Images
        skeleton_array = self.transform_array(self.template_skeleton_image, self.matching_skeleton_image, self.variable_dictionary)
        self.skeleton_display_view.setImage(skeleton_array)

        # Draw Anatomical Images
        functional_array = self.transform_array(self.template_functional_image, self.matching_functional_image, self.variable_dictionary)
        self.functional_display_view.setImage(functional_array)


        # Transform Masked Matching Image
        transformed_matching_max_projection = np.copy(self.matching_max_projection)

        # Rotate
        angle = self.variable_dictionary['rotation']
        x_shift = self.variable_dictionary['x_shift']
        y_shift = self.variable_dictionary['y_shift']

        transformed_matching_max_projection = ndimage.rotate(transformed_matching_max_projection, angle, reshape=False)
        transformed_matching_max_projection = np.roll(a=transformed_matching_max_projection, axis=0, shift=y_shift)
        transformed_matching_max_projection = np.roll(a=transformed_matching_max_projection, axis=1, shift=x_shift)

        # Masked Image
        mask = np.load(self.template_directory + "/mask.npy")
        transformed_matching_max_projection = np.multiply(mask, transformed_matching_max_projection)

        self.masked_display_view.setImage(transformed_matching_max_projection)


    def set_alignment(self):

        # Get Save Directory
        save_directory = self.matching_directory + "/Transformation_Dictionary.npy"
        print("Save Directory", save_directory)

        # Create Transformation Array
        transformation_dictionary = {}
        transformation_dictionary["template"] = self.template_directory
        transformation_dictionary["rotation"] = self.variable_dictionary["rotation"]
        transformation_dictionary["y_shift"] = self.variable_dictionary["y_shift"]
        transformation_dictionary["x_shift"] = self.variable_dictionary["x_shift"]

        np.save(save_directory, transformation_dictionary)





if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = session_matching_window()
    window.show()

    app.exec_()


