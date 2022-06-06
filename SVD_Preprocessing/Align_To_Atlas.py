import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import signal, ndimage, stats
from skimage.transform import resize
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
        self.setWindowTitle("Align To Allen Atlas")
        self.setGeometry(0, 0, 1000, 500)

        # Create Variable Holders
        self.template_directory = None
        self.template_name = None
        self.template_max_projection = None
        self.template_functional_image = None
        self.template_mask = None
        self.allen_atlas_outline = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy")
        self.allen_atlas_regions = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Allen_Atlas_Mapping.npy")
        self.variable_dictionary = self.create_variable_dictionary()

        # Add Session Buttons
        self.select_template_session_button = QPushButton("Select Template Session")
        self.select_template_session_button.clicked.connect(self.select_template_session)
        self.template_label = QLabel("Template Session: ")

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

        self.enlarge_button = QPushButton("Enlarge")
        self.enlarge_button.clicked.connect(self.enlarge)

        self.shrink_button = QPushButton("Shrink")
        self.shrink_button.clicked.connect(self.shrink)

        self.enlarge_x_button = QPushButton("Enlarge X")
        self.enlarge_x_button.clicked.connect(self.enlarge_x)

        self.shrink_x_button = QPushButton("Shrink X")
        self.shrink_x_button.clicked.connect(self.shrink_x)

        self.enlarge_y_button = QPushButton("Enlarge Y")
        self.enlarge_y_button.clicked.connect(self.enlarge_y)

        self.shrink_y_button = QPushButton("Shrink Y")
        self.shrink_y_button.clicked.connect(self.shrink_y)

        self.map_button = QPushButton("Set Alignment")
        self.map_button.clicked.connect(self.set_alignment)


        # Create Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Transformation Widgets
        self.layout.addWidget(self.template_label,                  0,  0,  1,  1)
        self.layout.addWidget(self.select_template_session_button,  1,  0,  1,  1)
        self.layout.addWidget(self.left_button,                     2,  0,  1,  1)
        self.layout.addWidget(self.right_button,                    3,  0,  1,  1)
        self.layout.addWidget(self.up_button,                       4,  0,  1,  1)
        self.layout.addWidget(self.down_button,                     5,  0,  1,  1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  0,  1,  1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  0,  1,  1)
        self.layout.addWidget(self.enlarge_button,                  8,  0,  1,  1)
        self.layout.addWidget(self.shrink_button,                   9,  0,  1,  1)
        self.layout.addWidget(self.enlarge_x_button,                10, 0,  1,  1)
        self.layout.addWidget(self.shrink_x_button,                 11, 0, 1, 1)
        self.layout.addWidget(self.enlarge_y_button,                12, 0, 1, 1)
        self.layout.addWidget(self.shrink_y_button,                 13, 0, 1, 1)
        self.layout.addWidget(self.map_button,                      14, 0,  1,  1)

        # Add Display Widgets
        self.layout.addWidget(self.anatomy_display_view_widget,     0, 1, 16, 16)
        self.layout.addWidget(self.functional_display_view_widget,  0, 17, 16, 16)

        self.show()


    def create_variable_dictionary(self):

        # Transformation Attributes
        x_shift = 0
        y_shift = 0
        rotation = 0
        x_scale_factor = 1
        y_scale_factor = 1

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
            'x_scale_factor': x_scale_factor,
            'y_scale_factor': y_scale_factor,

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

        # Load Retinotopy Contours
        retinotopy_array = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/Atlas_Template_V2.npy")
        self.template_functional_image = retinotopy_array

        # Load Mask
        mask = np.load(new_session_directory + "/Generous_Mask_Array.npy")
        self.template_mask = mask

        # Get New Directory + Session Name
        new_session_directory_split = new_session_directory.split("/")
        new_session_name = new_session_directory_split[-2] + "_" + new_session_directory_split[-1]

        # Add These To The Lists
        self.template_name = new_session_name
        self.template_label.setText("Template Session: " + new_session_name)

    def move_left(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] + 2
        self.draw_images()

    def move_right(self):
        self.variable_dictionary['x_shift'] = self.variable_dictionary['x_shift'] - 2
        self.draw_images()

    def move_up(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] - 2
        self.draw_images()

    def move_down(self):
        self.variable_dictionary['y_shift'] = self.variable_dictionary['y_shift'] + 2
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] - 1
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary['rotation'] = self.variable_dictionary['rotation'] + 1
        self.draw_images()

    def enlarge(self):
        self.variable_dictionary['x_scale_factor'] = self.variable_dictionary['x_scale_factor'] + 0.01
        self.variable_dictionary['y_scale_factor'] = self.variable_dictionary['y_scale_factor'] + 0.01
        self.draw_images()

    def shrink(self):
        self.variable_dictionary['x_scale_factor'] = self.variable_dictionary['x_scale_factor'] - 0.01
        self.variable_dictionary['y_scale_factor'] = self.variable_dictionary['y_scale_factor'] - 0.01
        self.draw_images()

    def enlarge_x(self):
        self.variable_dictionary['x_scale_factor'] = self.variable_dictionary['x_scale_factor'] + 0.01
        self.draw_images()

    def shrink_x(self):
        self.variable_dictionary['x_scale_factor'] = self.variable_dictionary['x_scale_factor'] - 0.01
        self.draw_images()

    def enlarge_y(self):
        self.variable_dictionary['y_scale_factor'] = self.variable_dictionary['y_scale_factor'] + 0.01
        self.draw_images()

    def shrink_y(self):
        self.variable_dictionary['y_scale_factor'] = self.variable_dictionary['y_scale_factor'] - 0.01
        self.draw_images()

    def transform_array(self, template_image, matching_image, variable_dictionary):

        # Load Data
        template_x_start = variable_dictionary['template_x_start']
        template_y_start = variable_dictionary['template_y_start']
        template_width   = variable_dictionary['template_width']
        template_height  = variable_dictionary['template_height']
        x_shift          = variable_dictionary['x_shift']
        y_shift          = variable_dictionary['y_shift']
        x_scale_factor   = variable_dictionary['x_scale_factor']
        y_scale_factor   = variable_dictionary['y_scale_factor']
        background_array = np.copy(variable_dictionary["background_array"])

        # Rescale Array
        matching_image = resize(matching_image, (int(y_scale_factor * template_height), int(x_scale_factor * template_width)))

        # Rotate
        angle = variable_dictionary['rotation']
        matching_image = ndimage.rotate(matching_image, angle, reshape=False)

        # Translate
        #matching_image = np.roll(a=matching_image, axis=0, shift=y_shift)
        #matching_image = np.roll(a=matching_image, axis=1, shift=x_shift)

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

        print("new size", image_height, image_width)

        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 2] = template_image
        background_array[template_y_start:template_y_start + template_height, template_x_start:template_x_start + template_width, 1] += 0.5 * template_image

        background_array[template_y_start + y_shift:template_y_start + image_height + y_shift, template_x_start + x_shift:template_x_start + image_width + x_shift, 0] = matching_image
        background_array[template_y_start + y_shift:template_y_start + image_height + y_shift, template_x_start + x_shift:template_x_start + image_width + x_shift, 1] += 0.5 * matching_image

        return background_array



    def draw_images(self):

        # Draw Anatomical Images
        anatomy_array = self.transform_array(self.template_max_projection, self.allen_atlas_outline, self.variable_dictionary)
        self.anatomy_display_view.setImage(anatomy_array)

        # Draw Anatomical Images
        functional_array = self.transform_array(self.template_functional_image, self.allen_atlas_outline, self.variable_dictionary)
        self.functional_display_view.setImage(functional_array)


    def set_alignment(self):

        # Get Save Directory
        save_directory = self.template_directory + "/Atlas_Alignment_Dictionary.npy"
        print("Save Directory", save_directory)

        # Create Transformation Array
        transformation_dictionary = {}
        transformation_dictionary["template"] = self.template_directory
        transformation_dictionary["rotation"] = self.variable_dictionary["rotation"]
        transformation_dictionary["y_shift"] = self.variable_dictionary["y_shift"]
        transformation_dictionary["x_shift"] = self.variable_dictionary["x_shift"]
        transformation_dictionary["x_scale_factor"] = self.variable_dictionary["x_scale_factor"]
        transformation_dictionary["y_scale_factor"] = self.variable_dictionary["y_scale_factor"]

        np.save(save_directory, transformation_dictionary)





if __name__ == '__main__':

    app = QApplication(sys.argv)

    window = session_matching_window()
    window.show()

    app.exec_()