import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy import ndimage
from skimage.feature import canny
import math

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys

import Registration_Utils

pyqtgraph.setConfigOptions(imageAxisOrder='row-major')


def load_regressor_maps(base_directory):

    # Load Regression Dict
    regression_dictionary = np.load(os.path.join(base_directory, "Regression_Coefs", "Regression_Dicionary.npy"), allow_pickle=True)[()]
    regression_coefs = regression_dictionary["Coefs"]
    regression_coefs = np.transpose(regression_coefs)
    regression_names = regression_dictionary["Regressor_Names"]

    # Load Mask
    indicies, image_height, image_width = Registration_Utils.load_downsampled_mask(base_directory)

    # View Coefs
    regressor_maps = []
    for coef in regression_coefs:
        coef_map = Registration_Utils.create_image_from_data(coef, indicies, image_height, image_width)
        coef_map = np.abs(coef_map)

        coef_map = np.divide(coef_map, np.percentile(coef_map, q=99))
        coef_map = np.clip(coef_map, a_min=0, a_max=1)

        coef_map = np.where(coef_map > 0.7, coef_map, 0)

        #plt.imshow(coef_map)
        #plt.show()

        regressor_maps.append(coef_map)

    return regressor_maps, regression_names



def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    oy, ox = origin
    py, px = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qy, qx


class masking_window(QWidget):

    def __init__(self, session_list, parent=None):
        super(masking_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("Align To Static Cross")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_template_index = 0
        self.current_matching_index = 0

        # Get List Of Max Projections
        self.max_projection_list = []
        self.regressor_map_list = []
        self.regressor_name_list = []

        for session in session_list:

            # Load Max Projection
            max_projection = np.load(os.path.join(session, "Blue_Example_Image.npy"))
            upper_bound = np.percentile(max_projection, 99)
            max_projection = np.divide(max_projection, upper_bound)
            max_projection = np.clip(max_projection, a_min=0, a_max=1)
            self.max_projection_list.append(max_projection)

            # Load Regressor Maps
            regressor_maps, regressor_names = load_regressor_maps(session)
            self.regressor_map_list.append(regressor_maps)
            self.regressor_name_list.append(regressor_names)


        # Get List Of Alignment Dictionaries
        self.variable_dictionary_list = []
        for session_index in range(self.number_of_sessions):

            variable_dictionary_directory = os.path.join(self.session_list[session_index],"Brain_Alignment_Dictionary.npy")
            if os.path.exists(variable_dictionary_directory):
                print("IT Exists")
                variable_dictionary = np.load(variable_dictionary_directory, allow_pickle=True)[()]
            else:
                variable_dictionary = {'x_shift': 0, 'y_shift': 0, 'rotation': 0}

            self.variable_dictionary_list.append(variable_dictionary)

        # Set Current Images
        self.selected_regressor_index = 0
        self.max_projection = self.max_projection_list[0]
        self.template_regressor = self.regressor_map_list[0][self.selected_regressor_index]
        self.matching_regressor = self.regressor_map_list[0][self.selected_regressor_index]

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

        # Add Radial Button Group
        self.radial_button_group = QButtonGroup()  # Letter group
        self.template_button = QRadioButton("template")
        self.matching_button = QRadioButton("matching")
        self.both_button = QRadioButton("both")

        self.radial_button_group.addButton(self.template_button)
        self.radial_button_group.addButton(self.matching_button)
        self.radial_button_group.addButton(self.both_button)

        self.template_button.toggled.connect(self.draw_images)
        self.matching_button.toggled.connect(self.draw_images)
        self.both_button.toggled.connect(self.draw_images)

        # Add Regressor Selection Boxes
        self.regressor_dropdown_box = QComboBox()
        for regressor_name in self.regressor_name_list[0]:
            self.regressor_dropdown_box.addItem(regressor_name)
        self.regressor_dropdown_box.currentIndexChanged.connect(self.change_selected_regressor)

        # Create Session Labels
        self.template_session_label = QLabel("Template Session: ")
        self.matching_session_label = QLabel("Matching Session: ")

        # Create Session List Views
        self.template_session_list_widget = QListWidget()
        self.matching_session_list_widget = QListWidget()

        self.matching_session_list_widget.currentRowChanged.connect(self.load_matching_session)
        self.template_session_list_widget.currentRowChanged.connect(self.load_template_session)

        for session in self.session_list:
            split_session = session.split('/')
            session_name = split_session[-2] + "_" + split_session[-1]
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
        self.layout.addWidget(self.matching_session_label,          0, 1, 1, 2)

        # Add List Views
        self.layout.addWidget(self.template_session_list_widget,    1, 0, 25, 1)
        self.layout.addWidget(self.matching_session_list_widget,    1, 1, 25, 1)

        # Add Display Views
        self.layout.addWidget(self.regressor_display_view_widget,       1, 2, 25, 1)
        self.layout.addWidget(self.cross_display_view_widget,       1, 3, 25, 1)

        # Add Radial Buttons
        self.layout.addWidget(self.template_button,                 27, 2, 1, 1)
        self.layout.addWidget(self.matching_button,                 28, 2, 1, 1)
        self.layout.addWidget(self.both_button,                 29, 2, 1, 1)

        # Add Regressor Combo Box
        self.layout.addWidget(self.regressor_dropdown_box, 27, 3, 1, 1)


        # Add Transformation Controls
        control_column = 4
        self.layout.addWidget(self.left_button,                     2,  control_column, 1, 1)
        self.layout.addWidget(self.right_button,                    3,  control_column, 1, 1)
        self.layout.addWidget(self.up_button,                       4,  control_column, 1, 1)
        self.layout.addWidget(self.down_button,                     5,  control_column, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button,         6,  control_column, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button,  7,  control_column, 1, 1)
        self.layout.addWidget(self.x_label,                         8,  control_column, 1, 1)
        self.layout.addWidget(self.y_label,                         9,  control_column, 1, 1)
        self.layout.addWidget(self.angle_label,                     10, control_column, 1, 1)
        self.layout.addWidget(self.map_button,                      11, control_column, 1, 1)


        # Add ROI
        horizontal_line_coords = [[152, 250], [152, 12]]
        vertical_line_coords = [[101, 50], [201, 50]]

        self.horizontal_roi = pyqtgraph.PolyLineROI(horizontal_line_coords, closed=False)
        self.vertical_roi = pyqtgraph.PolyLineROI(vertical_line_coords, closed=False)

        self.cross_display_view.addItem(self.horizontal_roi)
        self.cross_display_view.addItem(self.vertical_roi)

    def change_selected_regressor(self):
        self.selected_regressor_index = self.regressor_dropdown_box.currentIndex()
        self.matching_regressor = self.regressor_map_list[self.current_matching_index][self.selected_regressor_index]
        self.template_regressor = self.regressor_map_list[self.current_template_index][self.selected_regressor_index]
        self.draw_images()

    def draw_images(self):
        
        transformed_max_projection = self.transform_image(self.max_projection, self.variable_dictionary_list[self.current_matching_index])
        transformed_matching_regressor = self.transform_image(self.matching_regressor, self.variable_dictionary_list[self.current_matching_index])
        transformed_template_regressor = self.transform_image(self.template_regressor, self.variable_dictionary_list[self.current_template_index])

        combined_edges_template = np.zeros((300, 304, 3))

        if self.template_button.isChecked():
            combined_edges_template[:, :, 1] = transformed_template_regressor

        elif self.matching_button.isChecked():
            combined_edges_template[:, :, 1] = transformed_matching_regressor

        elif self.both_button.isChecked():
            combined_edges_template[:, :, 0] = transformed_matching_regressor
            combined_edges_template[:, :, 1] = transformed_template_regressor


        self.cross_display_view.setImage(transformed_max_projection)
        self.regressor_display_view.setImage(combined_edges_template)

        
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

    def transform_point(self, point, variable_dictionary):

        blank_array = np.zeros(np.shape(self.max_projection_list[self.current_matching_index]))

        point_y_start = point[0] - 10
        point_y_stop = point[0] + 10
        point_x_start = point[1] - 10
        point_x_stop = point[1] + 10
        blank_array[point_y_start:point_y_stop, point_x_start:point_x_stop] = 1
        #plt.imshow(blank_array)
        #plt.show()

        blank_array = self.transform_image(blank_array, variable_dictionary)
        blank_array = np.where(blank_array > 0.1, 1, 0)

        blank_indicies = np.nonzero(blank_array)
        transformed_point = np.mean(blank_indicies, axis=1)

        """
        # Rotate
        angle = variable_dictionary['rotation']
        x_shift = variable_dictionary['x_shift']
        y_shift = variable_dictionary['y_shift']

        height, width = np.shape(self.max_projection_list[self.current_matching_index])
        centre = [int(height/2), int(width/2)]

        # Rotate
        point = list(rotate(centre, point, angle))
        print("Post Rotated", point)
        # Translate
        point[0] += y_shift
        point[1] += x_shift
        """

        return transformed_point


    def move_left(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] - 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_right(self):
        self.variable_dictionary_list[self.current_matching_index]['x_shift'] = self.variable_dictionary_list[self.current_matching_index]['x_shift'] + 1
        self.x_label.setText("x: " + str(self.variable_dictionary_list[self.current_matching_index]['x_shift']))
        self.draw_images()

    def move_up(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] - 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def move_down(self):
        self.variable_dictionary_list[self.current_matching_index]['y_shift'] = self.variable_dictionary_list[self.current_matching_index]['y_shift'] + 1
        self.y_label.setText("y: " + str(self.variable_dictionary_list[self.current_matching_index]['y_shift']))
        self.draw_images()

    def rotate_clockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] - 0.5
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()

    def rotate_counterclockwise(self):
        self.variable_dictionary_list[self.current_matching_index]['rotation'] = self.variable_dictionary_list[self.current_matching_index]['rotation'] + 0.5
        self.angle_label.setText("Angle: " + str(self.variable_dictionary_list[self.current_matching_index]['rotation']))
        self.draw_images()


    def set_alignment(self):

        for session_index in  range(self.number_of_sessions):

            # Get Save Directory
            session_directory = self.session_list[session_index]
            save_directory = os.path.join(session_directory, "Cluster_Alignment_Dictionary.npy")

            # Save Dictionary
            np.save(save_directory, self.variable_dictionary_list[session_index])


    def load_matching_session(self):
        self.current_matching_index = self.matching_session_list_widget.currentRow()
        self.max_projection = self.max_projection_list[self.current_matching_index]
        self.matching_regressor = self.regressor_map_list[self.current_matching_index][self.selected_regressor_index]
        self.draw_images()

    def load_template_session(self):
        self.current_template_index = self.template_session_list_widget.currentRow()
        self.template_regressor = self.regressor_map_list[self.current_template_index][self.selected_regressor_index]
        self.draw_images()

    def lambda_roi_changed(self):
        self.lambda_pos = [int(self.lambda_roi.pos().y()), int(self.lambda_roi.pos().x())]
        self.lamba_pos_label.setText("Lambda Pos: " + str(self.lambda_pos))
        self.draw_images()

    def coronal_roi_changed(self):
        self.coronal_pos = [int(self.coronal_suture_roi.pos().y()), int(self.coronal_suture_roi.pos().x())]
        self.coronal_pos_label.setText("Coronal Pos: " + str(self.coronal_pos))
        self.draw_images()


def align_sessions(session_list):

    app = QApplication(sys.argv)
    window = masking_window(session_list)
    window.show()
    app.exec_()




session_list = [
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_23_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_03_31_Transition_Imaging",
    r"/media/matthew/29D46574463D2856/Processed_New_Pipeline/NXAK7.1B/2021_04_02_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_02_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_08_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK4.1B/2021_04_10_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_13_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_15_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK14.1A/2021_06_17_Transition_Imaging",

    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_10_29_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_03_Transition_Imaging",
    r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Transition_Reprocessed/NXAK22.1A/2021_11_05_Transition_Imaging"
]

print(session_list)
align_sessions(session_list)