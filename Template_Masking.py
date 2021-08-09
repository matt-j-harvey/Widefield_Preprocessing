from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import imageio
import numpy as np
from scipy import ndimage
import skimage.transform
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib import cm
import PIL.Image
from skimage.transform import resize
from scipy import stats
import h5py
import os


def check_led_colours(blue_array, violet_array):
    figure_1 = plt.figure()
    axes_1 = figure_1.subplots(1, 2)

    blue_image = blue_array[:, 0]
    blue_image = np.reshape(blue_image, (600,608))
    axes_1[0].set_title("Blue?")
    axes_1[0].imshow(blue_image)

    violet_image = violet_array[:, 0]
    violet_image = np.reshape(violet_image, (600,608))
    axes_1[1].set_title("Violet?")
    axes_1[1].imshow(violet_image)
    plt.show()


def get_max_projection(array, home_directory):
    print("Getting Max Projection")

    sample = array[:, 0:1000]
    max_projection = np.max(sample, axis=1)
    max_projection = np.reshape(max_projection, (600, 608))

    plt.imshow(max_projection)
    plt.show()

    np.save(home_directory + "/max_projection", max_projection)

def create_background_array(image_data, background_height, background_width, image_y_start, image_x_start, image_height,
                            image_width):
    background_array = np.zeros((background_height, background_width))
    background_array[image_y_start:image_y_start + image_height, image_x_start:image_x_start + image_width] = image_data
    background_array = np.divide(background_array, np.max(background_array))
    return background_array


def place_array_in_bounding_rect(atlas_boundaries_data, bounding_height, bounding_width):
    # Load Atlas Array
    atlas_height = np.shape(atlas_boundaries_data)[0]
    atlas_width = np.shape(atlas_boundaries_data)[1]

    # Create Bounding Rectangle
    atlas_bounding_rect = np.zeros((bounding_height, bounding_width))

    # Insert Array Into Binding Rectangle
    x_start = int((bounding_width - atlas_width) / 2)
    y_start = int((bounding_height - atlas_height) / 2)
    atlas_bounding_rect[y_start:y_start + atlas_height, x_start:x_start + atlas_width] = atlas_boundaries_data

    return atlas_bounding_rect


def transform_atlas(alignment_data):
    # Extract Used Variables
    print("loading variables for transformation")
    original_atlas_array = alignment_data["original_atlas_array"]
    rotation = alignment_data["rotation"]
    template_x = alignment_data["template_x"]
    template_y = alignment_data["template_y"]
    bounding_height = alignment_data["bounding_height"]
    bounding_width = alignment_data["bounding_width"]
    background_width = alignment_data["background_width"]
    background_height = alignment_data["background_height"]

    print("coping array")
    atlas_array = np.copy(original_atlas_array)
    print("atlas array size post resizing", np.shape(atlas_array))

    # Rotate Array
    print("rotating")
    atlas_array = ndimage.rotate(input=atlas_array, angle=rotation, reshape=False)
    print("atlas array size post resizing", np.shape(atlas_array))

    # Rescale Array
    print("resizing")
    atlas_array = resize(atlas_array, (bounding_height, bounding_width))

    print("clipping")
    # Ensure Template Coords Are Valid
    template_y = np.clip(template_y, a_min=0, a_max=background_height - bounding_height)
    template_x = np.clip(template_x, a_min=0, a_max=background_width - bounding_width)

    print("reinserting")
    # Insert These Variables Back Into Alignment Data
    alignment_data["atlas_array"] = atlas_array
    alignment_data["template_x"] = template_x
    alignment_data["template_y"] = template_y

    return alignment_data


def overlay_images(alignment_data):
    # Extract The Variables We Want From The Dictionary
    print("extracting")
    background_array = alignment_data["background_array"]
    atlas_array = alignment_data["atlas_array"]
    template_x = alignment_data["template_x"]
    template_y = alignment_data["template_y"]
    bounding_height = alignment_data["bounding_height"]
    bounding_width = alignment_data["bounding_width"]
    anatomical_image_data = alignment_data["anatomical_image_data"]
    image_x_start = alignment_data["image_x_start"]
    image_y_start = alignment_data["image_y_start"]

    print("overlaying")

    # Create New Combined Array
    combined_array = np.copy(background_array)
    print("combined array")

    # Get Boundary Pixels from Atlas
    flatlas_array = np.copy(atlas_array)
    print(np.shape(flatlas_array))

    flatlas_array = np.ndarray.flatten(flatlas_array)
    indicies = np.argwhere(flatlas_array)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    # Get Subset of Existing Background
    print("getting subset")
    subset_background = combined_array[template_y:template_y + bounding_height, template_x:template_x + bounding_width]
    print(np.shape(subset_background))
    #

    # Put The Boundaries Into The Subset
    print("putting in subset")
    data = np.ones(shape=np.shape(indicies))
    subset_background = np.ndarray.flatten(subset_background)
    print(np.shape(subset_background))
    print(np.shape(indicies))
    print(np.shape(data))
    np.put(subset_background, indicies, data, mode="clip")

    print("3")
    subset_background = np.ndarray.reshape(subset_background, (bounding_height, bounding_width))

    # Put The Subset Back
    print("returning subset")
    combined_array[template_y:template_y + bounding_height,
    template_x:template_x + bounding_width] = subset_background * 30000
    print("returned")

    # Add Anatomical Image Back
    print("anatomical image", np.shape(anatomical_image_data))

    anatomical_height = np.shape(anatomical_image_data)[0]
    anatomical_width = np.shape(anatomical_image_data)[1]

    print("anatomical height", anatomical_height)
    print("anatomical width", anatomical_width)

    anatomical_image_max = np.max(anatomical_image_data)
    print("anatomical iamge max", anatomical_image_max)

    combined_array[image_y_start:image_y_start + anatomical_height, image_x_start:image_x_start + anatomical_width] += anatomical_image_data

    return combined_array


def draw_image(figure, canvas, alignment_data):
    app = alignment_data["app"]

    # Perform Geometric Transformations
    print("Transforming")
    alignment_data = transform_atlas(alignment_data)

    # Rebinarise Image Following Transformations (interpolations may have given some pixels an intermediate value)
    print("rebinarising")
    alignment_data = rebinarise_array(alignment_data)

    # Overlay the Transformed Mask Onto The Anatomical Image
    print("overlaying")
    anatomy_combined_array = overlay_images(alignment_data)
    update_figure(figure, canvas, anatomy_combined_array, app)


def update_figure(figure, canvas, image, app):
    print("ypdating figure")

    figure.clear()
    print("cleared")
    axis = figure.add_subplot(111)
    axis.set_xlim(0.0, np.shape(image)[1])
    axis.set_ylim(np.shape(image)[1], 0.0)
    axis.imshow(image)
    axis.axis('off')
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)

    print("plotted")
    figure.tight_layout(pad=0)

    canvas.draw()
    print("draw")
    canvas.update()
    print("update")
    # global app
    app.processEvents()
    print("process")


def rebinarise_array(alignment_data):
    # Load Variables We Need From Dictionary
    atlas_array = alignment_data["atlas_array"]

    # Binarise the array
    atlas_array = np.divide(atlas_array, np.max(atlas_array))
    atlas_array = np.around(atlas_array, decimals=0)
    atlas_array = np.where(atlas_array >= 0.9, 1, 0)
    atlas_array = np.ndarray.astype(atlas_array, int)

    # Put These Back Into The Dictionary
    alignment_data["atlas_array"] = atlas_array

    return alignment_data


def majority_3_by_3(array):
    majority_array = np.zeros(np.shape(array))

    window_size = 10
    half_window = int(window_size / 2)

    image_height = np.shape(array)[0]
    image_width = np.shape(array)[1]

    print("starting")

    for y in range(image_height):
        print(" y:", y)
        for x in range(image_width):
            top = y - half_window
            left = x - half_window
            window_height = top + window_size
            window_width = left + window_size

            top = np.clip(top, a_min=0, a_max=image_height)
            left = np.clip(left, a_min=0, a_max=image_width)
            window_height = np.clip(window_height, a_min=0, a_max=image_height)
            window_width = np.clip(window_width, a_min=0, a_max=image_width)

            sample = array[top:window_height, left:window_width]
            sample = np.ndarray.flatten(sample)

            mode = int(stats.mode(sample)[0][0])
            majority_array[y, x] = mode

    return majority_array


def display_mapping(mapping):
    regions = np.unique(mapping)
    print("Number of regions", np.shape(regions)[0])

    for region in regions:
        plt.title(region)
        display = np.where(mapping == region, 2, 0)
        display = np.add(display, atlas_array)
        plt.imshow(display)
        plt.show()


def map_regions():
    print("Setting Mask")

    # Load Alignment Data
    alignment_data = window_instance.alignment_data

    original_atlas_array = alignment_data["original_atlas_array"]
    atlas_array = alignment_data["atlas_array"]
    anatomical_image_array = alignment_data["anatomical_image_array"]
    anatomical_image_data = alignment_data["anatomical_image_data"]
    background_array = alignment_data["background_array"]
    mask_regions_data = alignment_data["mask_regions_data"]

    template_x = alignment_data["template_x"]
    template_y = alignment_data["template_y"]
    rotation = alignment_data["rotation"]
    scale = alignment_data["scale"]

    bounding_height = alignment_data["bounding_height"]
    bounding_width = alignment_data["bounding_width"]
    background_width = alignment_data["background_width"]
    background_height = alignment_data["background_height"]
    original_bounding_height = alignment_data["original_bounding_height"]
    original_bounding_width = alignment_data["original_bounding_width"]
    image_x_start = alignment_data["image_x_start"]
    image_y_start = alignment_data["image_y_start"]
    original_height = alignment_data["original_height"]
    original_width  = alignment_data["original_width"]
    output_file = alignment_data["output_file"]
    rotation_file = alignment_data["rotation_file"]

    print("loaded everything")

    # Transform Mask
    bounding_regions_mask = place_array_in_bounding_rect(mask_regions_data, original_bounding_height,
                                                         original_bounding_width)
    bounding_regions_mask = ndimage.rotate(input=bounding_regions_mask, angle=rotation, reshape=False)
    bounding_regions_mask = resize(bounding_regions_mask, (bounding_height, bounding_width))

    # Put Mask Into Backround
    map_in_background = np.copy(background_array)
    map_in_background[template_y:template_y + bounding_height,
    template_x:template_x + bounding_width] = bounding_regions_mask

    # Take Central Subset
    original_height, original_width = np.shape(anatomical_image_data)
    final_mask = map_in_background[image_y_start:image_y_start + original_height,
                 image_x_start:image_x_start + original_width]

    np.save(output_file, final_mask)
    np.save(rotation_file, rotation)
    print("mapped")


class atlas_matching_window(QWidget):

    def __init__(self, alignment_data, parent=None):
        super(atlas_matching_window, self).__init__(parent)

        self.alignment_data = alignment_data

        # Setup Window
        self.setWindowTitle("Atlas Matching")
        self.setGeometry(0, 0, 1000, 500)
        self.show()

        # Create Figures
        self.anatomy_figure = Figure()
        self.activity_figure = Figure()
        self.anatomy_canvas = FigureCanvas(self.anatomy_figure)

        # Create Buttons
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

        self.map_button = QPushButton("Map Regions")
        self.map_button.clicked.connect(map_regions)

        # Add Labels
        self.x_label = QLabel()
        self.y_label = QLabel()
        self.height_label = QLabel()
        self.width_label = QLabel()
        self.angle_label = QLabel()

        self.y_label.setText("y: " + str(self.alignment_data["template_y"]))
        self.x_label.setText("x: " + str(self.alignment_data["template_x"]))
        self.width_label.setText("Width: " + str(self.alignment_data["bounding_width"]))
        self.height_label.setText("Height: " + str(self.alignment_data["bounding_height"]))
        self.angle_label.setText("Angle: " + str(self.alignment_data["rotation"]))

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.layout.addWidget(self.anatomy_canvas, 0, 0, 13, 8)

        self.layout.addWidget(self.left_button, 0, 16, 1, 1)
        self.layout.addWidget(self.right_button, 1, 16, 1, 1)
        self.layout.addWidget(self.up_button, 2, 16, 1, 1)
        self.layout.addWidget(self.down_button, 3, 16, 1, 1)
        self.layout.addWidget(self.rotate_clockwise_button, 4, 16, 1, 1)
        self.layout.addWidget(self.rotate_counterclockwise_button, 5, 16, 1, 1)
        self.layout.addWidget(self.enlarge_button, 6, 16, 1, 1)
        self.layout.addWidget(self.shrink_button, 7, 16, 1, 1)

        self.layout.addWidget(self.x_label, 8, 16, 1, 1)
        self.layout.addWidget(self.y_label, 9, 16, 1, 1)
        self.layout.addWidget(self.height_label, 10, 16, 1, 1)
        self.layout.addWidget(self.width_label, 11, 16, 1, 1)
        self.layout.addWidget(self.angle_label, 12, 16, 1, 1)

        self.layout.addWidget(self.map_button, 13, 16, 1, 1)

    def move_left(self):
        self.alignment_data["template_x"] = self.alignment_data["template_x"] - 10
        self.x_label.setText("x: " + str(self.alignment_data["template_x"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def move_right(self):
        self.alignment_data["template_x"] = self.alignment_data["template_x"] + 10
        self.x_label.setText("x: " + str(self.alignment_data["template_x"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def move_up(self):
        self.alignment_data["template_y"] = self.alignment_data["template_y"] - 10
        self.x_label.setText("y: " + str(self.alignment_data["template_y"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def move_down(self):
        self.alignment_data["template_y"] = self.alignment_data["template_y"] + 10
        self.x_label.setText("y: " + str(self.alignment_data["template_y"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def rotate_clockwise(self):
        self.alignment_data["rotation"] = self.alignment_data["rotation"] - 2
        self.angle_label.setText("Angle: " + str(self.alignment_data["rotation"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def rotate_counterclockwise(self):
        self.alignment_data["rotation"] = self.alignment_data["rotation"] + 2
        self.angle_label.setText("Angle: " + str(self.alignment_data["rotation"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def enlarge(self):
        self.alignment_data["bounding_width"] = self.alignment_data["bounding_width"] + 10
        self.alignment_data["bounding_height"] = self.alignment_data["bounding_height"] + 10
        self.width_label.setText("Width: " + str(self.alignment_data["bounding_width"]))
        self.height_label.setText("Height: " + str(self.alignment_data["bounding_height"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)

    def shrink(self):
        self.alignment_data["bounding_width"] = self.alignment_data["bounding_width"] - 10
        self.alignment_data["bounding_height"] = self.alignment_data["bounding_height"] - 10
        self.width_label.setText("Width: " + str(self.alignment_data["bounding_width"]))
        self.height_label.setText("Height: " + str(self.alignment_data["bounding_height"]))
        draw_image(self.anatomy_figure, self.anatomy_canvas, self.alignment_data)


def get_blue_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Blue" in file:
            return base_directory + "/" + file

def get_violet_file(base_directory):
    file_list = os.listdir(base_directory)
    for file in file_list:
        if "Violet" in file:
            return base_directory + "/" + file


def perform_template_masking(base_directory):
    global window_instance

    app = QApplication(sys.argv)

    blue_file = get_blue_file(base_directory)
    violet_file = get_violet_file(base_directory)

    print("Blue File", blue_file)
    print("Violet File", violet_file)

    # File Locations
    mask_boundaries_file = "/home/matthew/Documents/Allen_Atlas_Templates/Outline_array.npy"
    mask_regions_file = "/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy"
    anatomical_image_file = base_directory + "/max_projection.npy"
    output_file = base_directory + "/Mask.npy"
    rotation_file = base_directory + "/rotation.npy"

    # Load Data
    blue_data_container = h5py.File(blue_file, 'r')
    blue_data = blue_data_container["Data"]

    violet_data_container = h5py.File(violet_file, 'r')
    violet_data = violet_data_container["Data"]

    # Check LED Colours
    check_led_colours(blue_data, violet_data)

    # Get Max Projection
    get_max_projection(blue_data, base_directory)

    # Load Files Into Arrays
    anatomical_image_data = np.load(anatomical_image_file)
    mask_boundaries_data = np.load(mask_boundaries_file)
    mask_regions_data = np.load(mask_regions_file)

    print("Anatomical image data", np.shape(anatomical_image_data))

    # Get Image Properties
    image_height = np.shape(anatomical_image_data)[0]
    image_width = np.shape(anatomical_image_data)[1]

    # Preset Image Dimensions
    original_bounding_width = 900
    original_bounding_height = 900
    image_x_start = 200
    image_y_start = 200
    background_width = 1000
    background_height = 1000

    # Transformation Properties
    template_y = 0
    template_x = 0
    bounding_width = original_bounding_height
    bounding_height = original_bounding_width
    rotation = -8
    scale = 1

    # Place Mask Into Bounding Box
    original_atlas_array = place_array_in_bounding_rect(mask_boundaries_data, bounding_height, bounding_width)
    atlas_array = np.copy(original_atlas_array)

    # Create Background Array
    background_array = np.zeros((background_height, background_width))

    # Create Image Array from camera
    anatomical_image_array = create_background_array(anatomical_image_data, background_height, background_width,
                                                     image_y_start, image_x_start, image_height, image_width)

    # Put all this into a handy dictionary so its easier to pass to functions
    alignment_data = {

        # Info About Transformations
        "template_x": template_x,
        "template_y": template_y,
        "rotation": rotation,
        "scale": scale,

        # Info about the edges of the Images
        "bounding_width": bounding_width,
        "bounding_height": bounding_height,
        "background_width": background_width,
        "background_height": background_height,
        "image_x_start": image_x_start,
        "image_y_start": image_x_start,
        "original_bounding_width": original_bounding_width,
        "original_bounding_height":original_bounding_height,

        # The Data Itself
        "original_atlas_array": original_atlas_array,
        "atlas_array": atlas_array,
        "anatomical_image_data": anatomical_image_data,
        "anatomical_image_array": anatomical_image_array,
        "background_array": background_array,
        "mask_regions_data": mask_regions_data,
        "original_height": image_height,
        "original_width": image_width,

        #The App
        "app": app,
        "output_file": output_file,
        "rotation_file": rotation_file
    }

    window_instance = atlas_matching_window(alignment_data)

    sys.exit(app.exec_())


global window_instance

