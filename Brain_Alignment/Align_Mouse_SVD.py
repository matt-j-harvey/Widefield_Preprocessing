import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import tqdm
import cv2

from Widefield_Utils import widefield_utils



def get_transformed_atlas_outlines():

    # Load Atlas Regions
    atlas_outlines = np.load(r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outlines = widefield_utils.transform_mask_or_atlas_300(atlas_outlines, atlas_alignment_dict)

    return atlas_outlines




def get_transformed_atlas_regions():

    # Load Atlas Dict
    atlas_alignment_dict = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]
    atlas_region_dict = np.load(os.path.join(r"/home/matthew/Documents/Giuseppie_Registration_Stuff/Allen_Region_Dict.npy"), allow_pickle=True)[()]

    pixel_labels = atlas_region_dict['pixel_labels']

    # Transform Atlas
    pixel_labels = widefield_utils.transform_atlas_regions(pixel_labels, atlas_alignment_dict)

    return pixel_labels



def create_example_video(base_directory, raw_u, registered_u, temporal_components, sample_size=5000, sample_start=4000):

    # Register Atlas
    atlas_outlines = get_transformed_atlas()
    atlas_indicies = np.nonzero(atlas_outlines)

    # Reconstruct Sample Video
    sample_end = sample_start + sample_size
    registered_video_sample = np.dot(registered_u, temporal_components[:, sample_start:sample_end])
    raw_video_sample = np.dot(raw_u, temporal_components[:, sample_start:sample_end])

    # Create Video File
    video_name = os.path.join(base_directory, "Example_Registration.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1000, 500), fps=30)  # 0, 12

    # Create Colourmap
    widefield_colourmap = widefield_utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)

    # Create Figure
    figure_1 = plt.figure(figsize=(10, 5))
    canvas = FigureCanvasAgg(figure_1)
    rows = 1
    columns = 2

    window_size = 2
    for sample_index in tqdm(range(sample_size), desc="Creating Sample Video"):

        # Create Axes
        raw_axis = figure_1.add_subplot(rows, columns, 1)
        registered_axis = figure_1.add_subplot(rows, columns, 2)

        # Get Frame Data - Smooth With Rolling Average
        sample_raw_frame = raw_video_sample[:, :, sample_index:sample_index + window_size]
        sample_registered_frame = registered_video_sample[:, :, sample_index:sample_index + window_size]

        sample_raw_frame = np.mean(sample_raw_frame, axis=2)
        sample_registered_frame = np.mean(sample_registered_frame, axis=2)

        # Set Colours
        sample_raw_frame = widefield_colourmap.to_rgba(sample_raw_frame)
        sample_registered_frame = widefield_colourmap.to_rgba(sample_registered_frame)

        # Add Atlas Outlines
        sample_registered_frame[atlas_indicies] = [1,1,1,1]

        # Display Images
        raw_axis.imshow(sample_raw_frame)
        registered_axis.imshow(sample_registered_frame)

        # Set Titles
        raw_axis.set_title("Unregstered Data")
        registered_axis.set_title("Registered Data")

        # Remove Axis
        raw_axis.axis('off')
        registered_axis.axis('off')

        # Draw Canvas
        figure_1.canvas.draw()

        # Write To Video
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        print(np.shape(image_from_plot))
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)

        plt.clf()
    cv2.destroyAllWindows()
    video.release()


def align_session_svd(base_directory):

    # Load Within Mouse Alignment Dict
    within_mouse_alignment_dict = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Load Across Mouse Alignment Dict
    across_mouse_alignment_dict = widefield_utils.load_across_mice_alignment_dictionary(base_directory)

    # Load U
    spatial_components = np.load(os.path.join(base_directory, "Churchland_Preprocessing", "U.npy"))
    image_height, image_width, n_components = np.shape(spatial_components)

    # Create Array To Hold Registered U
    registered_u = np.zeros(np.shape(spatial_components), dtype=float)

    # Iterate Through Components and Transform Each One
    for component_index in tqdm(range(n_components), desc="Registering Components"):

        component_data = spatial_components[:, :, component_index]

        # Align Within Mouse
        component_data = widefield_utils.transform_image(component_data, within_mouse_alignment_dict)

        # Align Across Mouse
        component_data = widefield_utils.transform_image(component_data, across_mouse_alignment_dict)

        # Add To Registered Components
        registered_u[:, :, component_index] = component_data

    # Save Registered U
    np.save(os.path.join(base_directory, "Churchland_Preprocessing", "Registered_U.npy"), registered_u)

    # View Sample As a Sanity Check
    temporal_components = np.load(os.path.join(base_directory, "Churchland_Preprocessing", "Corrected_SVT.npy"))
    create_example_video(base_directory, spatial_components, registered_u, temporal_components, sample_size=5000, sample_start=4000)








session_list = [

                r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_02_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_21_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_02_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_15_Spontaneous",

                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_10_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_23_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_21_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_20_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_01_29_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_04_Spontaneous",

]

for base_directory in session_list:
    align_session_svd(base_directory)