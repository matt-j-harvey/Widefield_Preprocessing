import os
import numpy as np
from skimage.transform import resize

def create_downsampled_mask_dict(base_directory):
    print("Getting Downsampled Mask For Session; ", base_directory)

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    mask_dict = {
    "indicies":indicies,
    "image_height":image_height,
    "image_width":image_width
    }

    np.save(os.path.join(base_directory, "Downsampled_mask_dict.npy"), mask_dict)



"""
base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Mutant_Data/NXAK16.1B"
session_list = os.listdir(base_directory)

for session in session_list:
    session_directory = os.path.join(base_directory, session)
    create_downsampled_mask_dict(session_directory)
"""
