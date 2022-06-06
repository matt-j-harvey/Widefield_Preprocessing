import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append("/home/matthew/Documents/Local_NMF/locaNMF/locanmf")



base_directory = r"/media/matthew/Expansion/Widefield_Analysis/NXAK20.1B/2021_09_28_Discrimination_Imaging"

# Load Mask
mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

# Get Indicies
mask = np.where(mask > 0.1, 1, 0)
mask = mask.astype(int)
flat_mask = np.ndarray.flatten(mask)
indicies = np.argwhere(flat_mask)
indicies = np.ndarray.astype(indicies, int)
indicies = np.ndarray.flatten(indicies)

# Load Components
local_nmf_components = np.load(os.path.join(base_directory, "Local_NMF_Components.npy"), allow_pickle=True)[()]



print(np.shape(local_nmf_components))

for x in range(200):
    component_data = local_nmf_components[:, x]