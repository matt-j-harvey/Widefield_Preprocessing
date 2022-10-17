


def load_downsampled_mask(base_directory):

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

    return indicies, image_height, image_width


def visualise_heamocorrection_changes(base_directory):


    # Get Filenames

    blue_df_file = os.path.join(base_directory, "Blue_DF.hdf5")
    violet_df_file = os.path.join(base_directory, "violet_DF.hdf5")
    delta_f_file = os.path.join(base_directory, "Delta_F.hdf5")

    # Get Data Structure
    number_of_pixels, number_of_images = np.shape(blue_matrix)

    # Open Files
    blue_df_file_container = h5py.File(blue_df_file, 'w')
    violet_df_file_container = h5py.File(violet_df_file, 'w')
    delta_f_file_container = h5py.File(delta_f_file, 'w')

    # Create Datasets
    blue_df_dataset = blue_df_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)
    violet_df_dataset = violet_df_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)
    df_dataset = delta_f_file_container.create_dataset("Data", (number_of_images, number_of_pixels), dtype=np.float32, chunks=True, compression=False)

    # Define Chunking Settings
    preferred_chunk_size = 5000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = Preprocessing_Utils.get_chunk_structure(preferred_chunk_size, number_of_pixels)

    print("Heamocorrecting")
    for chunk_index in tqdm(range(number_of_chunks)):
        chunk_start = int(chunk_starts[chunk_index])
        chunk_stop = int(chunk_stops[chunk_index])

        # Extract Data
        blue_data = blue_matrix[chunk_start:chunk_stop]
        violet_data = violet_matrix[chunk_start:chunk_stop]

        # Remove Early Cutoff
        blue_data = blue_data[:, exclusion_point:]
        violet_data = violet_data[:, exclusion_point:]

        # Remove NaNs
        blue_data = np.nan_to_num(blue_data)
        violet_data = np.nan_to_num(violet_data)

        # Calculate Delta F
        blue_data = calculate_delta_f(blue_data)
        violet_data = calculate_delta_f(violet_data)

        pre_filter_blue_mean = np.mean(blue_data, axis=0)
        pre_filter_violet_mean = np.mean(violet_data, axis=0)

        print("Pre Filter shape", np.shape(blue_data))

        # Lowcut Filter
        if lowcut_filter == True:
            blue_data = perform_lowcut_filter(blue_data, b, a)
            violet_data = perform_lowcut_filter(violet_data, b, a)

        print("Post Filter shape", np.shape(blue_data))

        post_filter_blue_mean = np.mean(blue_data, axis=0)
        post_filter_violet_mean = np.mean(violet_data, axis=0)

        # Plot Traces
        plot_pre_and_post_filter_traces(pre_filter_blue_mean, pre_filter_violet_mean, post_filter_blue_mean, post_filter_violet_mean, output_directory, chunk_index)

        # Perform Regression
        processed_data = heamocorrection_regression(blue_data, violet_data)

        # Transpose
        processed_data = np.transpose(processed_data)
        blue_data = np.transpose(blue_data)
        violet_data = np.transpose(violet_data)

        # Convert to 32 Bit Float
        processed_data = np.ndarray.astype(processed_data, np.float32)
        blue_data = np.ndarray.astype(blue_data, np.float32)
        violet_data = np.ndarray.astype(violet_data, np.float32)

        # Put Back
        blue_df_dataset[exclusion_point:, chunk_start:chunk_stop] = blue_data
        violet_df_dataset[exclusion_point:, chunk_start:chunk_stop] = violet_data
        df_dataset[exclusion_point:, chunk_start:chunk_stop] = processed_data

    # Close Motion Corrected Data
    motion_corrected_data_container.close()

    # Close Heamocorrection Files
    blue_df_file_container.close()
    violet_df_file_container.close()
    delta_f_file_container.close()


base_directory = r"/media/matthew/External_Harddrive_2/Widefield_Data_New_Pipeline/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging/Downsampled_Data"