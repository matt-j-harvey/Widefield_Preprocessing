import numpy as np
import cv2


def reconstruct_raw_video(blue_array, violet_array, reconstructed_video_file):
    print("Reconstructing Video ")

    #Take Sample of Data
    blue_array   = blue_array[:, 1000:2000]
    violet_array = violet_array[:, 1000:2000]

    #Transpose Data
    blue_array = np.transpose(blue_array)
    violet_array = np.transpose(violet_array)

    # Convert From 16 bit to 8 bit
    blue_array   = np.divide(blue_array, 65536)
    violet_array = np.divide(violet_array, 65536)

    blue_array = np.multiply(blue_array, 255)
    violet_array = np.multiply(violet_array, 255)

    # Get Original Pixel Dimenions
    frame_width = 608
    frame_height = 600

    video_name = reconstructed_video_file
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(frame_width * 2, frame_height), fps=30)  # 0, 12

    number_of_frames = np.shape(blue_array)[0]


    for frame in range(number_of_frames):

        blue_frame = blue_array[frame]
        blue_frame = np.ndarray.astype(blue_frame, np.uint8)

        violet_frame = violet_array[frame]
        violet_frame = np.ndarray.astype(violet_frame, np.uint8)

        blue_frame   = np.reshape(blue_frame, (600,608))
        violet_frame = np.reshape(violet_frame, (600, 608))

        image = np.hstack((violet_frame, blue_frame))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()
