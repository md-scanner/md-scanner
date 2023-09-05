import os
import cv2
import numpy as np
import random

# get the images folder
images_folder = "images/"

# get the dataset folder
dataset_folder = "dataset/"

# create the dataset folder if it does not exist
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

sharp_images_folder = dataset_folder + "sharp/"
blurred_images_folder = dataset_folder + "blurred/"

# create the sharp and blurred folders if they do not exist
if not os.path.exists(sharp_images_folder):
    os.makedirs(sharp_images_folder)

if not os.path.exists(blurred_images_folder):
    os.makedirs(blurred_images_folder)

# get 200 random images from the images folder
files = os.listdir(images_folder)
files = random.sample(files, 200)

for file in files:
    # get the source path
    source_path = images_folder + file

    # get the destination path
    destination_path = sharp_images_folder + file

    # copy the file
    with open(source_path, 'rb') as f:
        with open(destination_path, 'wb') as g:
            g.write(f.read())

    img = cv2.imread(destination_path)
    
    # Specify the kernel size.
    # Random kernel size to have different types of blur
    kernel_size = np.random.randint(5, 15)
    
    # Create the vertical kernel.
    kernel_v = np.zeros((kernel_size, kernel_size))
    
    # Create a copy of the same for creating the horizontal kernel.
    kernel_h = np.copy(kernel_v)
    
    # Fill the middle row with ones.
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    
    # Normalize.
    kernel_v /= kernel_size
    kernel_h /= kernel_size

    # Apply the vertical kernel.
    vertical_mb = cv2.filter2D(img, -1, kernel_v)
    
    # Apply the horizontal kernel.
    horizonal_mb = cv2.filter2D(img, -1, kernel_h)

    # save the image to the blurred folder
    destination_path = blurred_images_folder +"blurred_" + file
    cv2.imwrite(destination_path, vertical_mb)