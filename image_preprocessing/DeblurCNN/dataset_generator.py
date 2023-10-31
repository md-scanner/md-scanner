import os
import cv2
import numpy as np
import random
import datetime

# print timestamp
print("Starting dataset generation...")
print(datetime.datetime.now())

# get the images folder
images_folder = "sources/"

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

for file in files:
    # get the source path
    source_path = images_folder + file

    img = cv2.imread(source_path)

    # Define the patch size
    patch_size = (200, 200)

    # Define the stride (how much the window shifts between patches), = 200 for non-overlapping
    stride = 200 

    # patch number
    patch_number = 0

            # Iterate over rows
    for y in range(0, img.shape[0] - patch_size[0] + 1, stride):
        # Iterate over columns
        for x in range(0, img.shape[1] - patch_size[1] + 1, stride):
            
            # Extract the patch
            patch = img[y:y+patch_size[0], x:x+patch_size[1]]
            # save the image to the sharp folder
            destination_path = sharp_images_folder +"sharp_"+ str(patch_number) +"_" + file
            cv2.imwrite(destination_path, patch)

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

            # pick a random int between 0 and 1
            random_int = random.randint(0, 1)

            if random_int == 0:
                # Apply the vertical kernel.
                blurred = cv2.filter2D(patch, -1, kernel_v)
            else:
                # Apply the horizontal kernel.
                blurred = cv2.filter2D(patch, -1, kernel_h)

            # save the image to the blurred folder
            destination_path = blurred_images_folder +"blurred_" + str(patch_number) +"_" + file
            cv2.imwrite(destination_path, blurred)

            patch_number += 1

    if img.shape[0] % patch_size[0] != 0:
        # select the part of the image that is not divisible
        patches = img[img.shape[0] - patch_size[0]:img.shape[0], 0:img.shape[1]]

        # divide the patch in patches
        for i in range(0, patches.shape[1], stride):
            # select the patch
            patch = patches[:patch_size[0], i:i + patch_size[1]]

            # save the patch
            destination_path = sharp_images_folder +"sharp_"+ str(patch_number) +"_" + file
            cv2.imwrite(destination_path, patch)

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

            # pick a random int between 0 and 1
            random_int = random.randint(0, 1)

            if random_int == 0:
                # Apply the vertical kernel.
                blurred = cv2.filter2D(patch, -1, kernel_v)
            else:
                # Apply the horizontal kernel.
                blurred = cv2.filter2D(patch, -1, kernel_h)

            # save the patch
            destination_path = blurred_images_folder +"blurred_" + str(patch_number) +"_" + file
            cv2.imwrite(destination_path, patch)

            # divide the patch in patches
            patch_number += 1

    
    if img.shape[1] % patch_size[1] != 0:
        # select the part of the image that is not divisible
        patches = img[0:img.shape[0], img.shape[1] - patch_size[1]:img.shape[1]]

        # divide the patch in patches
        for i in range(0, patch.shape[0], stride):
            # select the patch
            patch = patches[i:i + patch_size[0], :patch_size[1]]

            # save the patch
            destination_path = sharp_images_folder +"sharp_"+ str(patch_number) +"_" + file
            cv2.imwrite(destination_path, patch)

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

            # pick a random int between 0 and 1
            random_int = random.randint(0, 1)

            if random_int == 0:
                # Apply the vertical kernel.
                blurred = cv2.filter2D(patch, -1, kernel_v)
            else:
                # Apply the horizontal kernel.
                blurred = cv2.filter2D(patch, -1, kernel_h)

            # save the patch
            destination_path = blurred_images_folder +"blurred_" + str(patch_number) +"_" + file
            cv2.imwrite(destination_path, patch)

            # divide the patch in patches
            patch_number += 1

print("Dataset generation completed!")
print(datetime.datetime.now())