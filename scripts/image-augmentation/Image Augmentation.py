import os
import cv2
import numpy as np
import random

# Get the current working directory
folder_path = os.getcwd()

if not os.path.exists('Sample'):
        os.makedirs('Sample')
        print(f"Folder '{'Sample'}' created.")
else:
    print(f"Folder '{'Sample'}' already exists.")

# List of valid image extensions
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]

# Initialize a list to store the images
images = []

# Iterate over all files in the current directory
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is an image
    if os.path.isfile(file_path) and os.path.splitext(file_name.lower())[1] in valid_extensions:
        try:
            # Read the image
            img = cv2.imread(file_path)
            if img is not None:
                # Optionally resize to ensure uniform dimensions (e.g., 256x256)
                img = cv2.resize(img, (600, 300))
                # Append the image to the list
                images.append(img)
                print(f"Loaded {file_name}: Shape - {img.shape}")
            else:
                print(f"Failed to read {file_name}")
        except Exception as e:
            print(f"Could not process {file_name}: {e}")

# Convert the list of images to a 4D NumPy array
# The shape will be (num_images, height, width, channels)
images_array = np.array(images)

print(f"4D array shape: {images_array.shape}")


scale_factor=float(input('Enter Scale Factor (0 to 1): '))
num_of_imgs=int(input('Enter Number of Images: '))

def generate_patches(images, scale_factor):

    num_images, h, w, c = images.shape
    patch_h, patch_w = int(h * scale_factor), int(w * scale_factor)

    patch_h = min(patch_h, h)
    patch_w = min(patch_w, w)

    # Generate random coordinates for the top-left corner of the patch
    y = random.randint(0, h - patch_h)
    x = random.randint(0, w - patch_w)

    # Extract patches using slicing
    patches = images[:, y:y+patch_h, x:x+patch_w, :]
    
    # Resize all patches back to the original dimensions
    resized_patches = np.array([cv2.resize(patch, (w, h)) for patch in patches])

    # Split into a list of individual image patches
    # patches_list = [resized_patches[i] for i in range(num_images)]

    return resized_patches, [x, y]


def generate_patches_2(images, scale_factor, i):

    num_images, h, w, c = images.shape
    patch_h, patch_w = int(h * scale_factor), int(w * scale_factor)

    patch_h = min(patch_h, h)
    patch_w = min(patch_w, w)

    # Generate random coordinates for the top-left corner of the patch
    y = random.randint(0, h - patch_h)
    x = random.randint(0, w - patch_w)

    patches = []
    for j in range(num_images):
        # Extract the patch for each image
        patch = images[j, y:y+patch_h, x:x+patch_w]
        # Resize the patch back to the original dimensions
        patch_resized = cv2.resize(patch, (w, h))
        patches.append(patch_resized)
        cv2.imwrite(f'Sample/patch_{i+1}_{j+1}_{scale_factor}_{y}_{x}.png', patch_resized)

    # Convert the list of patches into a 4D array
    patches_array = np.array(patches)
    return patches_array, [x, y]


for i in range(num_of_imgs):
    patch, [x,y]=generate_patches_2(images_array, scale_factor,i)
    # cv2.imshow(patch)
    print(patch.shape)
    print([x,y])


