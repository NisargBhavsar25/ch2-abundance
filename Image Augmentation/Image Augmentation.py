import random
import cv2

image=cv2.imread(<file_path>)


scale_factor=float(input('Enter Scale Factor (0 to 1): '))
num_of_imgs=int(input('Enter Number of Images: '))


def generate_patch(image, scale_factor):
    
    h, w, _ = image.shape
    patch_h, patch_w = int(h * scale_factor), int(w * scale_factor)

    patch_h = min(patch_h, h)
    patch_w = min(patch_w, w)

    y = random.randint(0, h - patch_h)
    x = random.randint(0, w - patch_w)

    patch = image[y:y+patch_h, x:x+patch_w]

    patch_resized = cv2.resize(patch, (w, h))

    return patch_resized, [x,y]



for i in range(num_of_imgs):
    patch, [x,y]=generate_patch(img, scale_factor)
    cv2.imshow(patch)
    print([x,y])


