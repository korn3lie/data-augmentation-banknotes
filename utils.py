import cv2
import numpy as np
import os
from PIL import Image


def split_and_stack(image: np.ndarray, p: int = 1, n: int = 10) -> np.ndarray:
    """
    Splits an image into two parts, deletes n rows & columns of pixels, and puts it back together.

    Args:
        image: The input image as a numpy array.
        p: The row/column size in pixels to be deleted. Defaults to 1.
        n: The number of rows & columns to be deleted. Defaults to 10.

    Returns:
        The modified image as a numpy array.
    """
    height, width, _ = image.shape
    offset = 10

    for _ in range(n):
        split_point = np.random.randint(offset, width - offset)
        part1 = image[:split_point, :]
        part2 = image[split_point + p:, :]
        image = np.vstack((part1, part2))

        split_point = np.random.randint(offset, height - offset)
        part1 = image[:, :split_point]
        part2 = image[:, split_point + p:]
        image = np.hstack((part1, part2))

    return image

# ----------------------------------------------------------------------------

def augment(image: np.ndarray, output_resolution: tuple[int, int]) -> np.ndarray:
    """Applies random augmentations to an image (border crop, lighting, resize)

    Args:
        image: An array representing the image
        output_resolution: A tuple of integers representing the output resolution (width, height)

    Returns:
        An augmented image with the specified output resolution
    """
    
    height, width, _ = image.shape

    # Randomly crop borders in range (0, offset)
    offset = 4
    random = np.random.randint
    image = image[random(0, offset):height - random(0, offset), 
                  random(0, offset):width - random(0, offset)]

    # Simulate lighting changes (e.g., brightness and contrast)
    #alpha = 1.0 + np.random.uniform(-0.1, 0.1)
    beta = np.random.uniform(-50, 20)
    image = cv2.convertScaleAbs(image, alpha=1, beta=beta)

    # Resize the image
    image = cv2.resize(image, output_resolution)

    return image

# ----------------------------------------------------------------------------

def rename_files(folder_path: str, n: int = 5) -> None:
    """Renames all files in a folder with a numerical index
    
    Args:
        folder_path: The path to the folder containing the files
        n: The number of leading zeros for the name (default: 5)
    """
    folder_path = os.path.join(folder_path, '')

    for index, filename in enumerate(os.listdir(folder_path), start=1):
        new_file_name = f'{str(index).zfill(n)}.png'

        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_file_name)

        os.rename(old_path, new_path)

    print('Done!')

# ----------------------------------------------------------------------------
    
def is_image(filename):
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in image_extensions

# ----------------------------------------------------------------------------
    
def resize_and_rotate(output_resolution, directory_path, output_path):

    os.makedirs(output_path, exist_ok=True)

    for filename in os.listdir(directory_path):
        if not is_image(filename):
            continue
        img_path = os.path.join(directory_path, filename)
        image = cv2.imread(img_path)
        image = cv2.rotate(image, cv2.ROTATE_180)
        image = cv2.resize(image, output_resolution)

        img_path_save = os.path.join(output_path, filename)
        if not cv2.imwrite(img_path_save, image):
            raise Exception("Could not write image")
        
    print(f"Done.")

# ----------------------------------------------------------------------------