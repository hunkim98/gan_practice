import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
from manual_diff_augment import DiffAugment, rand_brightness, rand_cutout, rand_contrast, rand_saturation, make_rgb_to_gray
import dezero
from dezero import cuda
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.models import Sequential
from dezero.optimizers import Adam
use_gpu = cuda.gpu_enable
print(use_gpu)

dict_data = np.load('8bit_characters_50x50.npz')

def make_rgb_to_black(batch, threshold=0.9):
    # Define a white threshold (adjust as needed)
    white_threshold = threshold# You can adjust this threshold value

    # Convert the batch of images to grayscale (if they are not already)
    batch_gray = make_rgb_to_gray(batch)

    # Create a mask where non-white pixels are set to black
    mask = batch_gray < white_threshold

    # Set non-white pixels to black
    batch[mask] = [0, 0, 0]  # Assuming the images are in RGB format

    return batch

def grayscale_diff(x,y):
    gray_x = make_rgb_to_gray(x)
    gray_y = make_rgb_to_gray(y)
    squared = (gray_x - gray_y) ** 2
    diff = squared.sum()
    return diff

def simply_diff(x,y):
    squared = (x - y) ** 2
    black_x = make_rgb_to_black(x)
    black_y = make_rgb_to_black(y)
    squared = (black_x - black_y) ** 2
    diff = squared.sum()
    return diff

data = dict_data['arr_0']

def plot_images(imgs, grid_size = 10, epoch = 0, loss = 0):
    """
    imgs: vector containing all the numpy images
    grid_size: 2x2 or 5x5 grid containing images
    """
     
    fig = plt.figure(figsize = (8, 8))
    columns = rows = grid_size
    plt.title(f"Training Images at epoch {epoch}, loss: {loss}")
    for i in range(1, columns*rows +1):
        if i >= len(imgs):
            break
        plt.axis("off")
        fig.add_subplot(rows, columns, i)
        img = imgs[i]
        img = (img)
        plt.imshow(img)
    plt.savefig(f"test_black.png")
    plt.show()
    plt.close()

print(grayscale_diff(data[0:32], data[6:38]), ' this is diff')
print(simply_diff(data[0:32], data[6:38]), ' this is diff')

plot_images(make_rgb_to_black(data[0:32], 0.9))