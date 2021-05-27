from PIL import Image, ImageOps
import numpy as np


def image_to_mat(image_file):
    image = Image.open(f"D:\\ML\\Hanwriting ML\\{image_file}")

    image = ImageOps.grayscale(image)
    pixel_mat = np.array(image)

    # r, g, b = pixel_mat[:,:,0], pixel_mat[:,:,1], pixel_mat[:,:,2]
    # pixel_vec = 0.2989 * r + 0.5870 * g + 0.1140 * b


    m, n = pixel_mat.shape

    pixel_vec = np.zeros((1, m*n))
    
    k = 0
    for i in range(m):
        for j in range(n):
            pixel_vec[0][k] = pixel_mat[i][j]
            k+=1

    return pixel_vec/255