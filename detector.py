from functions import predict
from extract_data import extract
from image_converter import image_to_mat
from scipy.io import loadmat
import numpy as np


def project(image):
    theta1 = np.loadtxt("D:\\ML\\Hanwriting ML\\data\\theta1.txt")
    theta2 = np.loadtxt("D:\\ML\\Hanwriting ML\\data\\theta2.txt")
    
    a = image_to_mat(image)
    
    return predict(a, theta1, theta2)