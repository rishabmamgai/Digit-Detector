from extract_data import extract, rand_initialise, split_data
from functions import neural_network, cost_function
from image_converter import image_to_mat
import numpy as np
import scipy.optimize as opt


print("\nReading data....\n")

X, y = extract("mnist-original.mat")

X_train, y_train, X_test, y_test = split_data(X, y)


# Parameters
m, n = X_train.shape
input_layer_size = n
hidden_units = 40
labels = 10


theta1 = rand_initialise(hidden_units, n)
theta2 = rand_initialise(labels, hidden_units)

theta = np.r_[theta1.flatten(), theta2.flatten()]


Lambda = 0.1
iters = 50
myargs = (X_train, y_train, Lambda, input_layer_size, hidden_units, labels)

# Optimizing
result = opt.minimize(neural_network, x0=theta, args=myargs, options={'disp':True, 'maxiter':iters}, method='L-BFGS-B', jac=True)

result = result['x']


theta1 = np.reshape(result[:hidden_units*(input_layer_size+1)], (hidden_units, input_layer_size+1), order='C')
theta2 = np.reshape(result[hidden_units*(input_layer_size+1):], (labels, hidden_units+1), order='C')

theta1_file = "D:\\ML\\Hanwriting ML\\data\\theta1.txt"
theta2_file = "D:\\ML\\Hanwriting ML\\data\\theta2.txt"

np.savetxt(theta1_file, theta1, delimiter=' ')
np.savetxt(theta2_file, theta2, delimiter=' ')
