import numpy as np


def neural_network(theta, X, y, Lambda, input_layer_size, hidden_units, labels):
    
    # Extracting Thetas
    theta1 = np.reshape(theta[:hidden_units*(input_layer_size+1)], (hidden_units, input_layer_size+1))
    theta2 = np.reshape(theta[hidden_units*(input_layer_size+1):], (labels, hidden_units+1))


    # Feedforward
    m = X.shape[0]

    one = np.ones((m, 1))
    act1_bias = np.c_[one, X]

    z2 = np.dot(act1_bias, np.transpose(theta1))
    act2 = sigmoid(z2)
    act2_bias = np.c_[one, act2]

    z3 = np.dot(act2_bias, np.transpose(theta2))
    act3 = sigmoid(z3)


    # Converting y into binary matrix
    y_new = np.zeros((m, labels))
    for i in range(m):
        y_new[i, int(y[i])] = 1


    # Checking Cost Function
    h = act3
    J = cost_function(h, y_new, theta1, theta2, Lambda)

    print(f"Cost{J}")
    # Back propagation
    delta_3 = h - y_new
    delta_2 = np.dot(delta_3, theta2[:, 1:]) * sigmoid_gradient(act2)

    theta1[:, 0] = 0
    r1 = (Lambda/m) * theta1

    theta2[:, 0] = 0
    r2 = (Lambda/m) * theta2

    theta1_grad = (1/m) * np.dot(delta_2.transpose(), act1_bias) + r1
    theta2_grad = (1/m) * np.dot(delta_3.transpose(), act2_bias) + r2

    theta_grad = np.r_[theta1_grad.flatten(), theta2_grad.flatten()]

    return J, theta_grad


def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g


def sigmoid_gradient(z):
    g = z * (1 - z)
    return g


def cost_function(h, y_new, theta1, theta2, Lambda):
    m = h.shape[0]

    J = (1/m) * sum(sum(-y_new* np.log(h) - (1 - y_new)*np.log(1 - h)))

    
    reg = (Lambda/(2*m)) * (sum(sum(pow(theta1[:, 1:], 2))) + sum(sum(pow(theta2[:, 1:], 2))))

    J_reg = J + reg

    return J_reg


def predict(X, theta1, theta2):
    
    # Feedforward
    m = X.shape[0]

    one = np.ones((m, 1))
    act1_bias = np.c_[one, X]

    z2 = np.dot(act1_bias, np.transpose(theta1))
    act2 = sigmoid(z2)
    act2_bias = np.c_[one, act2]

    z3 = np.dot(act2_bias, np.transpose(theta2))
    act3 = sigmoid(z3)

    prediction = np.argmax(act3, axis=1)

    return prediction