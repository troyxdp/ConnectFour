import numpy as np

class NeuralNetwork():
    def __init__(self, num_neurons, activation_functions, activation_function_derivatives):
        # Check for invalid initialization parameters
        if (len(num_neurons) - 1) != len(activation_functions):
            raise Exception
        if not len(activation_functions) == len(activation_function_derivatives):
            raise Exception

        # z values
        self.z_values = []

        # initialize layers    
        self.layers = []
        for i in num_neurons:
            self.layers.append(np.zeros(i))
            self.z_values.append(np.zeros(i))        

        # initialize weights
        self.weights = []
        for i in range(1, len(num_neurons)):
            self.weights.append(np.random.rand(num_neurons[i], num_neurons[i-1]))

        self.biases = []
        for i in range(1, len(num_neurons)):
            self.biases.append(np.random.rand(num_neurons[i]))

        # Initialize activation functions and their derivatives
        self.activation_functions = activation_functions
        self.activation_function_derivatives = activation_function_derivatives


    def set_weights(self, weight_matrices):
        # Check number of weight matrices is correct
        if len(weight_matrices) != len(self.weights):
            raise Exception
        # Check dimensions of weight matrices are correct
        for i in range(len(weight_matrices)):
            if not weight_matrices[i].shape == self.weights[i].shape:
                raise Exception

        # Copy across values
        for i in range(len(weight_matrices)):
            self.weights[i] = weight_matrices[i].copy()


    def set_biases(self, bias_vectors):
        # Check number of bias vectors is correct
        if len(bias_vectors) != len(self.biases):
            raise Exception
        # Check dimensions of bias vectors
        for i in range(len(self.biases)):
            if not len(self.biases[i]) == len(bias_vectors[i]):
                raise Exception
        
        # Copy values across
        for i in range(len(self.biases)):
            self.biases[i] = bias_vectors[i].copy()
        

    def feed_forward(self, nn_input):
        # Check length of input is correct
        if len(self.layers[0]) != len(nn_input):
            raise Exception

        # Copy input across to input layer
        for i in range(len(self.layers[0])):
            self.layers[0][i] = nn_input[i]

        # Feed forward through the layers
        self.layers[0] = np.array(self.layers[0])
        for i in range(1, len(self.layers)):
            layer = self.layers[i-1]
            weight_values = self.weights[i-1]
            bias_values = self.biases[i-1]
            z = np.dot(weight_values, layer)
            for j in range(len(self.z_values[i])):
                self.z_values[i][j] = z[j] + bias_values[j]
            self.layers[i] = self.activation_functions[i-1](self.z_values[i].copy())

        # Return output
        return np.array(self.layers[-1])


    def reset_layers(self):
        for layer in self.layers:
            for i in range(len(layer)):
                layer[i] = 0

    def get_output(self):
        return self.layers[-1]

    # Adapted this code from http://neuralnetworksanddeeplearning.com/chap2.html
    def back_propogate(self, error):
        # Get backpropogation values for each weights matrix and bias vector
        delta_w = [np.zeros(weight_matrix.shape) for weight_matrix in self.weights]
        delta_b = [np.zeros(bias_vector.shape) for bias_vector in self.biases]

        # Get initial backpropogation values
        delta_b[-1] = error.copy()
        delta_w[-1] = np.outer(error, self.layers[-2])

        # Backpropogate throughout layers
        delta = error
        for l in range(2, len(self.layers)):
            act_fn_dx = self.activation_function_derivatives[-l](self.z_values[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * act_fn_dx
            delta_b[-l] = delta
            delta_w[-l] = np.outer(delta, self.layers[-l-1])
        return delta_b, delta_w

    def update_network(self, lr, error):
        # Get error values
        delta_b, delta_w = self.back_propogate(error)

        # Update weights
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= lr * delta_w[i][j][k]
                self.biases[i][j] -= lr * delta_b[i][j]

    def __str__(self):
        to_ret = ''
        for i, weight_matrix in enumerate(self.weights):
            to_ret += f'Layer {i}:\n'
            to_ret += f'Weights:\n'
            to_ret += str(weight_matrix) + '\n'
            to_ret += 'Biases:\n'
            to_ret += str(self.biases[i])
            to_ret += '\n\n'
        return to_ret[:-2]



if __name__ == '__main__':
    # Activation functions
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    sigmoid_dx = lambda x: sigmoid(x) * (1 - sigmoid(x))
    def relu(x):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(val)
            else:
                to_ret.append(0)
        return np.array(to_ret)
    def relu_dx(x):
        to_ret = []
        for val in x:
            if val > 0:
                to_ret.append(1)
            else:
                to_ret.append(0)
        return np.array(to_ret)



    # Create a first neural network
    print("==================================================================================")
    print("NETWORK 1:")
    nn = NeuralNetwork((2, 2, 2), (sigmoid, sigmoid), (sigmoid_dx, sigmoid_dx))
    nn.set_weights([np.array([[0.15, 0.2], [0.25, 0.3]]), np.array([[0.4, 0.45], [0.5, 0.55]])])
    nn.set_biases([np.array([0.35, 0.35]), np.array([0.6, 0.6])])

    # Feedforward
    print("Network Output:")
    print(output:=nn.feed_forward([0.05, 0.1]))

    # Update network using backpropogation
    target = np.array([0.01, 0.99])
    error = (output - target) * sigmoid_dx(nn.z_values[-1])
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    # Create a second neural network
    print("\n\n\n==================================================================================")
    print("NETWORK 2:")
    nn = NeuralNetwork((2, 2, 1), (sigmoid, sigmoid), (sigmoid_dx, sigmoid_dx))
    nn.set_weights([np.array([[0.15, 0.2], [0.25, 0.3]]), np.array([[0.4, 0.45]])])
    nn.set_biases([np.array([0.35, 0.35]), np.array([0.6])])

    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward([0.05, 0.1]))

    # Test backpropogation
    target = np.array([0.01])
    error = (output - target) * sigmoid_dx(nn.z_values[-1])
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    # Create a third neural network
    print("\n\n\n==================================================================================")
    print("NETWORK 3:")
    nn = NeuralNetwork((3, 4, 2), (sigmoid, sigmoid), (sigmoid_dx, sigmoid_dx))
    nn.set_weights([np.array([
                              [0.2,  0.4,  0.1],
                              [0.5,  0.3,  0.7],
                              [0.6,  0.9,  0.2],
                              [0.8,  0.1,  0.5]
                             ]), 
                    np.array([
                              [0.3,  0.7, 0.5, 0.9],
                              [0.8,  0.2, 0.6, 0.4]
                             ]),
                   ])
    nn.set_biases([np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.2, 0.5])])
    
    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward(np.array([0.1, 0.5, 0.9])))

    # Test backpropogation
    target = np.array([0.0, 1.0])
    error = (output - target) * sigmoid_dx(nn.z_values[-1])
    print("\nError of output:")
    print(error)
    nn.update_network(0.5, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")



    print("\n\n\n==================================================================================")
    print("NETWORK 4:")
    nn = NeuralNetwork((3, 4, 2), (relu, sigmoid), (relu_dx, sigmoid_dx))
    nn.set_weights([np.array([
                              [0.2,  0.4,  0.1],
                              [0.5,  0.3,  0.7],
                              [0.6,  0.9,  0.2],
                              [0.8,  0.1,  0.5]
                             ]), 
                    np.array([
                              [0.3,  0.7, 0.5, 0.9],
                              [0.8,  0.2, 0.6, 0.4]
                             ]),
                   ])
    nn.set_biases([np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.15, 0.5])])
    
    # Test feedforward
    print("Network Output:")
    print(output:=nn.feed_forward(np.array([0.9, 0.1, 0.8])))

    # Test backpropogation
    target = np.array([0.2, 0.6])
    error = (output - target) * relu_dx(nn.z_values[-1])
    print("\nError of output:")
    print(error)
    nn.update_network(0.1, error)
    print("\nFinal network:")
    print(nn)
    print("==================================================================================")


    # Get error
    # Delta_L = Grad(CostFn)(target) * OutputActivationFunctionDerivative, which in our case is: 
    # error = np.zeros(len(self.layers[-1]))
    # error[target_a] = (self.layers[-1][target_a] - target_q)
    # error = error * self.activation_function_derivatives[-1](self.z_values[-1]) # * 1 since output activation function f(x) = x