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
        

    def feed_forward(self, input):
        # Check length of input is correct
        if len(self.layers[0]) != len(input):
            raise Exception

        # Copy input across to input layer
        for i in range(len(self.layers[0])):
            self.layers[0][i] = input[i]

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
        return self.layers[-1]


    def reset_layers(self):
        for layer in self.layers:
            for i in range(len(layer)):
                layer[i] = 0

    def get_output(self):
        return self.layers[-1]

    def back_propogate_deep_q_learning(self, target_q, target_a):
        # Delta_L = Grad(CostFn)(target) * OutputActivationFunctionDerivative, which in our case is: 
        error = self.layers[i][target_a] - target_q # * 1 since output activation function f(x) = x



if __name__ == '__main__':
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    dx = lambda x: sigmoid(x)(1 - sigmoid(x))
    nn = NeuralNetwork((3, 2, 1), (sigmoid, sigmoid), (dx, dx))
    nn.set_weights([np.array([[-1, 0.5, 1], [-2, 1, -1]]), np.array([[2,1]])])
    nn.set_biases([np.array([1, 2]), np.array([-1])])
    print(nn.feed_forward([-1, 1, -1]))
