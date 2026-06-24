import numpy as np

class FeedForwardNN:
    """
    A Feed-Forward Neural Network implemented from scratch using NumPy.
    """
    def __init__(self, layer_sizes, hidden_activation='relu', output_activation='sigmoid', loss_type='bce', learning_rate=0.01):
        """
        Instance Variables:
            self.layer_sizes (list of int): Stored layer dimensions.
            self.hidden_act (str):  hidden activation type. ('relu' or 'sigmoid')
            self.output_act (str):  output activation type. ('sigmoid' or 'linear')
            self.loss_type (str):  loss function type. ('mse' or 'bce')
            self.learning_rate (float):  learning rate alpha.
            self.L (int): Total number of layers (number of weight matrices).
            self.weights (list of numpy.ndarray): Weight matrices. weights[l] = W for layer l.
            self.biases (list of numpy.ndarray): Bias vectors. biases[l] = b for layer l.
            self.pre_activations (list of numpy.ndarray): Pre-activations z stored during forward pass.
            self.activations (list of numpy.ndarray): Activations a stored during forward pass.
                                                      activations[0] = X (network input).
        """
        self.layer_sizes = layer_sizes
        self.hidden_act = hidden_activation
        self.output_act = output_activation
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.L = len(layer_sizes) - 1

        self.weights = []
        self.biases = []
        self.pre_activations = []
        self.activations = []

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        TODO 1: Initialize weights and biases for each layer and append them to
        self.weights and self.biases.
        """
        np.random.seed(42)
        # --- YOUR CODE HERE ---
        for layer_index in range(self.L):
            input_size = self.layer_sizes[layer_index]
            output_size = self.layer_sizes[layer_index+1]
            W = np.random.randn(input_size, output_size) * 0.01
            b = np.zeros((1, output_size))
            self.weights.append(W)
            self.biases.append(b)
        # ----------------------

    def relu(self, z):
        """
        TODO 2a: Implement the ReLU activation.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Activated values.
        """
        # --- YOUR CODE HERE ---
        result = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i][j] > 0:
                    result[i][j] = z[i][j]
                else:
                    result[i][j] = 0
        return result
        # ----------------------

    def relu_derivative(self, z):
        """
        TODO 2b: Implement the derivative of ReLU.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Element-wise derivative values.
        """
        # --- YOUR CODE HERE ---
        result = np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                if z[i][j] > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        return result
        # ----------------------

    def sigmoid(self, z):
        """
        TODO 3a: Implement the Sigmoid activation.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Activated values.
        """
        # --- YOUR CODE HERE ---
        result = 1 / (1 + np.exp(-z))
        return result
        # ----------------------

    def sigmoid_derivative(self, z):
        """
        TODO 3b: Implement the derivative of Sigmoid.

        Args:
            z (numpy.ndarray): Pre-activation values.

        Returns:
            numpy.ndarray: Element-wise derivative values.
        """
        # --- YOUR CODE HERE ---
        sig_val = self.sigmoid(z)
        result = sig_val * (1 - sig_val)
        return result
        # ----------------------

    def activate(self, z, activation_type):
        """
        Routes to the correct activation function.

        Args:
            z (numpy.ndarray): Pre-activation values.
            activation_type (str): One of 'relu', 'sigmoid', or 'linear'.

        Returns:
            numpy.ndarray: Activated values.
        """
        if activation_type == 'relu': return self.relu(z)
        elif activation_type == 'sigmoid': return self.sigmoid(z)
        elif activation_type == 'linear': return z
        else: raise ValueError("Unsupported activation")

    def forward_propagation(self, X):
        """
        TODO 4: Implement forward propagation.

        Args:
            X (numpy.ndarray): Input data of shape (num_samples, num_features).

        Returns:
            numpy.ndarray: Final output predictions.
        """
        self.activations = [X]
        self.pre_activations = []

        # --- YOUR CODE HERE ---
        current_activation = X
        for layer_index in range(self.L):
            W = self.weights[layer_index]
            b = self.biases[layer_index]
            z = np.dot(current_activation, W) + b
            self.pre_activations.append(z)
            if layer_index == self.L-1:
                current_activation = self.activate(z, self.output_act)
            else:
                current_activation = self.activate(z, self.hidden_act)
            self.activations.append(current_activation)
        return current_activation
        # ----------------------

    def compute_loss(self, output, y):
        """
        TODO 5: Compute and return the loss (MSE or BCE).

        Args:
            output (numpy.ndarray): Predictions from the forward pass.
            y (numpy.ndarray): True labels/targets.

        Returns:
            float: Scalar loss value.
        """
        # --- YOUR CODE HERE ---
        num_samples = y.shape[0]
        loss_value = 0
        if self.loss_type == 'mse':
            total_sq_error = 0
            for i in range(num_samples):
                diff = output[i] - y[i]
                total_sq_error = total_sq_error + np.sum(diff*diff)
            loss_value = total_sq_error / num_samples
        else:
            epsilon_val = 1e-8
            total_bce = 0
            for i in range(num_samples):
                bce_i = y[i]*np.log(output[i]+epsilon_val) + (1-y[i])*np.log(1-output[i]+epsilon_val)
                total_bce = total_bce + np.sum(bce_i)
            loss_value = -total_bce / num_samples
        return loss_value
        # ----------------------

    def backward_propagation(self, y):
        """
        TODO 6: Implement backpropagation.

        Args:
            y (numpy.ndarray): True labels/targets.

        Returns:
            tuple: (grad_weights, grad_biases) where each is a list of gradients,
                   one per layer.
        """
        grad_weights = [np.zeros_like(w) for w in self.weights]
        grad_biases  = [np.zeros_like(b) for b in self.biases]

        # --- YOUR CODE HERE ---
        num_samples = y.shape[0]
        output = self.activations[self.L]
        if self.loss_type == 'mse':
            delta = (output - y) / num_samples
        else:
            epsilon_val = 1e-8
            delta = -(y/(output+epsilon_val) - (1-y)/(1-output+epsilon_val)) / num_samples
        if self.output_act == 'sigmoid':
            delta = delta * self.sigmoid_derivative(self.pre_activations[self.L-1])
        elif self.output_act == 'relu':
            delta = delta * self.relu_derivative(self.pre_activations[self.L-1])
        layer_index = self.L - 1
        grad_weights[layer_index] = np.dot(self.activations[layer_index].T, delta)
        grad_biases[layer_index] = np.sum(delta, axis=0, keepdims=True)
        current_delta = delta
        back_index = self.L - 2
        while back_index >= 0:
            current_delta = np.dot(current_delta, self.weights[back_index+1].T)
            if self.hidden_act == 'relu':
                current_delta = current_delta * self.relu_derivative(self.pre_activations[back_index])
            else:
                current_delta = current_delta * self.sigmoid_derivative(self.pre_activations[back_index])
            grad_weights[back_index] = np.dot(self.activations[back_index].T, current_delta)
            grad_biases[back_index] = np.sum(current_delta, axis=0, keepdims=True)
            back_index = back_index - 1
        return grad_weights, grad_biases
        # ----------------------

    def update_parameters(self, grad_weights, grad_biases):
        """
        TODO 7: Update weights and biases using gradient descent.

        Args:
            grad_weights (list of numpy.ndarray): Weight gradients per layer.
            grad_biases  (list of numpy.ndarray): Bias gradients per layer.
        """
        # --- YOUR CODE HERE ---
        for layer_index in range(self.L):
            self.weights[layer_index] = self.weights[layer_index] - self.learning_rate*grad_weights[layer_index]
            self.biases[layer_index] = self.biases[layer_index] - self.learning_rate*grad_biases[layer_index]
        # ----------------------


    def train(self, X, y, epochs=10000, print_freq=1000):
        """
        TODO 8: Implement the training loop.

        Args:
            X (numpy.ndarray): Input data.
            y (numpy.ndarray): True labels/targets.
            epochs (int): Number of training iterations.
            print_freq (int): Frequency of printing loss.
        """
        # --- YOUR CODE HERE ---
        epoch_counter = 0
        while epoch_counter < epochs:
            output = self.forward_propagation(X)
            loss_val = self.compute_loss(output, y)
            grad_weights, grad_biases = self.backward_propagation(y)
            self.update_parameters(grad_weights, grad_biases)
            is_print_time = False
            if epoch_counter % print_freq == 0:
                is_print_time = True
            if is_print_time:
                print("Epoch", epoch_counter, "Loss:", loss_val)
            epoch_counter = epoch_counter + 1
        # ----------------------

    def predict(self, X, threshold=0.5):
        """
        TODO 9: Implement the prediction function.

        Args:
            X (numpy.ndarray): Input data.
            threshold (float): Threshold for binary classification.

        Returns:
            numpy.ndarray: Predicted labels or regression outputs.
        """
        # --- YOUR CODE HERE ---
        output = self.forward_propagation(X)
        num_samples = output.shape[0]
        if self.output_act == 'linear':
            return output
        pred_labels = np.zeros(num_samples)
        for i in range(num_samples):
            if output[i] >= threshold:
                pred_labels[i] = 1
            else:
                pred_labels[i] = 0
        return pred_labels
        # ----------------------