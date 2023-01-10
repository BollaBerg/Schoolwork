# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os


def sigmoid(x : float):
    """Activation function"""
    denominator = 1 + np.exp(-x)
    return 1 / denominator

def sigmoid_derivative(x : float):
    """Derivative of activation function"""
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


class NeuralNetworkLayer:
    def __init__(self, num_nodes : int, num_prev_layer_nodes : int):
        self.num_nodes = num_nodes
        self.prev_layer_nodes = num_prev_layer_nodes
        # self.weights contains all weights for the layer
        # Usage:
        #   self.weights[node_prev_layer][node_this_layer]
        # Example:
        #   Accessing the weight from node 2 to node 5:
        #   self.weights[2][5]
        # The bias is in self.weights[-1]
        self.weights = np.random.random_sample(
            (num_prev_layer_nodes + 1, num_nodes)
            ) - 0.5     # -0.5 to get initial weights in range [-0.5, 0.5]

    def __len__(self):
        return self.num_nodes

    def compute(self, inputs : np.ndarray) -> np.ndarray:
        # Assert that inputs has the right length
        # Using len(self.weights) - 1 due to the bias weight
        assert len(inputs) == (len(self.weights) - 1)
        
        inputs_with_bias = np.append(inputs, -1)

        non_normalized_output = np.empty(self.num_nodes)
        for this_node in range(self.num_nodes):
            node_value = np.dot(
                # Must transpose self.weights to get all inputs to this_node
                # in one array
                self.weights.transpose()[this_node]
                , inputs_with_bias
            )

            # Add to non_normalized_output
            non_normalized_output[this_node] = node_value
        
        output = sigmoid(non_normalized_output)
        return output, non_normalized_output



class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # --- PLEASE READ --
        # Use the parameters below to train your feed-forward neural network.

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        # See 18.6.1 in AIMA 3rd edition (page 719).
        # This is the value of α on Line 25 in Figure 18.24.
        self.lr = 1e-3

        # Line 6 in Figure 18.24 says "repeat".
        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # Initialize layers as a list of layers
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.layers = []
        if hidden_layer:
            self.layers.append(NeuralNetworkLayer(self.hidden_units, input_dim))
            self.layers.append(NeuralNetworkLayer(1, self.hidden_units))
        else:
            self.layers.append(NeuralNetworkLayer(1, input_dim))

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network
        
        Based on the back-propagation algorithm outlined in Figure 18.24
        (page 734) in AIMA 3rd edition.
        
        Some stuff from the algorithm:
            examples = self.x_train, self.y_train
            alpha = self.lr
        """
        if self.hidden_layer: self.train_network()
        else: self.train_perceptron()
    
    def train_network(self) -> None:
        """ Train the network if it has a hidden layer

        This repeats a lot of code from train_perceptron, and has a lot of
        repeating, hard-coded-for-only-one-hidden-layer code.

        TODO:
            Combine with train_perceptron
            Improve speed of execution (current time: approx. 200s to run tests)
            Soften up hard-coding so it can handle arbitrary number of hidden layers
        """
        for _ in range(self.epochs):
            for example, example_result in zip(self.x_train, self.y_train):
                
                # We have two layers to deal with:
                input_layer_output, input_layer_activation = self.layers[0].compute(example)
                hidden_layer_output, _ = self.layers[1].compute(input_layer_output)

                delta_result = (sigmoid_derivative(hidden_layer_output[0])
                    * (example_result - hidden_layer_output[0])
                )

                hidden_layer_deltas = sigmoid_derivative(input_layer_output)
                for hidden_node in range(len(input_layer_output)):
                    hidden_layer_deltas[hidden_node] *= (
                        self.layers[1].weights[hidden_node] * delta_result
                    )

                # Update network weights:
                # Input to hidden layer
                for input_node in range(len(self.layers[0].weights)-1):
                    for hidden_node in range(len(self.layers[0].weights[0])):
                        self.layers[0].weights[input_node][hidden_node] = (
                            self.layers[0].weights[input_node][hidden_node]
                            + self.lr
                            * example[input_node]
                            * hidden_layer_deltas[hidden_node]
                        )

                # Calculate weight for bias "manually"
                for hidden_node in range(len(self.layers[0].weights[-1])):
                    self.layers[0].weights[-1][hidden_node] = (
                        self.layers[0].weights[-1][hidden_node]
                        + self.lr
                        * (-1)
                        * hidden_layer_deltas[hidden_node]
                    )

                # Update network weights:
                # Hidden layer to output
                for hidden_node in range(len(self.layers[1].weights)-1):
                    self.layers[1].weights[hidden_node] = (
                        self.layers[1].weights[hidden_node]
                        + self.lr
                        * input_layer_activation[hidden_node]
                        * delta_result
                    )

                # Calculate weight for bias "manually"
                self.layers[1].weights[-1] = (
                    self.layers[1].weights[-1]
                    + self.lr
                    * (-1)
                    * delta_result
                )


    def train_perceptron(self) -> None:
        """ Train the network if it has a hidden layer

        This repeats a lot of code from train_network.

        TODO:
            Combine with train_network
            Improve speed of execution (current time: approx. 200s to run tests)
        """
        for _ in range(self.epochs):
            for example, example_result in zip(self.x_train, self.y_train):

                # We only have one layer to deal with
                layer = self.layers[0]
                network_output, _ = layer.compute(example)

                delta_result = (sigmoid_derivative(network_output[0])
                    * (example_result - network_output[0])
                )

                for input_node in range(len(self.layers[0].weights)-1):
                    self.layers[0].weights[input_node] = (
                        self.layers[0].weights[input_node]
                        + self.lr
                        * example[input_node]
                        * delta_result
                    )
                # Calculate weight for bias "manually"
                self.layers[0].weights[-1] = (
                    self.layers[0].weights[-1]
                    + self.lr
                    * (-1)
                    * delta_result
                )
        


    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """
        # Start with "result" from input layer (aka inputs):
        layer_result = x

        # Iteratively compute layer outputs, based on previous layer's outputs
        for layer in self.layers:
            layer_result, _ = layer.compute(layer_result)

        # After last layer has been traversed - layer_result has len == 1
        return layer_result[0]


class TestSubpartsOwnInitiative(unittest.TestCase):
    def test_sigmoid(self) -> None:
        """Test sigmoid works correctly"""
        self.assertAlmostEqual(sigmoid(0.5), 0.62246, places=5)
        self.assertAlmostEqual(sigmoid(1), 0.73106, places=5)

    def test_sigmoid_derivative(self) -> None:
        """Test sigmoid_derivative works correctly"""
        self.assertAlmostEqual(sigmoid_derivative(0.5), 0.23500, places=5)
        self.assertAlmostEqual(sigmoid_derivative(1), 0.19661, places=5)

    def test_sigmoid_on_ndarray(self) -> None:
        """Test sigmoid function on numpy.ndarray"""
        result = sigmoid(np.array([0, 0.5, 1]))
        expected_result = np.array([0.5, 0.62246, 0.73106])
        np.testing.assert_allclose(result, expected_result, rtol=1e-4)

    def test_sigmoid_derivative_on_ndarray(self) -> None:
        """Test sigmoid_derivative function on numpy.ndarray"""
        result = sigmoid_derivative(np.array([0, 0.5, 1]))
        expected_result = np.array([0.25, 0.23500, 0.19661])
        np.testing.assert_allclose(result, expected_result, rtol=1e-4)

    def test_layer_compute(self) -> None:
        """Test that a layer computes correctly with known values"""
        layer = NeuralNetworkLayer(1, 2)
        layer.weights[0][0] = 0.5
        layer.weights[1][0] = 1
        layer.weights[2][0] = -0.5

        result, _ = layer.compute(np.array([1, -1]))
        expected_result = np.array([
            sigmoid(1 * 0.5 + (-1) * 1 + (-1) * (-0.5))
        ])
        self.assertEqual(result, expected_result)

    def test_network_predict(self) -> None:
        """Test that a network predicts correctly with known values"""
        network = NeuralNetwork(2, False)
        network.layers[0].weights[0][0] = 0.5
        network.layers[0].weights[1][0] = 1
        network.layers[0].weights[2][0] = -0.5

        result = network.predict(np.array([1, -1]))
        expected_result = sigmoid(1 * 0.5 + (-1) * 1 + (-1) * (-0.5))
        self.assertEqual(result, expected_result)


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
