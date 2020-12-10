# Justin Ventura & Blaine Mason COSC311
# Neural Network for the Titanic Dataset

"""
Project Titanic Neural Network for classifying whether
or not a give person with a specific set of featues had
survived the titanic sinking or not.
"""

import math
import numpy as np
from typing import Iterable


# -----------------------------------------------------------------------------


# Titanic Neural Network Vector Class.
class TNN_Vector:

    # Constructor
    def __init__(self, data: Iterable = None) -> None:
        """ Constructor for TNN_Vector Class.

        Args:
            data (Iterable, optional): Data for vector. Defaults to None.

        Returns:
            None.
        """
        self.data = np.array(data) if data is not None else None

    # Static method for dot product.
    @staticmethod
    def dot(u, v) -> float:
        """ Dot product.

        Args:
            u (TNN_Vector): vect 1.
            v (TNN_Vector): vect 2.

        Returns:
            float: dot product result.
        """
        return sum([u_i * v_i for u_i, v_i in zip(u.data, v.data)])


# -----------------------------------------------------------------------------


# Neural Network Class
class Tneural_network:
    """ Neural Network class for the titanic dataset. """

    # Constructor.
    def __init__(self, input_len: int, layer_len: int, layer_cnt: int) -> None:
        # These are the layers of the neural network.
        self.temp = layer_cnt

        self.w_in = [[np.random.random() for _ in range(input_len + 1)]
                     for _ in range(4)]  # 4 for now
        self.w_out = [[np.random.random() for _ in range(4)]
                      for _ in range(2)]

        self.network = [self.w_in, self.w_out]

    # Training method:
    def train_network(self, vectors) -> None:
        # TODO: Feed forward; backward propagation.
        pass

    # Feeding helping method.
    def feed_forward(self, input_vector):
        """ Forward process of the neural network, no backwards propagation.

        Args:
            input_vector (vector of data): Input vector for the network.

        Returns:
            [np.array]: The outputs of each layer.  To get the result of
            the neural network, get index -1.
        """
        outputs = []

        # Iterate layers of the neural network.
        for layer in self.network:
            input_with_bias = input_vector + [1]
            output = [Tneural_network.neuron_output(neuron, input_with_bias)
                      for neuron in layer]

            outputs.append(output)

        # Returns the output of each layer.
        return outputs

    # Function to print the weights [NOTE: DEBUGS]
    def get_weights(self):
        return self.w_in, self.w_out

    # Static sigmoid function.
    @staticmethod
    def sigmoid(t: float) -> float:
        return 1 / (1 + math.exp(-t))

    # Static argmax function.
    @staticmethod
    def argmax(xs):
        return max(range(len(xs)), key=lambda i: xs[i])

    # Static neuron output function.
    @staticmethod
    def neuron_output(weights, inputs) -> float:
        return Tneural_network.sigmoid(np.dot(weights, inputs))


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    print('-Justin Ventura λ')
    print(f'{np.pi}')
