# Justin Ventura & Blaine Mason COSC311
# Neural Network for the Titanic Dataset

"""
Project Titanic Neural Network for classifying whether
or not a give person with a specific set of featues had
survived the titanic sinking or not.
"""

import math
import numpy as np
from typing import Iterable, List


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
    def __init__(self, input_len: int, layer_shape: List[int] = None) -> None:
        # These are the layers of the neural network.
        # self.layer_count = len(layer_shape)

        self.network = [
            [
                [np.random.random() for _ in range(input_len + 1)]
                for _ in range(2)
            ],
            [
                [np.random.random() for _ in range(input_len)]
                for _ in range(2)
            ],
            [
                [np.random.random() for _ in range(2 + 1)]
                for _ in range(2)
            ]
        ]

    # Training method:
    def train_network(self, input_vector, target_vector) -> None:
        # TODO: Feed forward; backward propagation.
        gamma = 1.0
        gradients = self.sqerror_gradients(input_vector, target_vector)

        self.network = [
            [self.gradient_step(neuron, grad, -gamma)
             for neuron, grad in zip(layer, layer_grad)]
            for layer, layer_grad in zip(self.network, gradients)
        ]

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
            input_vector = output

        # Returns the output of each layer.
        return outputs[:-1], outputs[-1]

    # New backprop
    def backward_propagation(self, input_vector, target_vector):
        pass

    # Gradients of the squared error loss:
    def sqerror_gradients(self, in_vect, targ_vect):
        hidden_outputs, outputs = self.feed_forward(in_vect)

        print(hidden_outputs)
        print(outputs)

        out_deltas = [output * (1 - output) * (output - target)
                      for output, target in zip(in_vect, targ_vect)]

        # output_grads = [[out_deltas[i] * hidden_output
        #                  for hidden_output in hidden_outputs + [1]]
        #                 for i, output_neuron in enumerate(self.network[-1])]
        output_grads = [[np.dot(out_deltas[i], hidden_output)
                         for hidden_output in hidden_outputs + [1]]
                        for i, output_neuron in enumerate(self.network[-1])]

        # hidden_deltas = [hidden_output * (1 - hidden_output) *
        #                  np.dot(out_deltas, [n[i] for n in self.network[-1]])
        #                  for i, hidden_output in enumerate(hidden_outputs)]

        hidden_deltas = [hidden_output * (1 - hidden_output) *
                         np.dot(out_deltas, [n[i] for n in self.network[-1]])
                         for i, hidden_output in enumerate(hidden_outputs)]

        hidden_grads = [[hidden_deltas[i] * input for input in in_vect + [1]]
                        for i, hidden_neuron in enumerate(self.network[0])]

        return [hidden_grads, output_grads]

    # Function to print the weights [NOTE: DEBUGS]
    def get_layers(self):
        return self.network

    # Calculate the loss of the result vector.
    @staticmethod
    def squared_distance(output_vector, target_vect):
        return sum([(o_i - t_i)**2
                    for o_i, t_i in zip(output_vector, target_vect)])

    # Gradient step method.
    @staticmethod
    def gradient_step(vect, gradient, step_size):
        assert len(vect) == len(gradient)
        step = [grad_i * step_size for grad_i in gradient]
        return [v_i + s_i for v_i, s_i in zip(vect, step)]

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
