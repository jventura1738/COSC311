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
    def __init__(self, dim: int, num_layers: int, layer_size: int) -> None:
        """Constuctor for the Neural Network Class.

        Args:
            dim (int): Dimension of the input vector.
            num_layers (int): Number of hidden layers.
            layer_size (int): Nodes per layer (static).

        Titanic Model will have the following structure:
        3 -> 2 -> 2 -> 1

        so we need:

        w_in  = [2 x 4]
        w_h   = [2 x 2]
        w_out = [1 x 2]

        """
        # This is for the bias.
        dim += 1

        # These are the layers of the neural network.
        # Input weights. Row i is the array of weights applied to x_i.
        # self.w_in = np.random.standard_normal((dim, layer_size))
        self.w_in = np.random.standard_normal((layer_size, dim))

        # "Tensor" (3-dim array) of hidden-layer output weights.
        # w_hidden[lay][i][j] is the weight between lay node i and lay+1 node j
        self.w_h = np.random.standard_normal((num_layers-1,
                                                   layer_size, layer_size))

        # output weights, comes from last layer
        self.w_out = np.random.standard_normal((1, layer_size))

    # Training method:
    def train_network(self, input_vector, target_vector) -> None:
        # TODO: Feed forward; backward propagation.
        # gamma = 1.0
        # gradients = self.sqerror_gradients(input_vector, target_vector)

        # self.network = [
        #     [self.gradient_step(neuron, grad, -gamma)
        #      for neuron, grad in zip(layer, layer_grad)]
        #     for layer, layer_grad in zip(self.network, gradients)
        # ]
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
        # outputs = []

        # # Iterate layers of the neural network.
        # for layer in self.network:
        #     input_with_bias = input_vector + [1]
        #     output = [Tneural_network.neuron_output(neuron, input_with_bias)
        #               for neuron in layer]

        #     outputs.append(output)
        #     input_vector = output

        # # Returns the output of each layer.
        # return outputs[-1], outputs
        pass

    # New backprop
    def backward_propagation(self, input_vector, pred, label):
        pass
        # dEyo = pred - label  # scalar
        # dExo = dEyo * sigmoid_deriv(np.dot(outputs[-1], outw[0])) # scalar
        # dEwo =  dExo * outputs[-1]  #np.zeros((1, layer_size)) # out

        # # hidden layer derivatives setup
        # dEwh = np.zeros((num_layers-1, layer_size, layer_size))
        # dExh = np.zeros((num_layers, layer_size))
        # dEyh = np.zeros((num_layers, layer_size))

        # # need to do output layer first, not a matrix product
        # dEyh[-1] = outw * dExo # 1-by-h times scalar

        # for i in range(num_layers-2,-1,-1):
        #     # i-1 to get the inputs to layer i
        #     x = outputs[i-1] @ hiddenw[i-1] # 1-by-h times h-by-h
        #     dExh[i] = dEyh[i] * sigmoid_deriv(x) # 1-by-h
        #     dEwh[i] = outputs[i-1] * dExh[i]
        #     if i > 0:
        #         # prep the next layer
        #         dEyh[i-1] = hiddenw[i] @ dExh[i].T # h-by-h times h-by-1

        # #dEwi = outputs[0] * dEyh[0] # take care of the input layer, again
        #                             # not a matrix product
        # data = numpy.array([data])
        # dEwi = np.matlib.repmat(data.T, 1, layer_size) *
        # np.matlib.repmat(dExh[0], dim, 1)
        # # dim-by-h broadcast dim-by-h

        # # adjust the hiden layer weights accoriding to the error.
        # # Check to see that this follows gradient descent!
        # hiddenw = hiddenw - rate * dEwh
        # inw = inw - rate * dEwi
        # outw[0] = outw[0] - rate * dEwo

        # # return the new weights
        # return inw, outw, hiddenw
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

    @staticmethod
    def sigmoid_derivative(x: float) -> float:
        return Tneural_network.sigmoid(x) * (1 - Tneural_network.sigmoid(x))

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
