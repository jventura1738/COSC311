# Justin Ventura & Blaine Mason COSC311
# Neural Network for the Titanic Dataset

"""
Project Titanic Neural Network for classifying whether
or not a give person with a specific set of featues had
survived the titanic sinking or not.
"""

import math
import numpy as np
import numpy.matlib
# import matplotlib.pyplot as plt
# from matplotlib import cm


# -----------------------------------------------------------------------------

# Neural Network Class
class Tneural_network:
    """ Neural Network class for the titanic dataset. """

    # Constructor.
    def __init__(self, dim: int, num_layers: int, layer_size: int) -> None:
        """Constructor for the Neural Network Class.

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
        self.dim = dim + 1  # Bias increment.
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.learning_rate = 0.1  # Gradient descent step.

        # These are the layers of the neural network.
        # Input weights. Row i is the array of weights applied to x_i.
        self.w_in = np.random.standard_normal((self.dim, layer_size))

        # "Tensor" (3-dim array) of hidden-layer output weights.
        # w_h[lay][i][j] is the weight between lay node i and lay+1 node j.
        self.w_h = np.random.standard_normal((num_layers-1,
                                              layer_size, layer_size))

        # Output weights, comes from last layer.
        self.w_out = np.random.standard_normal((1, layer_size))

    # Training method:
    def train_network(self, train_x, train_y, num_rounds) -> None:
        """Trains the network with given data and labels for n rounds.

        Args:
            train_x (np.array): training vectors.
            train_y (np.array): labels to the x's.
            num_rounds (int): number of training rounds.

        Returns:
            None, just trains network.
        """
        # TODO: Feed forward; backward propagation.

        for i in range(1, num_rounds+1):
            # Iterate each data point
            loss = 0
            for j in range(0, train_x.shape[0]):
                dat = train_x[j]
                dat = np.append(train_x[j], [1])

                # Get the prediction for the point, using the current
                # weights (model).
                pred, vals = self.feed_forward(dat)
                # Adjust the weights (model) to account for whether
                # we're incorrect or not.
                self.backward_propagation(vals, pred, dat, train_y[j])
                loss += abs(pred - train_y[j])**2

        print("Current loss: " + str(loss))
        
        # # Set up the plotting
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.ion()
        # fig.show()
        # fig.canvas.draw()
        # plt.axis([-4, 4, -4, 4])
        # ax.axis([-4, 4, -4, 4])

        # ax.clear()
        # p_x, p_y = (train_x[np.where(train_y == 1)]).T
        # ax.plot(p_x, p_y, 'ob')
        # p_x, p_y = (train_x[np.where(train_y == -1)]).T
        # ax.plot(p_x, p_y, 'or')
        # ax.axis([-4, 4, -4, 4])

        # """
        # Plot the decision area contours
        # """

        # # Set up some arrays to compute the contours, store in an image
        # im_x = np.arange(-4, 4, 0.1)
        # im_y = np.arange(-4, 4, 0.1)
        # im_X, im_Y = np.meshgrid(im_x, im_y)

        # # values
        # im_Z = []  # np.zeros(im_X.shape)

        # # TODO: use list comp, zipping and mapping for this, for-loop is slow
        # for j in range(len(im_X)):  # walk over rows
        #     for i in range(len(im_X[0])):  # walk over columns
        #         # Get the value for
        #         # swap i and j to compensate for grid layout
        #         # dat = np.append(dat, [1])  # add bias input

        #         res, _ = self.feed_forward(dat)  # with bias
        #         im_Z.append(res)

        # im_Z = numpy.array(im_Z).reshape(im_X.shape)

        # # see the matplotlib contourf documentation
        # cset1 = plt.contourf(im_X, im_Y, im_Z, cmap='RdBu', alpha=0.5)
        # fig.canvas.draw()
        return loss

    # Actual Prediction method.
    def titanic_network_predict(self, test_x, verbose: bool = False) -> float:
        """ Prediction engineered for the titanic dataset.

        Args:
            test_x (np.array): The vector of values to test on.

        Returns:
            float: the result of the titanic vector.
        """
        # TODO: perhaps optimize for flexibility.
        test_x = np.append(test_x, [1])  # This is for the bias.
        res, _ = self.feed_forward(test_x)  # Feed forward for result.

        # Logging:
        if verbose:
            print(f'Result from NN: {res}')

        return res  # Result

    # Feeding helping method.
    def feed_forward(self, in_vect):
        """ Forward process of the neural network, no backwards propagation.

        Args:
            input_vector (vector of data): Input vector for the network.

        Returns:
            [np.array]: The outputs of each layer.  To get the result of
            the neural network, get index -1.
        """
        # 1-by-dim times dim-by-h
        outs = np.array([self.sigmoid(in_vect @ self.w_in)])
        for i in range(1, self.num_layers):
            # i-1 here because w[i] is output weights
            # get output of last layer (sig(x)) and weigh it into this layer
            ins = outs[-1] @ self.w_h[i-1]  # 1-by-h times h-by-h
            outs = np.append(outs, [self.sigmoid(ins)], axis=0)

        # last row of outs now holds the weighted output of last hidden layer
        ret = self.sigmoid(outs[-1] @ self.w_out.T)
        return ret[0], outs

    # New backprop
    def backward_propagation(self, outputs, pred, in_vect, label):
        """ Backward process of the neural network, no forward feeding.

        Args:
            outputs (np.array): Outputs from the feed_forward().
            pred (float): Predicted output.
            in_vect (np.array): Input vector.
            label (float): Expected output.
        """
        dEyo = pred - label  # Scalar 'cost.'
        dExo = dEyo * self.sigmoid_deriv(np.dot(outputs[-1], self.w_out[0]))
        dEwo = dExo * outputs[-1]

        # Hidden layer derivatives setup.
        dEwh = np.zeros((self.num_layers-1, self.layer_size, self.layer_size))
        dExh = np.zeros((self.num_layers, self.layer_size))
        dEyh = np.zeros((self.num_layers, self.layer_size))

        # Need to do output layer first, not a matrix product.
        dEyh[-1] = self.w_out * dExo  # 1-by-h times scalar.

        for i in range(self.num_layers - 2, -1, -1):
            # i-1 to get the inputs to layer i.
            x = outputs[i-1] @ self.w_h[i-1]  # 1-by-h times h-by-h.
            dExh[i] = dEyh[i] * self.sigmoid_deriv(x)  # 1-by-h.
            dEwh[i] = outputs[i-1] * dExh[i]
            if i > 0:
                # Prep the next layer.
                dEyh[i-1] = self.w_h[i] @ dExh[i].T  # h-by-h times h-by-1.

        # dEwi = outputs[0] * dEyh[0] # take care of the input layer, again
        # not a matrix product.
        in_vect = np.array([in_vect])
        dEwi = (np.matlib.repmat(in_vect.T, 1, self.layer_size) *
                np.matlib.repmat(dExh[0], self.dim, 1))
        # dim-by-h broadcast dim-by-h.

        # Adjust the hiden layer weights accoriding to the error.
        # Check to see that this follows gradient descent!
        self.w_h = self.w_h - self.learning_rate * dEwh
        self.w_in = self.w_in - self.learning_rate * dEwi
        self.w_out[0] = self.w_out[0] - self.learning_rate * dEwo

    # Static neuron output function.
    @staticmethod
    def neuron_output(weights, inputs) -> float:
        """[summary]

        Args:
            weights (np.array): Weight values as array.
            inputs (np.array): Inputs 'into' the weights.

        Returns:
            float: The output of the neuron.
        """
        return Tneural_network.sigmoid(np.dot(weights, inputs))

    # Static sigmoid function.
    @staticmethod
    def sigmoid(t: float) -> float:
        """Sigmoid Activation Function.

        Args:
            t (float): Scalar to be evaluated.
            or,
            t (np.array): Applies sigmoid to whole array.

        Returns:
            float/np.array: Output of the sigmoid function.
        """
        if isinstance(t, float):
            return 1 / (1 + math.exp(-t))
        else:  # Assume np.array ;)
            return np.array([1 / (1 + math.exp(-t_i)) for t_i in t])

    @staticmethod
    def sigmoid_deriv(t: float) -> float:
        """Sigmoid Derivative: sig(x) * (1 - sig(x))

        Args:
            t (float): Scalar to be evaluated.
            or,
            t (np.array): Applies sigmoid to whole array.

        Returns:
            float/np.array: Output of the sigmoid derivative function.
        """
        tnn = Tneural_network  # Shorthand...
        if isinstance(t, float):
            return tnn.sigmoid(t) * (1 - tnn.sigmoid(t))
        else:
            return np.array([tnn.sigmoid(t_i) * (1 - tnn.sigmoid(t_i))
                             for t_i in t])

    # Reset method for the network.
    def reset(self):
        """
        Resets the networks weights (and biases).
        """
        self.w_in = np.random.standard_normal((self.dim, self.layer_size))
        self.w_h = np.random.standard_normal((self.num_layers - 1,
                                              self.layer_size,
                                              self.layer_size))

        self.w_out = np.random.standard_normal((1, self.layer_size))


# -----------------------------------------------------------------------------


if __name__ == '__main__':
    print('-Justin Ventura λ')
    print(f'{np.pi}')
