"""
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np

class linear_regression(object):
    """Train linear regression problem algorithm.

    Args:
        epochs (int): number of iterations over the entire dataset.
        lr (int): learning rate.
        m (int): number of datas in our dataset.
        w0 (float): first weight, sometime called theta 0 or bias.
        w1 (float): second weight, sometime called theta 1.
        costs (list): a list to save the costs values during the training.
    """

    def __init__(self, epochs=400, lr=0.01):
        self.epochs = epochs
        self.lr = lr
        self.w0 = 0
        self.w1 = 0
        self.costs = list()

    def predict(self, X):
        """Predict the price given the mileage and the weights."""
        return self.w0 + self.w1 * X

    def cost(self, X, m):
        """The cost function that we want to minimize during the training."""
        cost = (1/(2*m)) * np.sum(np.square(self.predict(X) - y))
        return cost

    def train(self, X, y):
        """The training allows to adjust little by little the weights (w0 and
        w1) thanks to the gradient descent algorithm."""
        m = X.shape[0]
        # The epoch loop:
        for epoch in range(self.epochs):
            # Compute each weights and update simultaneously: 
            tmp_w0 = self.w0 - self.lr * (1 / m) * np.sum(self.predict(X) - y)
            tmp_w1 = self.w1 - self.lr * (1 / m) * np.sum((self.predict(X) - y) * X)
            self.w0 = tmp_w0
            self.w1 = tmp_w1
            # Compute the new cost:
            cost = self.cost(X, m)
            self.costs.append(cost)

    def export_weights(self):
        """Pickle the weights to be use in predict program."""
        weights = np.array([self.w0, self.w1])
        np.savetxt('weights.csv', weights)
        # print("w0: ", self.w0, " w1: ", self.w1)

def visual():
    """"""
    def display_func(self):
        """Display the linear function and the scatter plot."""
        x = np.arange(0, 5, 0.1)
        y = self.w0 + self.w1 * x
        plt.plot(x, y)
        plt.show()        

    def display_cost(self):
        """Use Matplotlib to display the cost function variations during
        training."""
        plt.plot(self.costs)
        plt.ylabel('Cost or error')
        plt.xlabel('Epochs')
        plt.show()


def normalize(self, X):
    """Normalize the datas.
    
    We use this formula:
        x_norm = (x - mean) / std
    """
    return (X - np.mean(X)) / np.std(X)


def load_datas(filename):
    """Return X and y from the file.
    
    Args:
        filename (str): the filename to parse.
    
    Returns:
        (np.ndarray) X and y as numpy arrays.
    """
    datas = np.loadtxt(filename, delimiter=',', skiprows=1)
    X = datas[:,0]
    y = datas[:,1]
    return X, y

def argparser():
    """Parse arguments and return a dict."""
    parser = argparse.ArgumentParser(description="This program train our linear regression.")
    parser.add_argument('-v', '--visual', action='store_true', help='Visual mode')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='Set number of epochs')
    parser.add_argument('-lr', '--learningrate', type=float, default=0.01, help='Set the learning rate')    
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = argparser()
    X, y = load_datas("data.csv")
    linear = linear_regression(epochs=args['epochs'], lr=args['learningrate'])
    X = normalize(X)
    linear.train(X, y)
    linear.export_weights()
