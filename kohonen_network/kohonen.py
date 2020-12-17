import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from .utils import get_neighbors


class KohonenNeuron:
    """ Create Kohonen Neuron. """
    def __init__(self, row, col, d):
        """
        Initialize Kohonen Neuron.

        Args:
            row (int): Row coordinate of the neuron in the network.
            col (int): Column coordinate of the neuron the network.
            d (int): Length of the dimension of the input.
        """
        self.row = row
        self.col = col
        self.W = 2 * np.random.random(size=(d, )) - 1

    def __repr__(self):
        """ Visualize neuron. """
        return f'Neuron({self.row, self.col})'

    def compute_local_distance(self, X):
        """ Calculate distance between weights and input. """
        return np.linalg.norm(self.W - X)

    def update_weights(self, X, row_min, col_min, learning_rate, sigma):
        """
        Change weights according to distance to winner neuron.

        Args:
            X (np.array): Input vector.
            row_min (int): Row coordinate of the winner unit in the network.
            col_min (int): Column coordinate of the winner unit in the network.
            learning_rate (float): Value of the learning rate.
            sigma (float): Width of the radius of neighbor function.
        """
        d = max(abs(self.row - row_min), abs(self.col - col_min))
        damping = np.exp(- (d ** 2) / (2 * sigma ** 2))
        self.W += learning_rate * damping * (X - self.W)


class KohonenNetwork:
    """ Create Kohonen Network. """
    def __init__(self, rows, cols):
        """
        Initialize Kohonen Network.

        Args:
            rows (int): Number of rows in the network.
            cols (int): Number of columns in the network.
        """
        self.rows = rows
        self.cols = cols
        self.grid = None

    def __str__(self):
        """ Visualize network. """
        return f'{self.grid}'

    def _initialize_grid(self, d):
        """
        Create neurons in the network.

        Args:
            d (int): Length of the dimension of the input.

        Returns:
            grid (np.array): Array with neurons in each coordinate.
        """
        grid = np.zeros((self.rows, self.cols), dtype=KohonenNeuron)
        for row in range(self.rows):
            for col in range(self.cols):
                grid[row][col] = KohonenNeuron(row, col, d)
        return grid

    def _compute_global_distance(self, X):
        """
        Calculates the distance between of the weights of each neuron and the input vector.

        Args:
            X (np.array): Input vector.

        Returns:
            distance (np.array): Array with the distance of each neuron.
        """
        distance = np.zeros(self.grid.shape)
        for row in range(self.rows):
            for col in range(self.cols):
                neuron = self.grid[row][col]
                distance[row][col] = neuron.compute_local_distance(X)
        return distance

    def _get_coord_min(self, distance):
        """
        Gets the coordinate in the network with smaller distance to the input.

        Args:
            distance (np.array): Array with the distance of each neuron.

        Returns:
            (row_min, col_min) (tuple): Coordinates of the neuron with less distance to input.
        """
        return np.unravel_index(distance.argmin(), distance.shape)

    def train(self, data, num_epochs=10, initial_learning_rate=1, initial_sigma=2):
        """
        Runs the Kohonen network.

        Args:
            data (np.array): Data where each row is an instance and each column a feature.
            num_epochs (int, optional): Number of iterations. Defaults to 10.
            initial_learning_rate (float, optional): Value of initial learning rate. Exponential decay. Defaults to 0.1.
            initial_sigma (int, optional): Value of the initial sigma. Exponential decay. Defaults to 1.
        """
        d = data.shape[1]
        self.grid = self._initialize_grid(d)
        for e in tqdm(range(num_epochs)):
            learning_rate = initial_learning_rate * np.exp(- e / num_epochs)
            sigma = initial_sigma * np.exp(- e / num_epochs)
            for X in data:
                distance = self._compute_global_distance(X)
                row_min, col_min = self._get_coord_min(distance)
                for row in self.grid:
                    for neuron in row:
                        neuron.update_weights(X, row_min, col_min, learning_rate, sigma)

    def plot(self, data, labels, labels_names, method='nearest', **kwargs):
        """
        Plots the network. Uses K-Means or KNN to clusterize the weights.

        Args:
            data (np.array): Data where each row is an instance and each column a feature.
        """
        if method == 'kmeans':
            weights = []
            for row in range(self.rows):
                for col in range(self.cols):
                    neuron = self.grid[row][col]
                    weights.append(neuron.W)
            kmeans = KMeans(**kwargs).fit(weights)
            labels = kmeans.labels_
            plt.imshow(np.reshape(labels, (self.rows, self.cols)), cmap='brg')
        elif method == 'nearest':
            clusters = np.full((self.rows, self.cols), np.nan)
            d = {}
            for X, y in zip(data, labels):
                distance = self._compute_global_distance(X)
                row_min, col_min = self._get_coord_min(distance)
                if (row_min, col_min) in d:
                    d[(row_min, col_min)].append(y)
                else:
                    d[(row_min, col_min)] = [y]
            for row, col in d:
                clusters[row][col] = np.mean(d[(row, col)])
            for row, col in np.argwhere(np.isnan(clusters)):
                neighbors = get_neighbors(row, col)
                count = 0
                value = 0
                for row_n, col_n in neighbors:
                    if row_n >= 0 and col_n >= 0 and row_n < self.rows and col_n < self.cols and not np.isnan(clusters[row_n][col_n]):
                        count += 1
                        value += clusters[row_n][col_n]
                if count:
                    clusters[row][col] = value / count
            plt.imshow(clusters, cmap='brg')
        cbar = plt.colorbar()
        cbar.set_ticks(range(len(labels_names)))
        cbar.set_ticklabels(labels_names)
        plt.axis('off')
