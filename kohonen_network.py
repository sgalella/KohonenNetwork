import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from tqdm import tqdm


class KohonenNeuron:
    def __init__(self, row, col, d):
        self.row = row
        self.col = col
        self.W = 2 * np.random.random(size=(d, )) - 1

    def __repr__(self):
        return f'Neuron({self.row, self.col})'

    def compute_local_similarity(self, X):
        return np.linalg.norm(self.W - X)

    def update_weights(self, X, row_min, col_min, learning_rate=0.1, sigma=1):
        d = max(abs(self.row - row_min), abs(self.col - col_min))
        damping = np.exp(- (d ** 2) / (2 * sigma ** 2))
        self.W += learning_rate * damping * (X - self.W)


class KohonenNetwork:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = None

    def __str__(self):
        return f'{self.grid}'

    def _initialize_grid(self, d):
        grid = np.zeros((self.rows, self.cols), dtype=KohonenNeuron)
        for row in range(self.rows):
            for col in range(self.cols):
                grid[row][col] = KohonenNeuron(row, col, d)
        return grid

    def _compute_global_similarity(self, X):
        similarity = np.zeros(self.grid.shape)
        for row in range(self.rows):
            for col in range(self.cols):
                neuron = self.grid[row][col]
                similarity[row][col] = neuron.compute_local_similarity(X)
        return similarity

    def _get_coord_min(self, similarity):
        return np.unravel_index(similarity.argmin(), similarity.shape)

    def train(self, data, num_epochs=10, initial_learning_rate=0.1, initial_sigma=1):
        d = data.shape[1]
        self.grid = self._initialize_grid(d)
        for e in tqdm(range(num_epochs)):
            learning_rate = initial_learning_rate * np.exp(- e / num_epochs)
            sigma = initial_sigma * np.exp(- e / num_epochs)
            for X in data:
                similarity = self._compute_global_similarity(X)
                row_min, col_min = self._get_coord_min(similarity)
                for row in self.grid:
                    for neuron in row:
                        neuron.update_weights(X, row_min, col_min, learning_rate, sigma)

    def plot(self, data, **kwargs):
        weights = []
        for row in range(self.rows):
            for col in range(self.cols):
                neuron = self.grid[row][col]
                weights.append(neuron.W)
        kmeans = KMeans(**kwargs).fit(weights)
        labels = kmeans.labels_
        plt.imshow(np.reshape(labels, (self.rows, self.cols)), cmap='brg')
        plt.axis('off')
        plt.show()


def main():
    # Random seed for reproducibility
    np.random.seed(1234)

    # Load iris dataset
    iris = datasets.load_iris(as_frame=True)
    df = iris.data
    norm_df = (df - df.mean()) / df.std()
    data = norm_df.to_numpy()

    # Run Kohonen Map
    kohonen = KohonenNetwork(rows=20, cols=20)
    kohonen.train(data, num_epochs=250)
    kohonen.plot(data, n_clusters=3)


if __name__ == '__main__':
    main()
