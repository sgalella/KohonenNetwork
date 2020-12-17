import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

from .kohonen import KohonenNetwork


# Random seed for reproducibility
np.random.seed(1234)

# Load iris dataset
iris = datasets.load_iris(as_frame=True)
df = iris.data
norm_df = (df - df.mean()) / df.std()
data = norm_df.to_numpy()
labels = iris.target.to_numpy()
labels_names = iris.target_names

# Run Kohonen Map
kohonen = KohonenNetwork(rows=20, cols=20)
kohonen.train(data, num_epochs=200)
kohonen.plot(data, labels, labels_names, n_clusters=3)
plt.show()
