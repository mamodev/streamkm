import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import argparse
import os
import struct

plt.style.use("dark_background")


def read_dataset(data_path):
    # --- Read data file ---
    with open(data_path, "rb") as f:
        magic = struct.unpack("I", f.read(4))[0]
        version = struct.unpack("I", f.read(4))[0]
        file_size = struct.unpack("Q", f.read(8))[0]
        e_per_item = struct.unpack("Q", f.read(8))[0]
        numeric_format = struct.unpack("B", f.read(1))[0]
        flags = struct.unpack("B", f.read(1))[0]
        num_dims = struct.unpack("I", f.read(4))[0]

        shape = np.frombuffer(f.read(4 * num_dims), dtype=np.uint32)
        X = np.frombuffer(f.read(), dtype=np.float32).reshape(shape)

    return X


def plot_multiple(datasets, titles, max_points=4000):
    n = len(datasets)
    features = datasets[0].shape[1]

    # consistency check
    for X in datasets:
        assert X.shape[1] == features, "All datasets must have same #features!"

    # --- Subsampling ---
    Xs = []
    for X in datasets:
        if len(X) > max_points:
            idx = np.random.choice(len(X), max_points, replace=False)
            Xs.append(X[idx])
        else:
            Xs.append(X)

    # --- PCA if >3 features ---
    if features > 3:
        pca = PCA(n_components=3)
        pca.fit(np.vstack(Xs))
        Xs = [pca.transform(X) for X in Xs]
        features = 3

    # --- Plot all datasets ---
    fig = plt.figure(figsize=(5 * n, 5))

    for i, (X, title) in enumerate(zip(Xs, titles)):
        if features == 2:
            ax = fig.add_subplot(1, n, i + 1)
            ax.scatter(X[:, 0], X[:, 1], s=5, c="cyan")
            ax.set_title(title)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")

        elif features == 3:
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=5, c="cyan")
            ax.set_title(title)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Visualizer (no labels)")
    parser.add_argument(
        "datasets",
        nargs="+",
        help="Paths to dataset feature binaries (*.bin)",
    )
    parser.add_argument(
        "--max-points", type=int, default=4000, help="Maximum points per dataset"
    )
    args = parser.parse_args()

    datasets = []
    titles = []

    for data_path in args.datasets:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        X = read_dataset(data_path)
        datasets.append(X)
        titles.append(os.path.basename(data_path))  # filename as title

    plot_multiple(datasets, titles, args.max_points)