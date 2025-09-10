import matplotlib.pyplot as plt
import numpy as np
import struct
import os


import argparse 

plt.style.use('dark_background')


parser = argparse.ArgumentParser(description='Generate data for testing.')
# add subparsers for kind of data
subparsers = parser.add_subparsers(dest='kind', required=True)
blobs_parser = subparsers.add_parser('blobs', help='Generate blobs data')
classification_parser = subparsers.add_parser('classification', help='Generate classification data')

# add arguments for blobs
blobs_parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
blobs_parser.add_argument('--n_features', type=int, default=128, help='Number of features')
blobs_parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')

# add arguments for classification
classification_parser.add_argument('--n_samples', type=int, default=1000, help='Number of samples')
classification_parser.add_argument('--n_features', type=int, default=3, help='Number of features')
classification_parser.add_argument('--n_informative', type=int, default=2, help='Number of informative features')
classification_parser.add_argument('--n_redundant', type=int, default=0, help='Number of redundant features')
classification_parser.add_argument('--n_classes', type=int, default=2, help='Number of classes')

parser.add_argument('--plot', action='store_true', help='Plot the data')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--outdir', type=str, default='.data', help='Output directory for the data files')


args = parser.parse_args()

from sklearn.datasets import make_classification, make_blobs
np.random.seed(args.seed)
if args.kind == 'blobs':
    data = make_blobs(
        n_samples=args.n_samples,
        n_features=args.n_features,
        centers=args.n_clusters,
        random_state=args.seed,
    )

elif args.kind == 'classification':
    data = make_classification(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        n_redundant=args.n_redundant,
        random_state=args.seed,
        n_classes=args.n_classes,
    )
else:
    raise ValueError("Invalid kind of data. Use 'blobs' or 'classification'.")

X = data[0]
y = data[1]

# ramdom shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]


def plot_3d(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a scatter plot
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', marker='o')

    # Set labels
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

    plt.show()

def plot_2d(X, y):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2D Scatter Plot')
    plt.show()


if args.plot:
    TMP_X = X
    if len(X) > 4000:
        TMP_X = X[:4000]

    print("Plotting data...")
    if len(X[0]) == 2:
        plot_2d(TMP_X, y[:len(TMP_X)])
    elif len(X[0]) == 3:
        plot_3d(TMP_X, y[:len(TMP_X)])
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        X3d = pca.fit_transform(X)
        plot_3d(X3d, y)


# Create directory if it doesn't exist
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

def str_to_uint32(s):
    assert len(s) == 4
    return struct.unpack('I', s.encode('utf-8'))[0]

with open(os.path.join(args.outdir, 'data.bin'), 'wb') as f:
    sizeof_uint32 = 4
    sizeof_uint16 = 2
    sizeof_uint64 = 8
    sizeof_float32 = 4
    sizeof_float64 = 8
    sizeof_float16 = 2
    sizeof_float8 = 1

    # <header> (Static size: 4 + 4 + 8 + 8 + 1 + 1 + 4 = 30 bytes)
        # <magic: uint32> (0xdeadbeef)
        # <version: uint32> (1)
        # <file_size: uint64> 
        # <e_per_item: uint64> (number of elements (numbers) per item (vector))
        # <numeric_format: uint8> (0: float8, 1: float16, 2: float32, 3: float64, ...)
        # <flags: uint8> (reserved for future use, set to 0)
        # <num_dims: uint32> (Shape length)
    # </header>
    # <payload> (Dynamic size)
        # <dim_0: uint32> ... <dim_n: uint32> (shape)
        # <data: numeric_format> (row-major)
    # </payload>

    HEADER_SIZE = 30  # bytes
    payload_size = sizeof_uint32 * (len(X.shape)) + sizeof_float32 * np.prod(X.shape)
    file_size = HEADER_SIZE + payload_size
    print(f"File size: {file_size} bytes")

    f.write(struct.pack('I', str_to_uint32("SKDS")))  # magic
    f.write(struct.pack('I', 1))  # version
    f.write(struct.pack('Q', file_size))  # file size
    f.write(struct.pack('Q', np.prod(X.shape)))  # elements per item
    f.write(struct.pack('B', 2))  # numeric format (2: float32)
    f.write(struct.pack('B', 0))  # flags
    f.write(struct.pack('I', len(X.shape)))  # num dims

    # payload
    f.write(np.array(X.shape, dtype=np.uint32).tobytes())
    f.write(X.astype(np.float32).tobytes())

with open(os.path.join(args.outdir, 'labels.bin'), 'wb') as f:
    Y = np.array(y, dtype=np.uint64)
    f.write(struct.pack('Q', len(Y.shape)))
    f.write(Y.tobytes())