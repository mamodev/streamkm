import argparse
import os
import struct
import numpy as np
from sklearn.datasets import fetch_openml

# ------------------------
# Args
# ------------------------
parser = argparse.ArgumentParser(
    description="Export any OpenML dataset in bin format"
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="OpenML dataset name or ID (e.g., 'mnist_784', 'Fashion-MNIST', 'covertype', or '1467')",
)
parser.add_argument("--outdir", type=str, default=".data", help="Output directory")
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional limit on number of samples (useful for very large datasets)",
)
args = parser.parse_args()

# ------------------------
# Load dataset from OpenML
# ------------------------
print(f"Fetching OpenML dataset '{args.dataset}' ...")

if args.dataset.isdigit():
    ds = int(args.dataset)
    X, y = fetch_openml(
        data_id=ds, return_X_y=True, as_frame=False
    )  # as numpy arrays

else: 
    X, y = fetch_openml(
        args.dataset, return_X_y=True, as_frame=False
    )  # as numpy arrays

# ensure y numeric if possible
try:
    y = y.astype(np.int64)
except Exception:
    from sklearn.preprocessing import LabelEncoder

    y = LabelEncoder().fit_transform(y)

# Limit if requested
if args.limit is not None:
    X = X[: args.limit]
    y = y[: args.limit]


print(
    f"Loaded dataset '{args.dataset}' with shape {X.shape}, "
    f"classes {len(set(y))}"
)

# Shuffle
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# ------------------------
# Save to bin format
# ------------------------
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


def str_to_uint32(s: str) -> int:
    assert len(s) == 4
    return struct.unpack("I", s.encode("utf-8"))[0]


with open(os.path.join(args.outdir, "data.bin"), "wb") as f:
    HEADER_SIZE = 30
    payload_size = 4 * len(X.shape) + 4 * np.prod(X.shape)
    file_size = HEADER_SIZE + payload_size

    f.write(struct.pack("I", str_to_uint32("SKDS")))  # magic (same as synthetic)
    f.write(struct.pack("I", 1))  # version
    f.write(struct.pack("Q", file_size))  # file size
    f.write(struct.pack("Q", np.prod(X.shape)))  # elements per item
    f.write(struct.pack("B", 2))  # numeric format (float32)
    f.write(struct.pack("B", 0))  # flags
    f.write(struct.pack("I", len(X.shape)))  # num dims

    f.write(np.array(X.shape, dtype=np.uint32).tobytes())
    f.write(X.astype(np.float32).tobytes())

with open(os.path.join(args.outdir, "labels.bin"), "wb") as f:
    Y = np.array(y, dtype=np.uint64)
    f.write(struct.pack("Q", len(Y.shape)))
    f.write(Y.tobytes())

print(f"Exported dataset '{args.dataset}' to {args.outdir}")