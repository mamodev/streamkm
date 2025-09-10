import struct
import numpy as np

def str_to_uint32(s: str) -> int:
    """Convert 4-char string to uint32."""
    assert len(s) == 4
    return struct.unpack("I", s.encode("utf-8"))[0]


def read_data_from_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        # ---- HEADER ----
        magic = struct.unpack("I", f.read(4))[0]
        version = struct.unpack("I", f.read(4))[0]
        file_size = struct.unpack("Q", f.read(8))[0]
        e_per_item = struct.unpack("Q", f.read(8))[0]
        numeric_format = struct.unpack("B", f.read(1))[0]
        flags = struct.unpack("B", f.read(1))[0]
        num_dims = struct.unpack("I", f.read(4))[0]

        if magic != str_to_uint32("SKDS"):
            raise ValueError("Invalid magic number, not a supported dataset file.")

        # ---- PAYLOAD ----
        shape = struct.unpack("I" * num_dims, f.read(4 * num_dims))

        dtype_map = {0: np.float16, 1: np.float16, 2: np.float32, 3: np.float64}
        if numeric_format not in dtype_map:
            raise ValueError(f"Unsupported numeric format: {numeric_format}")

        data = np.frombuffer(f.read(), dtype=dtype_map[numeric_format])
        data = data.reshape(shape)

    return data


def write_data_to_file(path: str, X: np.ndarray, numeric_format: int = 2):
    with open(path, "wb") as f:
        sizeof_uint32 = 4
        sizeof_float = {0: 1, 1: 2, 2: 4, 3: 8}[numeric_format]

        HEADER_SIZE = 30
        payload_size = sizeof_uint32 * len(X.shape) + sizeof_float * np.prod(X.shape)
        file_size = HEADER_SIZE + payload_size

        # ---- HEADER ----
        f.write(struct.pack("I", str_to_uint32("SKDS")))  # magic
        f.write(struct.pack("I", 1))  # version
        f.write(struct.pack("Q", file_size))  # file size
        f.write(struct.pack("Q", np.prod(X.shape)))  # elements per item
        f.write(struct.pack("B", numeric_format))  # numeric format
        f.write(struct.pack("B", 0))  # flags
        f.write(struct.pack("I", len(X.shape)))  # num dims

        # ---- PAYLOAD ----
        f.write(np.array(X.shape, dtype=np.uint32).tobytes())

        dtype_map = {0: np.float8, 1: np.float16, 2: np.float32, 3: np.float64}
        f.write(X.astype(dtype_map[numeric_format]).tobytes())


def read_labels_from_file(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        num_dims = struct.unpack("Q", f.read(8))[0]
        data = np.frombuffer(f.read(), dtype=np.uint64)
    return data


def write_labels_to_file(path: str, y: np.ndarray):
    with open(path, "wb") as f:
        Y = np.array(y, dtype=np.uint64)
        f.write(struct.pack("Q", len(Y.shape)))
        f.write(Y.tobytes())


def read_dataset(data_path: str, labels_path: str):
    X = read_data_from_file(data_path)
    y = read_labels_from_file(labels_path)
    return X, y


def write_dataset(data_path: str, labels_path: str, X: np.ndarray, y: np.ndarray):
    write_data_to_file(data_path, X)
    write_labels_to_file(labels_path, y)