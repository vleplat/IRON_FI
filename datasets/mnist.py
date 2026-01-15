import gzip
import os
import struct
import ssl
import urllib.request
from typing import Tuple

import numpy as np


_MNIST_BASE_URLS = [
    "https://yann.lecun.com/exdb/mnist/",
    # common mirror used by torchvision
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(urls: list[str], dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return

    # Use certifi if available (helps on machines where system CA store is not configured for Python).
    try:
        import certifi  # type: ignore

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        ctx = ssl.create_default_context()

    last_err: Exception | None = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ironfi-mnist-downloader"})
            with urllib.request.urlopen(req, context=ctx) as r:  # noqa: S310
                data = r.read()
            with open(dst, "wb") as f:
                f.write(data)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            # try next mirror
            continue

    # Optional escape hatch for environments with broken SSL verification.
    # Enable by setting IRONFI_ALLOW_INSECURE_SSL=1.
    if os.environ.get("IRONFI_ALLOW_INSECURE_SSL", "0") == "1":
        insecure_ctx = ssl._create_unverified_context()  # noqa: SLF001
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ironfi-mnist-downloader"})
                with urllib.request.urlopen(req, context=insecure_ctx) as r:  # noqa: S310
                    data = r.read()
                with open(dst, "wb") as f:
                    f.write(data)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue

    raise RuntimeError(f"Failed to download {os.path.basename(dst)} from all mirrors") from last_err


def _read_idx_images_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic for images: {magic}")
        data = f.read()
    x = np.frombuffer(data, dtype=np.uint8)
    x = x.reshape(n, rows, cols)
    return x


def _read_idx_labels_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic for labels: {magic}")
        data = f.read()
    y = np.frombuffer(data, dtype=np.uint8)
    y = y.reshape(n)
    return y


def prepare_mnist(data_dir: str, *, download: bool = True) -> str:
    """
    Ensure MNIST is available under data_dir and return the path to cached mnist.npz.

    Parameters
    ----------
    data_dir : str
        Directory to store MNIST artifacts, e.g. ``data/mnist``.
    download : bool, default=True
        If True, download missing MNIST .gz files into ``data_dir/raw``.

    Returns
    -------
    npz_path : str
        Path to ``data_dir/mnist.npz`` (created if needed).

    Notes
    -----
    - If your Python installation has broken SSL verification, you may set
      ``IRONFI_ALLOW_INSECURE_SSL=1`` to enable an insecure (verification-disabled)
      download fallback. This should only be used on trusted networks.
    """
    os.makedirs(data_dir, exist_ok=True)
    npz_path = os.path.join(data_dir, "mnist.npz")
    if os.path.exists(npz_path):
        return npz_path

    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if download:
        for _, fname in _FILES.items():
            urls = [base + fname for base in _MNIST_BASE_URLS]
            dst = os.path.join(raw_dir, fname)
            _download(urls, dst)

    paths = {k: os.path.join(raw_dir, v) for k, v in _FILES.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing {k} at {p}. Run with download=True to fetch MNIST into {raw_dir}."
            )

    xtr = _read_idx_images_gz(paths["train_images"])
    ytr = _read_idx_labels_gz(paths["train_labels"])
    xte = _read_idx_images_gz(paths["test_images"])
    yte = _read_idx_labels_gz(paths["test_labels"])

    np.savez_compressed(npz_path, X_train=xtr, y_train=ytr, X_test=xte, y_test=yte)
    return npz_path


def load_mnist(
    data_dir: str,
    *,
    download: bool = False,
    flatten: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load MNIST arrays from a local cache in data_dir.

    Parameters
    ----------
    data_dir : str
        Directory containing ``mnist.npz`` or where it should be created.
    download : bool, default=False
        If True, download MNIST files when missing.
    flatten : bool, default=True
        If True, return images as (N, 784). Otherwise (N, 28, 28).
    normalize : bool, default=True
        If True, scale uint8 images to float32 in [0,1].

    Returns:
      X_train: (N, 784) float32 if flatten else (N, 28, 28)
      y_train: (N,) int64
      X_test:  (M, 784) float32 if flatten else (M, 28, 28)
      y_test:  (M,) int64
    """
    npz_path = prepare_mnist(data_dir, download=download)
    with np.load(npz_path) as d:
        X_train = d["X_train"]
        y_train = d["y_train"]
        X_test = d["X_test"]
        y_test = d["y_test"]

    if normalize:
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train.astype(np.int64), X_test, y_test.astype(np.int64)

