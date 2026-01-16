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
    # common mirror used by TF/Keras
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
_KERAS_NPZ_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:  # noqa: BLE001
        return float(default)


def _download_stream(url: str, dst: str, *, ctx: ssl.SSLContext, timeout_s: float, verbose: bool) -> None:
    tmp = dst + ".part"
    if os.path.exists(tmp):
        try:
            os.remove(tmp)
        except OSError:
            pass

    req = urllib.request.Request(url, headers={"User-Agent": "ironfi-mnist-downloader"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout_s) as r:  # noqa: S310
        total = r.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else None
        if verbose:
            if total_i is None:
                print(f"[mnist] downloading {os.path.basename(dst)} from {url} ...", flush=True)
            else:
                print(f"[mnist] downloading {os.path.basename(dst)} ({total_i/1e6:.1f} MB) from {url} ...", flush=True)

        read = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                if verbose and total_i and total_i > 0:
                    # print every ~5MB
                    if read % (5 * 1024 * 1024) < len(chunk):
                        pct = 100.0 * read / total_i
                        print(f"[mnist]   {pct:5.1f}% ({read/1e6:.1f}/{total_i/1e6:.1f} MB)", flush=True)

    os.replace(tmp, dst)
    if verbose:
        print(f"[mnist] saved {dst}", flush=True)


def _download(urls: list[str], dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return

    timeout_s = _env_float("IRONFI_MNIST_TIMEOUT_S", 30.0)
    verbose = _env_flag("IRONFI_MNIST_VERBOSE")

    # Use certifi if available (helps on machines where system CA store is not configured for Python).
    try:
        import certifi  # type: ignore

        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        ctx = ssl.create_default_context()

    last_err: Exception | None = None
    for url in urls:
        try:
            _download_stream(url, dst, ctx=ctx, timeout_s=timeout_s, verbose=verbose)
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
                _download_stream(url, dst, ctx=insecure_ctx, timeout_s=timeout_s, verbose=verbose)
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
        try:
            for _, fname in _FILES.items():
                urls = [base + fname for base in _MNIST_BASE_URLS]
                dst = os.path.join(raw_dir, fname)
                _download(urls, dst)
        except Exception as e:  # noqa: BLE001
            # Final fallback: fetch the pre-packed MNIST .npz used by TF/Keras and convert to our cache format.
            # This is often reachable even when the idx.gz mirrors are blocked by a network.
            timeout_s = _env_float("IRONFI_MNIST_TIMEOUT_S", 30.0)
            verbose = _env_flag("IRONFI_MNIST_VERBOSE")
            try:
                tmp_npz = os.path.join(raw_dir, "mnist_keras.npz")
                ctx = ssl._create_unverified_context() if _env_flag("IRONFI_ALLOW_INSECURE_SSL") else ssl.create_default_context()
                if verbose:
                    print("[mnist] idx.gz download failed; falling back to TF/Keras mnist.npz ...", flush=True)
                _download_stream(_KERAS_NPZ_URL, tmp_npz, ctx=ctx, timeout_s=timeout_s, verbose=verbose)
                with np.load(tmp_npz) as d:
                    xtr = d["x_train"]
                    ytr = d["y_train"]
                    xte = d["x_test"]
                    yte = d["y_test"]
                np.savez_compressed(npz_path, X_train=xtr, y_train=ytr, X_test=xte, y_test=yte)
                return npz_path
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(
                    "MNIST download failed (idx.gz mirrors + keras mnist.npz fallback). "
                    "Try setting IRONFI_MNIST_VERBOSE=1 and/or increasing IRONFI_MNIST_TIMEOUT_S (e.g., 120)."
                ) from (e2 or e)

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

