import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, Optional


def make_run_dir(root: str, prefix: str, config: Dict[str, Any]) -> str:
    """
    Create a timestamped run directory under ``root`` and write ``config.json``.

    Parameters
    ----------
    root : str
        Root directory for logs, e.g. ``logs``.
    prefix : str
        Run name prefix, e.g. ``mnist_softmax_partB``.
    config : dict
        Configuration dictionary to write to ``config.json``.

    Returns
    -------
    run_dir : str
        Path to the created run directory.
    """
    os.makedirs(root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{prefix}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    write_json(os.path.join(run_dir, "config.json"), config)
    return run_dir


def write_json(path: str, obj: Any) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def append_csv_row(path: str, row: Dict[str, Any], header: Optional[list] = None) -> None:
    """
    Append a row to a CSV file, creating it (and writing the header) if needed.

    Parameters
    ----------
    path : str
        CSV file path.
    row : dict
        Mapping of column name -> value.
    header : list or None, default=None
        Column names. If None, uses ``row.keys()``.
    """
    import csv

    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        if header is None:
            header = list(row.keys())
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)

