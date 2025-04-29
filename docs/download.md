# User Guide: Downloading Data with the `Fusion` Class

## Overview
The `Fusion` class provides powerful methods to interact with the Fusion Data Management Platform. It allows you to download datasets, convert them into Pandas DataFrames, or retrieve them as in-memory bytes objects for flexible data handling. The primary methods for downloading and processing data are `download()`, `to_df()`, and `to_bytes()`.

---

## `download()` Method

### Overview
The `download()` method is used to retrieve datasets from the Fusion platform. It supports downloading files in various formats, managing file paths, and handling parallel downloads for efficiency.

### Syntax
```python
download(
    dataset: str,
    dt_str: str,
    dataset_format: str = None,
    catalog: str = "common",
    n_par: int = 1,
    show_progress: bool = False,
    force_download: bool = False,
    download_folder: str = "./downloads",
    return_paths: bool = False,
    partitioning: bool = False,
    preserve_original_name: bool = True
)
```