# User Guide: Uploading Data with the `Fusion` Class

## Overview
The `upload()` method allows you to upload files or folders to the Fusion Data Management Platform. It supports local file systems and cloud storage (e.g., S3), automatically handles file formats, and validates schemas for datasets with defined structures. This guide explains the key arguments, functionality, and best practices for using the `upload()` method.

### Syntax
```python
upload(
    path: str,
    dataset: str | None = None,
    dt_str: str = "latest",
    catalog: str | None = None,
    n_par: int | None = None,
    show_progress: bool = True,
    return_paths: bool = False,
    multipart: bool = True,
    chunk_size: int = 5 * 2**20,
    from_date: str | None = None,
    to_date: str | None = None,
    preserve_original_name: bool | None = False,
    additional_headers: dict[str, str] | None = None
) -> list[tuple[bool, str, str | None]] | None
```


## Arguments

- **`path`**: The path to the file or folder to upload. Can be a local path or an S3 path, depending on the filesystem you set.

- **`dataset`**: The dataset identifier to which the file will be uploaded. If not provided, the dataset will be inferred from the file name.

- **`dt_str`**: A string (usually a date) representing the series member. Defaults to `"latest"`. Must be in `YYYYMMDD` format for range-based downloads.

- **`catalog`**: The catalog identifier. Defaults to `"common"`.

- **`n_par`**: The number of parallel uploads. Defaults to the number of available CPUs.

- **`show_progress`**: Whether to display a progress bar during the upload. Defaults to `True`.

- **`return_paths`**: If `True`, returns the paths and success statuses of the uploaded files. Defaults to `False`.

- **`multipart`**: Whether to use multipart uploads. Defaults to `True`.

- **`chunk_size`**: The maximum chunk size for multipart uploads. Defaults to `5 MB`.

- **`from_date`**: The start date of the data range contained in the distribution. Defaults to the upload date.

- **`to_date`**: The end date of the data range contained in the distribution. Defaults to the upload date.

- **`preserve_original_name`**: Whether to preserve the original file name. Defaults to `False`.

- **`additional_headers`**: Additional headers to include in the upload request.


## Key Features and Best Practices

### Path
The `path` argument specifies the file or folder to upload. It supports both:

#### Local File Systems
Provide the absolute or relative path to the file or folder.

#### S3 Paths
If your filesystem is configured for S3, you can provide an S3 path (e.g., `s3://my-bucket/my-folder/`).

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog"
)
```

### Automatic File Format Detection
The `upload()` method automatically detects the file format based on the file extension if it is a supported format. Supported formats include:

- `csv`
- `parquet`
- `json`
- `raw` (assumed to be zipped CSV files)

If the file format is not recognized, it defaults to `raw`.

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog"
)
```

### Formatting the `dt_str` Argument
The `dt_str` argument is used to specify the series member for the upload. It is typically a date in `YYYYMMDD` format. For range-based downloads, this format is required.

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    dt_str="20231001"
)
```

If `dt_str` is not provided, it defaults to `"latest"`, which represents the most recent series member.

### Using the `preserve_original_name` Argument
The `preserve_original_name` argument ensures that the original file name is preserved during the upload. This is useful when the file name contains meaningful metadata.

**Important:** This argument can only be used when both `catalog` and `dataset` are provided.

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    preserve_original_name=True
)
```

### Schema Validation
For datasets with defined schemas (i.e., not raw datasets), the `upload()` method validates the uploaded file against the schema. This includes:

#### Column Headers
The column names and their order must match the defined schema.

#### Data Types
The data types of the columns must match the schema.

If the file does not conform to the schema, the upload will fail.

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_STRUCTURED_DATASET",
    catalog="my_catalog"
)
```

### Example Usage

#### Uploading a Single File
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    dt_str="20231001",
    show_progress=True
)
```

#### Uploading a Folder
```python
fusion.upload(
    path="data/my_folder/",
    catalog="my_catalog",
    show_progress=True
)
```

#### Preserving the Original File Name
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    preserve_original_name=True
)
```

### Tips for Efficient Uploads
- Use `n_par` to increase the number of parallel uploads for large datasets.
- Set `show_progress=True` to monitor the upload progress.
- Ensure your file conforms to the dataset schema if uploading to a structured dataset.
- Use `preserve_original_name` to retain meaningful file names during the upload.
- For large files, consider adjusting the `chunk_size` for optimal performance during multipart uploads.

