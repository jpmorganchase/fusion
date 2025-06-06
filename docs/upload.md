# Uploading Data with the `Fusion` Class

## Overview
The `upload()` method allows you to upload files or folders to the Fusion Data Management Platform. It supports local file systems and cloud storage (e.g., S3), automatically handles file formats, and validates schemas for datasets with defined structures. This guide explains the key arguments, functionality, and best practices for using the `upload()` method.

### Syntax

The following examples will assume you have already successfully established a connection to the API and have instantiated your client as below.

User guide for this set up can be found in the [Getting Started](quickstart.md) tab.

```python
fusion = Fusion()
```

In order to upload data to a dataset, the following information is required:


- **path**: The path to the file you wish to upload.
- **dataset**: The dataset identifier.
- **dt_str**: A string (usually a date) representing the series member. Refer to the populating the ``dt_str`` argument [guide](#populating-the-dt_str-argument) for more details on how the ``dt_str`` argument is used during download. Must be in `YYYYMMDD` format for range-based downloads. Defaults to ``"latest"``.
- **catalog**: The catalog identifier. Defaults to ``"common"``.


```python
fusion.upload(
    path = "path/to/my/file.csv",
    dataset= "MY_DATASET",
    dt_str= "20250430",
    catalog="my_catalog"
)
```


### Additional Arguments

- **`n_par`**: The number of parallel uploads. Defaults to the number of available CPUs.

- **`show_progress`**: Whether to display a progress bar during the upload. Defaults to `True`.

- **`return_paths`**: If `True`, returns the paths and success statuses of the uploaded files. Defaults to `False`.

- **`multipart`**: Whether to use multipart uploads. Defaults to `True`.

- **`chunk_size`**: The maximum chunk size for multipart uploads. Defaults to `5 MB`.

- **`from_date`**: The start date of the data range contained in the distribution. Defaults to the upload date.

- **`to_date`**: The end date of the data range contained in the distribution. Defaults to the upload date.

- **`preserve_original_name`**: Whether to preserve the original file name. Defaults to `False`.

- **`additional_headers`**: Additional headers to include in the upload request.


### Path
The `path` argument specifies the file or folder to upload. It supports both:

#### Local File Systems
Provide the absolute or relative path to the file or folder.

#### S3 Paths
If your file system is configured for S3, you can provide an S3 path (e.g., `s3://my-bucket/my-folder/`).

The file system is defined during instantiation of the `Fusion` client, and will default to your local file system.

To customize your file system, populate the `fs` argument when you define your `Fusion` client.

```python
import fsspec

my_fs = fsspec.filestsytem('s3')

fusion = Fusion(fs=my_fs)

```

Example:
```python
fusion.upload(
    path="s3://app-id-my--bucket/data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog"
)
```

### Automatic File Format Detection
The `upload()` method automatically detects the file format based on the file extension in order to preserve the original the file format during download for supported file extensions.

!!! info Supported extensions include:
    - `csv`
    - `parquet`
    - `psv`
    - `json`
    - `pdf`
    - `txt`
    - `doc`
    - `docx`
    - `htm`
    - `html`
    - `xls`
    - `xlsx`
    - `xlsm`
    - `dot`
    - `dotx`
    - `docm`
    - `dotm`
    - `rtf`
    - `odt`
    - `xltx`
    - `xlsb`
    - `jpg`
    - `jpeg`
    - `bmp`
    - `png`
    - `tif`
    - `gif`
    - `mp3`
    - `wav`
    - `mp4`
    - `mov`
    - `mkv`
    - `gz`

!!! note If the file format is not recognized, it defaults to `raw`.


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

Example:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    preserve_original_name=True
)
```

When a file is uploaded with `preserve_original_name` set to `True`, the file name will be preserved when users call the `download()` method on this file.

### Schema Validation
For datasets with defined schemas (i.e., not raw datasets), the `upload()` method validates the uploaded file against the schema. This includes:

#### Column Headers
The column names and their order must match the defined schema.

#### Data Types
The data types of the columns must match the schema.

If the file does not conform to the schema, the upload will fail.


### Tips for Efficient Uploads
- Use `n_par` to increase the number of parallel uploads for large datasets.
- Set `show_progress=True` to monitor the upload progress.
- Ensure your file conforms to the dataset schema if uploading to a structured dataset.
- Use `preserve_original_name` to retain meaningful file names during the upload.
- For large files, consider adjusting the `chunk_size` for optimal performance during multipart uploads.

