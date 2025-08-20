# Uploading Data with the `Fusion` Class

## Overview
The `upload()` method allows you to upload files or folders to the Fusion Data Management Platform. It supports upload from local file systems and cloud storage (e.g., S3), automatically handles file formats, and validates schemas for datasets with defined structures. This guide explains the key arguments, functionality, and best practices for using the `upload()` method.

### Syntax

The following examples will assume you have already successfully established a connection to the API and have instantiated your client as below.

User guide for this set up can be found in the [Getting Started](quickstart.md) tab.

```python
fusion = Fusion()
```

In order to upload data to a dataset, the following information is required:


- **path**: The path to the file you wish to upload.
- **dataset**: The dataset identifier.
- **dt_str**: A string (usually a date) representing the series member. Refer to the populating the ``dt_str`` argument [guide](download.md#populating-the-dt_str-argument) for more details on how the ``dt_str`` argument is used during download. Must be in `YYYYMMDD` format for range-based downloads. Defaults to ``"latest"``.
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


### Providing the Correct Path
The `path` argument specifies the file or folder to upload. 

Supported File Systems:

#### Local File Systems
Provide the absolute or relative path to the file or folder.

#### S3 File Systems
If your `Fusion` file system is configured for S3, you can provide an S3 path (e.g., `s3://app-id-my-bucket/data/my_file.csv`).

The file system to be used by the `Fusion` client is defined when the session is instantiated, and will default to your local file system.

To customize your file system to use s3, populate the `fs` argument when you define your `Fusion` client, as below:

```python
import fsspec

my_fs = fsspec.filestsytem('s3')

fusion = Fusion(fs=my_fs)

```

Uploading with s3 file system example:
```python
fusion.upload(
    path="s3://app-id-my-bucket/data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog"
)
```


### Formatting the `dt_str` Argument
The `dt_str` argument is used to define the series member identifier for the uploaded file. This value should be a ``string`` uniquely defining the series member to associate with the file. For range-based downloads, series members in a dataset must be formatted as a date in `YYYYMMDD` format.

Example with `YYYYMMDD` format:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    dt_str="20231001"
)
```

Example for non-time series related data:
```python
fusion.upload(
    path="data/my_file.csv",
    dataset="MY_DATASET",
    catalog="my_catalog",
    dt_str="MyData"
)
```

If `dt_str` is not provided, the created series member will be defined as `"latest"`, which represents the most recent series member.

### Automatic File Format Detection
The `upload()` method automatically detects the file format based on the file extension in order to preserve the original the file format during download for supported file extensions.

!!! info "Supported extensions include:"
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
    - `xml`

!!! note "If the file format is not recognized, it defaults to `raw`."


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

!!! info "Preserving file name during `download()`:"
    If the original series member was uploaded with the above code example file name, this file name will be preserved during download using the following example:

    ```python
        fusion.download(
            dataset="MY_DATASET",
            dt_str="20250430",
            dataset_format="csv",
            catalog="my_catalog",
            preserve_original_name=True
        )
    ```

    The data will be downloaded to the following file path:

    ``'downloads/my_file.csv'``

    With the original name of the file preserved.

!!! note "If `preserve_original_file_name` is not set to `True`:"
    The file name will use Fusion's file naming convention, ```<download_folder>/<DATASET_ID>__<catalog_id>__<seriesmember_id>.<file_format>```, during download:

    ```python
        fusion.download(
            dataset="MY_DATASET",
            dt_str="20250430",
            dataset_format="csv",
            catalog="my_catalog",
            preserve_original_name=True
        )
    ```

    ```downloads/MY_DATASET__my_catalog__20250430.csv```


### Schema Validation
For datasets with defined schemas (i.e., the dataset `isRawData` flag is defined as `False` in the dataset metadata, see the Dataset [module](datasets.md) for additional information), the `upload()` method validates the uploaded file against the schema. This validation includes:

#### Column Headers
The column names and their order must match the defined schema.

**Example:**
If the schema defines the following column order:
```plaintext
["id", "name", "age", "email"]
```

But the uploaded file has the columns in this order:
```plaintext
["name", "id", "email", "age"]
```

The upload will fail with an error indicating that the column order does not match the schema.

---

#### Data Types
The data types of the columns must match the schema.

**Example:**
If the schema defines the following column data types:
```plaintext
id: integer
name: string
age: integer
email: string
```

But the uploaded file contains data like this:
```plaintext
id,name,age,email
1,John,twenty-five,john@example.com
2,Jane,30,jane@example.com
```

The upload will fail because the value `"twenty-five"` in the `age` column is not an integer, as required by the schema.

---

If the file does not conform to the schema, the upload will fail with a detailed error message indicating the specific issue (e.g., mismatched column order or invalid data types).

Learn more about defining dataset schemas from the [`attributes`](attributes.md) module.


### Uploading a Directory

The `upload()` method allows you to upload an entire directory of files to Fusion. This is particularly useful when you have multiple files organized in subdirectories that need to be uploaded as part of a dataset.

#### Key Features
- **Preserves Directory Structure**: The method flattens the directory structure into unique file names by appending subdirectory paths to the file names.
- **Automatic File Validation**: The method validates file formats and names to ensure they meet Fusion's requirements.

#### Syntax
```python
fusion.upload(
    path="path/to/directory",
    dataset="DATASET_ID",
    catalog="CATALOG_ID",
    n_par=4,
    show_progress=True
)
```

#### Arguments
- **`path`**: The path to the directory containing the files to upload.
- **`dataset`**: The dataset identifier to which the files will be uploaded. This is mandatory when uploading a directory.
- **`catalog`**: The catalog identifier. Defaults to `"common"`.
- **`n_par`**: The number of files to upload in parallel. Defaults to the number of available CPUs.
- **`show_progress`**: Whether to display a progress bar during the upload. Defaults to `True`.

#### Example
Suppose you have the following directory structure:

```
data_folder/
├── sub1/
│   ├── file1.csv
│   ├── file2.csv
├── sub2/
│   ├── file3.csv
│   ├── file4.csv
```

You can upload the entire directory as follows:

```python
fusion.upload(
    path="data_folder",
    dataset="MY_DATASET",
    catalog="my_catalog",
    n_par=4,
    show_progress=True
)
```

During the upload, the directory structure will be flattened, and the file names will be transformed to ensure uniqueness. For example:

- `data_folder/sub1/file1.csv` → `data_folder__sub1__file1.csv`
- `data_folder/sub2/file3.csv` → `data_folder__sub2__file3.csv`

#### Notes
- **File Validation**: The method automatically validates file formats and names. Files with unsupported formats or invalid names will be skipped.
- **`dt_str` Argument Ignored**: When uploading a directory, the `dt_str` argument is ignored. The series member identifiers are derived from the file names.
