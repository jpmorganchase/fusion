# Downloading Data with the `Fusion` Class

The `Fusion` class provides powerful methods to retrieve your data from the Fusion Data Management Platform. It allows you to download datasets, convert them into Pandas DataFrames, or retrieve them as in-memory bytes objects for flexible data handling. The primary methods for downloading and processing data are `download()`, `to_df()`, `to_table()`, and `to_bytes()`. We will also demonstrate how to download using `fsync()`.

---

!!! warning
    You can only download datasets that you are **Subscribed** to. To check if you are subscribed to a dataset, you can navigate to the dataset on the Fusion catalog page or programmatically check your access using the below command:

    ```python
    fusion = Fusion()

    fusion.list_datasets("<DATASET_ID>", catalog="<CATALOG_ID>")['status']
    ```

    If you are subscribed, the output should be:
    ```python
    0    Subscribed
    Name: status, dtype: object
    ```
    Access to a dataset can be requested via the Fusion catalog page. Please reach out to the appropriate team for more information.


## Using the `download()` Method

### Overview
The `download()` method is used to retrieve datasets from the Fusion platform. It supports downloading files in various formats, managing file paths, and handling downloads efficiently.

### Syntax

The following examples will assume you have already successfully established a connection to the API and have instantiated your client as below.

User guide for this set up can be found in the [Getting Started](quickstart.md) tab.

```python
fusion = Fusion()
```
To download a dataset, you will need the following information:

- **catalog**: The catalog identifier for the catalog the dataset exists in.
- **dataset**: The dataset identifier for the dataset you would like to download.
- **dt_str**: The series member identifier for the series member you would like to download. For details, see [dt_str](#populating-the-dt_str-argument) section below.
- **dataset_format**: The distribution format of the series member you would like to download. For details on determining correct format, see [dataset_format](#populating-the-dataset_format-argument) section below.

```python
fusion.download(
    dataset="MY_DATASET",
    dt_str="20250430",
    dataset_format="csv",
    catalog="my_catalog"
)
```

Given that this dataset, series member, format, and catalog combination exists, the data will be downloaded to the following file path:

``'downloads/MY_DATASET__my_catalog__20250430.csv'``

### Download Folder

By default, downloaded files will be downloaded to the ``'downloads'`` folder, which will be created if it does not exist. Users can define an alternate path for downloading using the ``download_folder`` argument:

```python
fusion.download(
    dataset="MY_DATASET",
    dt_str="20250430",
    dataset_format="csv",
    catalog="my_catalog",
    download_folder="path/to/my/downloads"
)
```

Given that this dataset, series member, format, and catalog combination exists, the data will be downloaded to the following file path:

``'path/to/my/downloads/MY_DATASET__my_catalog__20250430.csv'``

### Preserving Original File Name

To preserve the original file name during download, set the ``preserve_original_name`` argument to ``True``:

```python
fusion.download(
    dataset="MY_DATASET",
    dt_str="20250430",
    dataset_format="csv",
    catalog="my_catalog",
    preserve_original_name=True
)
```

Given that this dataset, series member, format, and catalog combination exists, if the original file name was ``'my_file.csv'``, the data will be downloaded to the following file path:

``'downloads/my_file.csv'``

With the original name of the file preserved.

!!! warning
    In order to preserve the original file name during download, the ``preserve_original_name`` needs to have been set during upload. Please see the [Upload](upload.md) page for more details.


### Populating the ``dt_str`` Argument

Correctly populating the ``dt_str`` argument is crucial for avoiding unexpected errors. In order to confirm you are requesting an available series member, you can utilize the ``list_datasetmembers`` method for a given dataset:

```python
fusion.list_datasetmembers(
    dataset="MY_DATASET",
    catalog="my_catalog"
)
```

This method will return a DataFrame containing a row for each available series member within the requested dataset. Utilizing one of the series member identifiers contained in this output as your ``dt_str`` argument will ensure you are requesting the download of an existing series member.


#### Downloading a Range of Series Members

Downloading a range of series members is currently supported when the series member identifiers contained in a dataset are in the format ``'YYYYMMDD'``.

For example, if there are three series members in my dataset with identifiers ``'20250428'``, ``'20250429'``, and ``'20250430'``, I can download all three at once by populating the ``'dt_str'`` argument as follows:

```python
fusion.download(
    dataset="MY_DATASET",
    dt_str="20250428:20250430",
    dataset_format="csv",
    catalog="my_catalog"
)
```

### Populating the ``dataset_format`` Argument

Correctly populating the ``dataset_format`` argument is also crucial for avoiding unexpected errors. In order to confirm you are requesting an available format for a specific series member, you can utilize the ``list_distributions`` method:


```python
fusion.list_distributions(
    dataset="MY_DATASET",
    series="20250430",
    catalog="my_catalog"
)
```

This method will return a DataFrame containing a row for each available distribution of the specified series member within a dataset. Utilizing one of these available formats as your ``dataset_format`` argument will ensure you are requesting the download of an existing distribution of a series member.

!!! note "Default value and setting ``dataset_format`` to ``None`` during ``download()``"
    The default value for the ``dataset_format`` argument in the ``download()`` method is ``'parquet'``. However, if the user sets the argument to ``None``, the following logic occurs:

    1. If only one format is available for the requested series member, that format will be downloaded.

    2. If multiple formats are detected, a ``FileFormatError`` will be raised, requesting the user to specify a format.
    
    **Useful when**:
    
    - The user knows there is only one format available, but doesn't want to specify it.


### Debugging Download

There are a few arguments available within the ``download()`` method that may be useful to users when unexpected errors occur: 

!!! tip "Return More Details in Output"
    Set the ``return_paths`` argument to ``True``:
    ```python
    fusion.download(
        dataset="MY_DATASET",
        dt_str="20250430",
        dataset_format="csv",
        return_paths=True
    )
    ```
    **What it does:**
    
    - Returns the paths where the files were downloaded (when the download is successful)
    - Provides a more detailed error message to help with debugging (when the download fails)

    **Useful when:**
    
    - Error message is unclear

!!! tip "Force Download"
    Set the ``force_download`` argument to ``True``:
    ```python
    fusion.download(
        dataset="MY_DATASET",
        dt_str="20250430",
        dataset_format="csv",
        force_download=True
    )
    ```
    **What it Does**

    - Overwrites the existing file if seriesmember was previously downloaded

    **Useful when:**

    - You want to ensure you have the latest version of the file
    - The existing file is corrupted or incomplete

Additionally, below you will find several common mistakes that users can make when attempting to download a file, as well as suggested work-arounds.

#### Common Errors & Workarounds

- **403**: Permission Denied. Please check that you have access to the dataset you are requesting. Keep in mind, you must be ``'Subscribed'`` to a dataset to download, as well as have the appropriate access to the catalog.
- **404**: File not found. Please check that the identifiers and arguments are correct. The ``dataset_format`` argument must be an available distribution of the series member (details [here](#populating-the-dataset_format-argument)), the ``dt_str`` must be an existing series member within the dataset (details [here](#populating-the-dt_str-argument)), and the dataset must exist within the catalog.
- **CredentialError**: There was an issue with your credentials. Please check that your credentials file exists and is correct. See the [Getting Started](quickstart.md) page for credentials details.


## Using the `to_df()` Method

### Overview
The ``to_df()`` method converts a dataset into a Pandas DataFrame for structured data analysis. It is particularly useful for working with tabular data formats.

#### Supported Distributions
!!! info "Supported Distributions"
    The ``to_df`` method supports the following file types: [``'csv'``, ``'parquet'``, ``'json'``, and ``'raw'``], where raw is assumed to be zipped csv files.

### Syntax

In order to retrieve a dataset as a Pandas DataFrame, the following information is required:


- **dataset**: The dataset identifier.
- **dt_str**: The specific series member of the dataset to retrieve. Refer to the populating the ``dt_str`` argument [guide](#populating-the-dt_str-argument) for more details. Defaults to ``"latest"``.
- **dataset_format**: The file format (e.g., ``"parquet"``, ``"csv"``). Defaults to ``"parquet"``. Refer to the populating the ``dataset_format`` [guide](#populating-the-dataset_format-argument) for more details.
- **catalog**: The catalog identifier. Defaults to ``"common"``.


```python
df = fusion.to_df(
    dataset="FXO_SP",
    dt_str="2023-10-01",
    dataset_format="csv",
    catalog="common"
)
```

!!! info "Argument Definitions"
    Under the hood, the ``to_df()`` method performs similar operations to the ``download()`` method, making the logic for populating arguments very similar. See the relevant sections for more information on shared arguments:

    - Populating the ``download_folder`` argument [guide](#download-folder).
    - Populating the ``dt_str`` argument [guide](#populating-the-dt_str-argument).
    - Populating the ``dataset_format`` argument [guide](#populating-the-dataset_format-argument). Keep in mind the [supported formats](#supported-distributions).

!!! warning
    The ``to_df()`` method retrieves the dataset as a Pandas DataFrame, which is stored entirely in memory (RAM). This means that the size of the DataFrame is limited by the available memory on your machine.

### Filtering the Returned DataFrame with ``filters``

The ``filters`` argument allows users to filter the data to be returned in the outputted DataFrame.


This argument should be formatted as a ``List[Tuple]`` or ``List[List[Tuple]]`` where the ``Tuple`` is structured as follows: ``("<column>", "<operation>", "<value>")``.

For example:

```python
my_filters = [("instrument_name", "==", "AUDUSD | Spot")]
```
By sending this filter condition into the ``filters`` argument, the user is filtering for rows in which the ``"instrument_name"`` column is equal to ``"AUDUSD | Spot"``:

```python
df = fusion.to_df(
    dataset="FXO_SP",
    dt_str="2023-10-01",
    dataset_format="csv",
    catalog="common",
    filters=my_filters,
)
```

The returned DataFrame will contain only rows where the filter condition is met.

!!! info "Supported Operations for ``filters``"
    equals: `"="`, `"=="`

    does not equal: `"!="`

    less than: `"<"`

    greater than: `">"`

    less than or equal to: `"<="`

    greater than or equal to: `">="`

    Is in: `"in"`

    Not in: `"not in"`

### Selecting Columns for the Returned DataFrame

The ``columns`` argument allows users to select the columns to be included in the outputted DataFrame. 

For example:
```python
df = fusion.to_df(
    dataset="FXO_SP",
    dt_str="2023-10-01",
    dataset_format="csv",
    catalog="common",
    columns=["instrument_name", "fx_rate"],
)
```

The returned DataFrame will contain two columns: ``'instrument_name'`` and ``'fx_rate'``.


!!! Danger "Important Notes"
    - If your dataset is very large, you may encounter memory errors if you load the entire dataset in with ``to_df()``.
    - Consider filtering your data using the `filters` argument to reduce the size of the dataset being loaded.
    - For extremely large datasets, consider using alternative methods like `to_bytes()` or downloading the data directly using the `download()` method.


## Using the `to_table()` Method

### Overview
The `to_table()` method retrieves a dataset and converts it into a PyArrow Table for efficient data processing and analysis. PyArrow Tables are particularly useful for handling large datasets and performing columnar operations, as they are optimized for memory usage and performance.

#### Supported Distributions
!!! info "Supported Distributions"
    The `to_table()` method supports the same file types as `to_df()`: [`'csv'`, `'parquet'`, `'json'`, and `'raw'`], where raw is assumed to be zipped CSV files.

### Syntax

To retrieve a dataset as a PyArrow Table, the following information is required:

- **dataset**: The dataset identifier.
- **dt_str**: The specific series member of the dataset to retrieve. Refer to the [populating the `dt_str` argument guide](#populating-the-dt_str-argument) for more details. Defaults to `"latest"`.
- **dataset_format**: The file format (e.g., `"parquet"`, `"csv"`). Defaults to `"parquet"`. Refer to the [populating the `dataset_format` guide](#populating-the-dataset_format-argument) for more details.
- **catalog**: The catalog identifier. Defaults to `"common"`.

```python
table = fusion.to_table(
    dataset="FXO_SP",
    dt_str="2023-10-01",
    dataset_format="parquet",
    catalog="common"
)
```

!!! info "Argument Definitions"
    The `to_table()` method shares many of the same arguments and underlying logic as the `to_df()` method. For detailed explanations of these arguments, refer to the relevant sections in the `to_df()` documentation:

    - [Populating the `download_folder` argument](#download-folder).
    - [Populating the `dt_str` argument](#populating-the-dt_str-argument).
    - [Populating the `dataset_format` argument](#populating-the-dataset_format-argument). Keep in mind the supported formats.

!!! warning
    Unlike `to_df()`, which loads the dataset into a Pandas DataFrame, `to_table()` loads the dataset into a PyArrow Table. PyArrow Tables are more memory-efficient but still require sufficient system memory to handle large datasets.

### Filtering the Returned Table with `filters`

The `filters` argument in `to_table()` works similarly to the `filters` argument in `to_df()`. It allows users to filter the data before it is returned in the PyArrow Table.

For details on how to use the `filters` argument, refer to the [Filtering the Returned DataFrame with `filters`](#filtering-the-returned-dataframe-with-filters) section in the `to_df()` documentation. The same syntax and supported operations apply.

### Selecting Columns for the Returned Table

The `columns` argument in `to_table()` allows users to specify which columns to include in the returned PyArrow Table. This functionality is identical to the `columns` argument in `to_df()`.

For more information, refer to the [Selecting Columns for the Returned DataFrame](#selecting-columns-for-the-returned-dataframe) section in the `to_df()` documentation.

### Key Differences Between `to_df()` and `to_table()`

- **Output Format**: While `to_df()` returns a Pandas DataFrame, `to_table()` returns a PyArrow Table.
- **Performance**: PyArrow Tables are optimized for columnar operations and are more memory-efficient, making them suitable for larger datasets.
- **Integration**: PyArrow Tables can be easily converted to other formats, such as Pandas DataFrames or Parquet files, for further processing.

### Example Usage

```python
table = fusion.to_table(
    dataset="FXO_SP",
    dt_str="2023-10-01",
    dataset_format="parquet",
    catalog="common",
    columns=["instrument_name", "fx_rate"],
    filters=[("instrument_name", "==", "AUDUSD | Spot")],
)
```

This example retrieves the dataset as a PyArrow Table, selecting only the `instrument_name` and `fx_rate` columns and filtering for rows where `instrument_name` equals `"AUDUSD | Spot"`.

!!! Danger "Important Notes"
    - Similar to `to_df()`, loading large datasets with `to_table()` may still encounter memory limitations, though PyArrow Tables are more efficient.
    - Consider using the `filters` argument to reduce the size of the data being loaded.
    - For extremely large datasets, consider alternative methods like `to_bytes()` or downloading the data directly using the `download()` method.


## Using the ``to_bytes()`` Method

### Overview
The ``to_bytes()`` method retrieves a dataset instance as a ``BytesIO`` object. This is useful for in-memory processing or when working with APIs that require binary data.

### Syntax

For retrieving data as a ``BytesIO`` object, the following details are required:

- **dataset**: The dataset identifier.
- **series_member**: The specific series member of the dataset to retrieve. Refer to the populating the ``dt_str`` argument [guide](#populating-the-dt_str-argument) for more details.
- **dataset_format**: The file format (e.g., ``"parquet"``, ``"csv"``). Defaults to ``"parquet"``. Refer to the populating the ``dataset_format`` [guide](#populating-the-dataset_format-argument) for more details.
- **catalog**: The catalog identifier. Defaults to ``"common"``.

```python
data_bytes = fusion.to_bytes(
    dataset="FXO_SP",
    series_member="2023-10-01",
    dataset_format="csv",
    catalog="common"
)
```

!!! note
    The ``to_bytes()`` method is ideal for scenarios where you need to process data without saving it to disk.
    The returned ``BytesIO`` object can be used directly with libraries that support in-memory file-like objects.

## Downloading with the `fsync()` Module

### Overview
The `fsync()` module provides advanced synchronization between your local filesystem and the Fusion Data Management Platform. When downloading, it creates a local folder structure that mirrors the Fusion file system, making it easy to organize and manage your data.

### How It Works
- The `fsync()` function can synchronize files in either direction: **download** (Fusion → local) or **upload** (local → Fusion).
- When downloading, it will create directories and files locally that match the structure in Fusion, including catalog, dataset, and series member folders.
- The function uses checksums to ensure only new or changed files are downloaded, and supports parallel downloads and progress tracking.

### Example Usage

```python
from fusion import Fusion
from fusion.fs_sync import fsync
import fsspec

client = Fusion()
fs_fusion = client.get_fusion_file_system()
fs_local = fsspec.filesystem('file')

fsync(
    fs_fusion=fs_fusion,
    fs_local=fs_local,
    datasets=["<your_dataset_id>"],
    dataset_format="<your_dataset_format>",
    catalog="<your_catalog_id>",
    direction="download"
)
```

### Resulting Directory Structure
After running the above code, your local directory will look like:

```<catalog_id>/<dataset_id>/<seriesmember_id>/<DATASET_ID>__<catalog_id>__<seriesmember_id>.<fileext>```


If your dataset has **many series members**, `fsync()` will synchronize **all available series members**. For example, if you download dataset `MY_DATASET` from catalog `my_catalog` and it contains multiple series members (`20250430`, `20250501`, `20250502`, etc.), your directory will look like:

```my_catalog/MY_DATASET/20250430/MY_DATASET__my_catalog__20250430.csv```
```my_catalog/MY_DATASET/20250430/MY_DATASET__my_catalog__20250501.csv```
```my_catalog/MY_DATASET/20250430/MY_DATASET__my_catalog__20250502.csv```

!!! tip 
    Use ```show_progress=True``` to monitor download progress.
    The folder structure makes it easy to locate and manage specific dataset versions.
    You can also use `fsync()` for uploads by setting ```direction="upload"```.

!!! note
    The `fsync()` module is ideal for bulk synchronization and for maintaining a local mirror of your Fusion datasets.

Refer to the `fsync()` module [page](fsync.md) for more details.




