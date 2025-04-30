# Downloading Data with the `Fusion` Class

## Overview
The `Fusion` class provides powerful methods to retrieve your data from the Fusion Data Management Platform. It allows you to download datasets, convert them into Pandas DataFrames, or retrieve them as in-memory bytes objects for flexible data handling. The primary methods for downloading and processing data are `download()`, `to_df()`, and `to_bytes()`.

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
The `download()` method is used to retrieve datasets from the Fusion platform. It supports downloading files in various formats, managing file paths, and handling parallel downloads for efficiency.

### Syntax

The following examples will assume you have already successfully established a connection to the API and have instantiated your client as below.

User guide for this set up can be found in the [Getting Started](quickstart.md) tab.

```python
fusion = Fusion()
```
To download a dataset, you will need the following information:

- **catalog**: The catalog identifier for the catalog the dataset exists in.
- **dataset**: The dataset identifier for the dataset you would like to download.
- **dt_str**: The series member identifier for the series member you would like to download. For details, see dt_str section below.
- **dataset_format**: The distribution format of the series member you would like to download. For details on determining correct format, see dataset_format section below.

```python
fusion.download(
    dataset="MY_DATASET",
    dt_str="20250430",
    dataset_format="csv",
    catalog="my_catalog"
)
```

Given that this dataset, series member, format, and catalog combination exists, the data will be downloaded to the following file path:

``'downloads/MY_DATASET_my_catalog_20250430.csv'``

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

``'path/to/my/downloads/MY_DATASET_my_catalog_20250430.csv'``

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

!!! note
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
fusion.list_datasetmembers(
    dataset="MY_DATASET",
    series="20250430",
    catalog="my_catalog"
)
```

This method will return a DataFrame containing a row for each available distribution of the specified series member within a dataset. Utilizing one of these available formats as your ``dataset_format`` argument will ensure you are requesting the download of an existing distribution of a series member.

### Debugging Download

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
    **Useful when:**
    
    Error message is unclear

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
    **Useful when:**

    You are re-downloading a file you have previously downloaded
    
    The file you are downloading is empty