"""Main Fusion module."""
from __future__ import annotations

import json as js
import logging
import re
import sys
import warnings
from dataclasses import asdict, dataclass, field, fields
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from zipfile import ZipFile

import pandas as pd
import pyarrow as pa
from joblib import Parallel, delayed
from tabulate import tabulate

from fusion._fusion import FusionCredentials

from .exceptions import APIResponseError
from .fusion_filesystem import FusionHTTPFileSystem
from .utils import (
    RECOGNIZED_FORMATS,
    _is_json,
    convert_date_format,
    cpu_count,
    csv_to_table,
    distribution_to_filename,
    distribution_to_url,
    # download_single_file_threading,
    get_default_fs,
    get_session,
    is_dataset_raw,
    joblib_progress,
    json_to_table,
    make_bool,
    make_list,
    normalise_dt_param_str,
    parquet_to_table,
    path_to_url,
    read_csv,
    read_json,
    read_parquet,
    tidy_string,
    # stream_single_file_new_session,
    upload_files,
    validate_file_names,
)

if TYPE_CHECKING:
    import fsspec
    import requests

    from .types import PyArrowFilterT

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


class Fusion:
    """Core Fusion class for API access."""

    @staticmethod
    def _call_for_dataframe(url: str, session: requests.Session) -> pd.DataFrame:
        """Private function that calls an API endpoint and returns the data as a pandas dataframe.

        Args:
            url (Union[FusionCredentials, Union[str, dict]): URL for an API endpoint with valid parameters.
            session (requests.Session): Specify a proxy if required to access the authentication server. Defaults to {}.

        Returns:
            pandas.DataFrame: a dataframe containing the requested data.
        """
        response = session.get(url)
        response.raise_for_status()
        table = response.json()["resources"]
        ret_df = pd.DataFrame(table).reset_index(drop=True)
        return ret_df

    @staticmethod
    def _call_for_bytes_object(url: str, session: requests.Session) -> BytesIO:
        """Private function that calls an API endpoint and returns the data as a bytes object in memory.

        Args:
            url (Union[FusionCredentials, Union[str, dict]): URL for an API endpoint with valid parameters.
            session (requests.Session): Specify a proxy if required to access the authentication server. Defaults to {}.

        Returns:
            io.BytesIO: in memory file content
        """

        response = session.get(url)
        response.raise_for_status()

        return BytesIO(response.content)

    def __init__(
        self,
        credentials: Union[str, FusionCredentials] = "config/client_credentials.json",
        root_url: str = "https://fusion.jpmorgan.com/api/v1/",
        download_folder: str = "downloads",
        log_level: int = logging.ERROR,
        fs: fsspec.filesystem = None,
        log_path: str = ".",
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (Union[str, FusionCredentials]): A path to a credentials file or a fully populated
            FusionCredentials object. Defaults to 'config/client_credentials.json'.
            root_url (_type_, optional): The API root URL.
                Defaults to "https://fusion.jpmorgan.com/api/v1/".
            download_folder (str, optional): The folder path where downloaded data files
                are saved. Defaults to "downloads".
            log_level (int, optional): Set the logging level. Defaults to logging.ERROR.
            fs (fsspec.filesystem): filesystem.
            log_path (str, optional): The folder path where the log is stored.
        """
        self._default_catalog = "common"

        self.root_url = root_url
        self.download_folder = download_folder
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename=f"{log_path}/fusion_sdk.log")
        logging.addLevelName(VERBOSE_LVL, "VERBOSE")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s:%(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        elif isinstance(credentials, str):
            self.credentials = FusionCredentials.from_file(Path(credentials))
        else:
            raise ValueError(
                "credentials must be a path to a credentials file or a dictionary containing the required keys"
            )

        self.session = get_session(self.credentials, self.root_url)
        self.fs = fs if fs else get_default_fs()
        self.events: Optional[pd.DataFrame] = None

    def __repr__(self) -> str:
        """Object representation to list all available methods."""
        return "Fusion object \nAvailable methods:\n" + tabulate(
            pd.DataFrame(  # type: ignore
                [
                    [
                        method_name
                        for method_name in dir(Fusion)
                        if callable(getattr(Fusion, method_name))
                        and not method_name.startswith("_")
                    ]
                    + [
                        p
                        for p in dir(Fusion)
                        if isinstance(getattr(Fusion, p), property)
                    ],
                    [
                        getattr(Fusion, method_name).__doc__.split("\n")[0]
                        for method_name in dir(Fusion)
                        if callable(getattr(Fusion, method_name))
                        and not method_name.startswith("_")
                    ]
                    + [
                        getattr(Fusion, p).__doc__.split("\n")[0]
                        for p in dir(Fusion)
                        if isinstance(getattr(Fusion, p), property)
                    ],
                ]
            ).T.set_index(0),
            tablefmt="psql",
        )

    @property
    def default_catalog(self) -> str:
        """Returns the default catalog.

        Returns:
            None
        """
        return self._default_catalog

    @default_catalog.setter
    def default_catalog(self, catalog: str) -> None:
        """Allow the default catalog, which is "common" to be overridden.

        Args:
            catalog (str): The catalog to use as the default

        Returns:
            None
        """
        self._default_catalog = catalog

    def _use_catalog(self, catalog: Optional[str]) -> str:
        """Determine which catalog to use in an API call.

        Args:
            catalog (str): The catalog value passed as an argument to an API function wrapper.

        Returns:
            str: The catalog to use
        """
        if catalog is None:
            return self.default_catalog

        return catalog

    def get_fusion_filesystem(self) -> FusionHTTPFileSystem:
        """Creates Fusion Filesystem.

        Returns: Fusion Filesystem

        """
        return FusionHTTPFileSystem(
            client_kwargs={"root_url": self.root_url, "credentials": self.credentials}
        )

    def list_catalogs(self, output: bool = False) -> pd.DataFrame:
        """Lists the catalogs available to the API account.

        Args:
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each catalog
        """
        url = f"{self.root_url}catalogs/"
        cat_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return cat_df

    def catalog_resources(
        self, catalog: Optional[str] = None, output: bool = False
    ) -> pd.DataFrame:
        """List the resources contained within the catalog, for example products and datasets.

        Args:
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
           class:`pandas.DataFrame`: A dataframe with a row for each resource within the catalog
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}"
        cat_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return cat_df

    def list_products(
        self,
        contains: Optional[Union[str, list[str]]] = None,
        id_contains: bool = False,
        catalog: Optional[str] = None,
        output: bool = False,
        max_results: int = -1,
        display_all_columns: bool = False,
    ) -> pd.DataFrame:
        """Get the products contained in a catalog. A product is a grouping of datasets.

        Args:
            contains (Union[str, list], optional): A string or a list of strings that are product
                identifiers to filter the products list. If a list is provided then it will return
                products whose identifier matches any of the strings. Defaults to None.
            id_contains (bool): Filter datasets only where the string(s) are contained in the identifier,
                ignoring description.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each product
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/products"
        full_prod_df: pd.DataFrame = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f"{s}" for s in contains)
            if id_contains:
                filtered_df = full_prod_df[
                    full_prod_df["identifier"].str.contains(contains, case=False)
                ]
            else:
                filtered_df = full_prod_df[
                    full_prod_df["identifier"].str.contains(contains, case=False)
                    | full_prod_df["description"].str.contains(contains, case=False)
                ]
        else:
            filtered_df = full_prod_df

        filtered_df["category"] = filtered_df.category.str.join(", ")
        filtered_df["region"] = filtered_df.region.str.join(", ")
        if not display_all_columns:
            filtered_df = filtered_df[
                filtered_df.columns.intersection(
                    [
                        "identifier",
                        "title",
                        "region",
                        "category",
                        "status",
                        "description",
                    ]
                )
            ]

        if max_results > -1:
            filtered_df = filtered_df[0:max_results]

        if output:
            pass

        return filtered_df

    def list_datasets(  # noqa: PLR0913
        self,
        contains: Optional[Union[str, list[str]]] = None,
        id_contains: bool = False,
        product: Optional[Union[str, list[str]]] = None,
        catalog: Optional[str] = None,
        output: bool = False,
        max_results: int = -1,
        display_all_columns: bool = False,
        status: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get the datasets contained in a catalog.

        Args:
            contains (Union[str, list], optional): A string or a list of strings that are dataset
                identifiers to filter the datasets list. If a list is provided then it will return
                datasets whose identifier matches any of the strings. Defaults to None.
            id_contains (bool): Filter datasets only where the string(s) are contained in the identifier,
                ignoring description.
            product (Union[str, list], optional): A string or a list of strings that are product
                identifiers to filter the datasets list. Defaults to None.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed
            status (str, optional): filter the datasets by status, default is to show all results.

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each dataset.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets"
        ds_df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f"{s}" for s in contains)
            if id_contains:
                ds_df = ds_df[ds_df["identifier"].str.contains(contains, case=False)]
            else:
                ds_df = ds_df[
                    ds_df["identifier"].str.contains(contains, case=False)
                    | ds_df["description"].str.contains(contains, case=False)
                ]

        if product:
            url = f"{self.root_url}catalogs/{catalog}/productDatasets"
            prd_df = Fusion._call_for_dataframe(url, self.session)
            prd_df = (
                prd_df[prd_df["product"] == product]
                if isinstance(product, str)
                else prd_df[prd_df["product"].isin(product)]
            )
            ds_df = ds_df[
                ds_df["identifier"].str.lower().isin(prd_df["dataset"].str.lower())
            ].reset_index(drop=True)

        if max_results > -1:
            ds_df = ds_df[0:max_results]

        ds_df["category"] = ds_df.category.str.join(", ")
        ds_df["region"] = ds_df.region.str.join(", ")
        if not display_all_columns:
            cols = [
                "identifier",
                "title",
                "containerType",
                "region",
                "category",
                "coverageStartDate",
                "coverageEndDate",
                "description",
                "status",
            ]
            cols = [c for c in cols if c in ds_df.columns]
            ds_df = ds_df[cols]

        if status is not None:
            ds_df = ds_df[ds_df["status"] == status]

        if output:
            pass

        return ds_df

    def dataset_resources(
        self, dataset: str, catalog: Optional[str] = None, output: bool = False
    ) -> pd.DataFrame:
        """List the resources available for a dataset, currently this will always be a datasetseries.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each resource
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}"
        ds_res_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return ds_res_df

    def list_dataset_attributes(
        self,
        dataset: str,
        catalog: Optional[str] = None,
        output: bool = False,
        display_all_columns: bool = False,
    ) -> pd.DataFrame:
        """Returns the list of attributes that are in the dataset.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each attribute
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/attributes"
        ds_attr_df = (
            Fusion._call_for_dataframe(url, self.session)
            .sort_values(by="index")
            .reset_index(drop=True)
        )

        if not display_all_columns:
            ds_attr_df = ds_attr_df[
                ds_attr_df.columns.intersection(
                    [
                        "identifier",
                        "title",
                        "dataType",
                        "isDatasetKey",
                        "description",
                        "source",
                    ]
                )
            ]

        if output:
            pass

        return ds_attr_df

    def list_datasetmembers(
        self,
        dataset: str,
        catalog: Optional[str] = None,
        output: bool = False,
        max_results: int = -1,
    ) -> pd.DataFrame:
        """List the available members in the dataset series.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.

        Returns:
            class:`pandas.DataFrame`: a dataframe with a row for each dataset member.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries"
        ds_members_df = Fusion._call_for_dataframe(url, self.session)

        if max_results > -1:
            ds_members_df = ds_members_df[0:max_results]

        if output:
            pass

        return ds_members_df

    def datasetmember_resources(
        self,
        dataset: str,
        series: str,
        catalog: Optional[str] = None,
        output: bool = False,
    ) -> pd.DataFrame:
        """List the available resources for a datasetseries member.

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each datasetseries member resource.
                Currently, this will always be distributions.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}"
        ds_mem_res_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return ds_mem_res_df

    def list_distributions(
        self,
        dataset: str,
        series: str,
        catalog: Optional[str] = None,
        output: bool = False,
    ) -> pd.DataFrame:
        """List the available distributions (downloadable instances of the dataset with a format type).

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each distribution.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions"
        distros_df = Fusion._call_for_dataframe(url, self.session)

        if output:
            pass

        return distros_df

    def _resolve_distro_tuples(
        self,
        dataset: str,
        dt_str: str = "latest",
        dataset_format: str = "parquet",
        catalog: Optional[str] = None,
    ) -> list[tuple[str, str, str, str]]:
        """Resolve distribution tuples given specification params.

        A private utility function to generate a list of distribution tuples.
        Each tuple is a distribution, identified by catalog, dataset id,
        datasetseries member id, and the file format.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.

        Returns:
            list: a list of tuples, one for each distribution
        """
        catalog = self._use_catalog(catalog)

        datasetseries_list = self.list_datasetmembers(dataset, catalog)
        if len(datasetseries_list) == 0:
            raise AssertionError(
                f"There are no dataset members for dataset {dataset} in catalog {catalog}"
            )

        if datasetseries_list.empty:
            raise APIResponseError(  # pragma: no cover
                f"No data available for dataset {dataset}. "
                f"Check that a valid dataset identifier and date/date range has been set."
            )

        if dt_str == "latest":
            dt_str = datasetseries_list.iloc[
                datasetseries_list["createdDate"].to_numpy().argmax()
            ]["identifier"]

        parsed_dates = normalise_dt_param_str(dt_str)
        if len(parsed_dates) == 1:
            parsed_dates = (parsed_dates[0], parsed_dates[0])

        if parsed_dates[0]:
            datasetseries_list = datasetseries_list[
                pd.Series(
                    [
                        pd.to_datetime(i, errors="coerce")
                        for i in datasetseries_list["identifier"]
                    ]
                )
                >= pd.to_datetime(parsed_dates[0])
            ].reset_index()

        if parsed_dates[1]:
            datasetseries_list = datasetseries_list[
                pd.Series(
                    [
                        pd.to_datetime(i, errors="coerce")
                        for i in datasetseries_list["identifier"]
                    ]
                )
                <= pd.to_datetime(parsed_dates[1])
            ].reset_index()

        if len(datasetseries_list) == 0:
            raise APIResponseError(  # pragma: no cover
                f"No data available for dataset {dataset} in catalog {catalog}.\n"
                f"Check that a valid dataset identifier and date/date range has been set."
            )

        required_series = list(datasetseries_list["@id"])
        tups = [
            (catalog, dataset, series, dataset_format) for series in required_series
        ]

        return tups

    def download(  # noqa: PLR0912, PLR0913
        self,
        dataset: str,
        dt_str: str = "latest",
        dataset_format: str = "parquet",
        catalog: Optional[str] = None,
        n_par: Optional[int] = None,
        show_progress: bool = True,
        force_download: bool = False,
        download_folder: Optional[str] = None,
        return_paths: bool = False,
        partitioning: Optional[str] = None,
        preserve_original_name: bool = False,
    ) -> Optional[list[tuple[bool, str, Optional[str]]]]:
        """Downloads the requested distributions of a dataset to disk.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to True.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            partitioning (str, optional): Partitioning specification.
            preserve_original_name (bool, optional): Preserve the original name of the file. Defaults to False.

        Returns:

        """
        catalog = self._use_catalog(catalog)

        valid_date_range = re.compile(
            r"^(\d{4}\d{2}\d{2})$|^((\d{4}\d{2}\d{2})?([:])(\d{4}\d{2}\d{2})?)$"
        )

        if valid_date_range.match(dt_str) or dt_str == "latest":
            required_series = self._resolve_distro_tuples(
                dataset, dt_str, dataset_format, catalog
            )
        else:
            # sample data is limited to csv
            if dt_str == "sample":
                dataset_format = self.list_distributions(dataset, dt_str, catalog)[
                    "identifier"
                ].iloc[0]
            required_series = [(catalog, dataset, dt_str, dataset_format)]

        if dataset_format not in RECOGNIZED_FORMATS + ["raw"]:
            raise ValueError(f"Dataset format {dataset_format} is not supported")

        if not download_folder:
            download_folder = self.download_folder

        download_folders = [download_folder] * len(required_series)

        if partitioning == "hive":
            members = [series[2].strip("/") for series in required_series]
            download_folders = [
                f"{download_folders[i]}/{series[0]}/{series[1]}/{members[i]}"
                for i, series in enumerate(required_series)
            ]

        for d in download_folders:
            if not self.fs.exists(d):
                self.fs.mkdir(d, create_parents=True)

        n_par = cpu_count(n_par)
        download_spec = [
            {
                "lfs": self.fs,
                "rpath": distribution_to_url(
                    self.root_url,
                    series[1],
                    series[2],
                    series[3],
                    series[0],
                    is_download=True,
                ),
                "lpath": distribution_to_filename(
                    download_folders[i],
                    series[1],
                    series[2],
                    series[3],
                    series[0],
                    partitioning=partitioning,
                ),
                "overwrite": force_download,
                "preserve_original_name": preserve_original_name,
            }
            for i, series in enumerate(required_series)
        ]

        logger.log(
            VERBOSE_LVL,
            f"Beginning {len(download_spec)} downloads in batches of {n_par}",
        )
        if show_progress:
            with joblib_progress("Downloading", total=len(download_spec)):
                res = Parallel(n_jobs=n_par)(
                    delayed(self.get_fusion_filesystem().download)(**spec)
                    for spec in download_spec
                )
        else:
            res = Parallel(n_jobs=n_par)(
                delayed(self.get_fusion_filesystem().download)(**spec)
                for spec in download_spec
            )

        if (len(res) > 0) and (not all(r[0] for r in res)):
            for r in res:
                if not r[0]:
                    warnings.warn(
                        f"The download of {r[1]} was not successful", stacklevel=2
                    )
        return res if return_paths else None

    def to_df(  # noqa: PLR0913
        self,
        dataset: str,
        dt_str: str = "latest",
        dataset_format: str = "parquet",
        catalog: Optional[str] = None,
        n_par: Optional[int] = None,
        show_progress: bool = True,
        columns: Optional[list[str]] = None,
        filters: Optional[PyArrowFilterT] = None,
        force_download: bool = False,
        download_folder: Optional[str] = None,
        dataframe_type: str = "pandas",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Gets distributions for a specified date or date range and returns the data as a dataframe.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            columns (List, optional): A list of columns to return from a parquet file. Defaults to None
            filters (List, optional): List[Tuple] or List[List[Tuple]] or None (default)
                Rows which do not match the filter predicate will be removed from scanned data.
                Partition keys embedded in a nested directory structure will be exploited to avoid
                loading files at all if they contain no matching rows. If use_legacy_dataset is True,
                filters can only reference partition keys and only a hive-style directory structure
                is supported. When setting use_legacy_dataset to False, also within-file level filtering
                and different partitioning schemes are supported.
                More on https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to False.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
            dataframe_type (str, optional): Type
        Returns:
            class:`pandas.DataFrame`: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        catalog = self._use_catalog(catalog)

        # sample data is limited to csv
        if dt_str == "sample":
            dataset_format = "csv"

        if not download_folder:
            download_folder = self.download_folder
        download_res = self.download(
            dataset,
            dt_str,
            dataset_format,
            catalog,
            n_par,
            show_progress,
            force_download,
            download_folder,
            return_paths=True,
        )

        if not download_res:
            raise ValueError(
                "Must specify 'return_paths=True' in download call to use this function"
            )

        if not all(res[0] for res in download_res):
            failed_res = [res for res in download_res if not res[0]]
            raise Exception(
                f"Not all downloads were successfully completed. "
                f"Re-run to collect missing files. The following failed:\n{failed_res}"
            )

        files = [res[1] for res in download_res]

        pd_read_fn_map = {
            "csv": read_csv,
            "parquet": read_parquet,
            "parq": read_parquet,
            "json": read_json,
            "raw": read_csv,
        }

        pd_read_default_kwargs: dict[str, dict[str, object]] = {
            "csv": {
                "columns": columns,
                "filters": filters,
                "fs": self.fs,
                "dataframe_type": dataframe_type,
            },
            "parquet": {
                "columns": columns,
                "filters": filters,
                "fs": self.fs,
                "dataframe_type": dataframe_type,
            },
            "json": {
                "columns": columns,
                "filters": filters,
                "fs": self.fs,
                "dataframe_type": dataframe_type,
            },
            "raw": {
                "columns": columns,
                "filters": filters,
                "fs": self.fs,
                "dataframe_type": dataframe_type,
            },
        }

        pd_read_default_kwargs["parq"] = pd_read_default_kwargs["parquet"]

        pd_reader = pd_read_fn_map.get(dataset_format)
        pd_read_kwargs = pd_read_default_kwargs.get(dataset_format, {})
        if not pd_reader:
            raise Exception(
                f"No pandas function to read file in format {dataset_format}"
            )

        pd_read_kwargs.update(kwargs)

        if len(files) == 0:
            raise APIResponseError(
                f"No series members for dataset: {dataset} "
                f"in date or date range: {dt_str} and format: {dataset_format}"
            )
        if dataset_format in ["parquet", "parq"]:
            data_df = pd_reader(files, **pd_read_kwargs)  # type: ignore
        elif dataset_format == "raw":
            dataframes = (
                pd.concat(
                    [pd_reader(ZipFile(f).open(p), **pd_read_kwargs) for p in ZipFile(f).namelist()],  # type: ignore
                    ignore_index=True,
                )
                for f in files
            )
            data_df = pd.concat(dataframes, ignore_index=True)
        else:
            dataframes = (pd_reader(f, **pd_read_kwargs) for f in files)  # type: ignore
            if dataframe_type == "pandas":
                data_df = pd.concat(dataframes, ignore_index=True)
            if dataframe_type == "polars":
                import polars as pl

                data_df = pl.concat(dataframes, how="diagonal")  # type: ignore

        return data_df

    def to_bytes(
        self,
        dataset: str,
        series_member: str,
        dataset_format: str = "parquet",
        catalog: Optional[str] = None,
    ) -> BytesIO:
        """Returns an instance of dataset (the distribution) as a bytes object.

        Args:
            dataset (str): A dataset identifier
            series_member (str,): A dataset series member identifier
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
        """

        catalog = self._use_catalog(catalog)

        url = distribution_to_url(
            self.root_url,
            dataset,
            series_member,
            dataset_format,
            catalog,
        )

        return Fusion._call_for_bytes_object(url, self.session)

    def to_table(  # noqa: PLR0913
        self,
        dataset: str,
        dt_str: str = "latest",
        dataset_format: str = "parquet",
        catalog: Optional[str] = None,
        n_par: Optional[int] = None,
        show_progress: bool = True,
        columns: Optional[list[str]] = None,
        filters: Optional[PyArrowFilterT] = None,
        force_download: bool = False,
        download_folder: Optional[str] = None,
        **kwargs: Any,
    ) -> pa.Table:
        """Gets distributions for a specified date or date range and returns the data as an arrow table.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            columns (List, optional): A list of columns to return from a parquet file. Defaults to None
            filters (List, optional): List[Tuple] or List[List[Tuple]] or None (default)
                Rows which do not match the filter predicate will be removed from scanned data.
                Partition keys embedded in a nested directory structure will be exploited to avoid
                loading files at all if they contain no matching rows. If use_legacy_dataset is True,
                filters can only reference partition keys and only a hive-style directory structure
                is supported. When setting use_legacy_dataset to False, also within-file level filtering
                and different partitioning schemes are supported.
                More on https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to False.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
        Returns:
            class:`pyarrow.Table`: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        catalog = self._use_catalog(catalog)
        n_par = cpu_count(n_par)
        if not download_folder:
            download_folder = self.download_folder
        download_res = self.download(
            dataset,
            dt_str,
            dataset_format,
            catalog,
            n_par,
            show_progress,
            force_download,
            download_folder,
            return_paths=True,
        )

        if not download_res:
            raise ValueError(
                "Must specify 'return_paths=True' in download call to use this function"
            )

        if not all(res[0] for res in download_res):
            failed_res = [res for res in download_res if not res[0]]
            raise RuntimeError(
                f"Not all downloads were successfully completed. "
                f"Re-run to collect missing files. The following failed:\n{failed_res}"
            )

        files = [res[1] for res in download_res]

        read_fn_map = {
            "csv": csv_to_table,
            "parquet": parquet_to_table,
            "parq": parquet_to_table,
            "json": json_to_table,
            "raw": csv_to_table,
        }

        read_default_kwargs: dict[str, dict[str, object]] = {
            "csv": {"columns": columns, "filters": filters, "fs": self.fs},
            "parquet": {"columns": columns, "filters": filters, "fs": self.fs},
            "json": {"columns": columns, "filters": filters, "fs": self.fs},
            "raw": {"columns": columns, "filters": filters, "fs": self.fs},
        }

        read_default_kwargs["parq"] = read_default_kwargs["parquet"]

        reader = read_fn_map.get(dataset_format)
        read_kwargs = read_default_kwargs.get(dataset_format, {})
        if not reader:
            raise AssertionError(f"No function to read file in format {dataset_format}")

        read_kwargs.update(kwargs)

        if len(files) == 0:
            raise APIResponseError(
                f"No series members for dataset: {dataset} "
                f"in date or date range: {dt_str} and format: {dataset_format}"
            )
        if dataset_format in ["parquet", "parq"]:
            tbl = reader(files, **read_kwargs)  # type: ignore
        else:
            tbl = (reader(f, **read_kwargs) for f in files)  # type: ignore
            tbl = pa.concat_tables(tbl)

        return tbl

    def upload(  # noqa: PLR0913
        self,
        path: str,
        dataset: Optional[str] = None,
        dt_str: str = "latest",
        catalog: Optional[str] = None,
        n_par: Optional[int] = None,
        show_progress: bool = True,
        return_paths: bool = False,
        multipart: bool = True,
        chunk_size: int = 5 * 2**20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        preserve_original_name: Optional[bool] = False,
        additional_headers: Optional[dict[str, str]] = None,
    ) -> Optional[list[tuple[bool, str, Optional[str]]]]:
        """Uploads the requested files/files to Fusion.

        Args:
            path (str): path to a file or a folder with files
            dataset (str, optional): Dataset identifier to which the file will be uploaded (for single file only).
                                    If not provided the dataset will be implied from file's name.
            dt_str (str, optional): A file name. Can be any string but is usually a date.
                                    Defaults to 'latest' which will return the most recent.
                                    Relevant for a single file upload only. If not provided the dataset will
                                    be implied from file's name.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            multipart (bool, optional): Is multipart upload.
            chunk_size (int, optional): Maximum chunk size.
            from_date (str, optional): start of the data date range contained in the distribution,
                defaults to upoad date
            to_date (str, optional): end of the data date range contained in the distribution,
                defaults to upload date.
            preserve_original_name (bool, optional): Preserve the original name of the file. Defaults to False.

        Returns:


        """
        catalog = self._use_catalog(catalog)

        if not self.fs.exists(path):
            raise RuntimeError("The provided path does not exist")

        fs_fusion = self.get_fusion_filesystem()
        if self.fs.info(path)["type"] == "directory":
            file_path_lst = self.fs.find(path)
            local_file_validation = validate_file_names(file_path_lst, fs_fusion)
            file_path_lst = [
                f for flag, f in zip(local_file_validation, file_path_lst) if flag
            ]
            file_name = [f.split("/")[-1] for f in file_path_lst]
            is_raw_lst = is_dataset_raw(file_path_lst, fs_fusion)
            local_url_eqiv = [
                path_to_url(i, r) for i, r in zip(file_path_lst, is_raw_lst)
            ]
        else:
            file_path_lst = [path]
            if not catalog or not dataset:
                local_file_validation = validate_file_names(file_path_lst, fs_fusion)
                file_path_lst = [
                    f for flag, f in zip(local_file_validation, file_path_lst) if flag
                ]
                is_raw_lst = is_dataset_raw(file_path_lst, fs_fusion)
                local_url_eqiv = [
                    path_to_url(i, r) for i, r in zip(file_path_lst, is_raw_lst)
                ]
                if preserve_original_name:
                    raise ValueError(
                        "preserve_original_name can only be used when catalog and dataset are provided."
                    )
            else:
                date_identifier = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
                if date_identifier.match(dt_str):
                    dt_str = (
                        dt_str
                        if dt_str != "latest"
                        else pd.Timestamp("today").date().strftime("%Y%m%d")
                    )
                    dt_str = pd.Timestamp(dt_str).date().strftime("%Y%m%d")

                if catalog not in fs_fusion.ls("") or dataset not in [
                    i.split("/")[-1] for i in fs_fusion.ls(f"{catalog}/datasets")
                ]:
                    msg = (
                        f"File file has not been uploaded, one of the catalog: {catalog} "
                        f"or dataset: {dataset} does not exit."
                    )
                    warnings.warn(msg, stacklevel=2)
                    return [(False, path, msg)]
                file_format = path.split(".")[-1]
                file_name = [path.split("/")[-1]]
                file_format = (
                    "raw" if file_format not in RECOGNIZED_FORMATS else file_format
                )

                local_url_eqiv = [
                    "/".join(
                        distribution_to_url(
                            "", dataset, dt_str, file_format, catalog, False
                        ).split("/")[1:]
                    )
                ]

        if not preserve_original_name:
            data_map_df = pd.DataFrame([file_path_lst, local_url_eqiv]).T
            data_map_df.columns = pd.Index(["path", "url"])
        else:
            data_map_df = pd.DataFrame([file_path_lst, local_url_eqiv, file_name]).T
            data_map_df.columns = pd.Index(["path", "url", "file_name"])

        n_par = cpu_count(n_par)
        parallel = len(data_map_df) > 1
        res = upload_files(
            fs_fusion,
            self.fs,
            data_map_df,
            parallel=parallel,
            n_par=n_par,
            multipart=multipart,
            chunk_size=chunk_size,
            show_progress=show_progress,
            from_date=from_date,
            to_date=to_date,
            additional_headers=additional_headers,
        )

        if not all(r[0] for r in res):
            failed_res = [r for r in res if not r[0]]
            msg = f"Not all uploads were successfully completed. The following failed:\n{failed_res}"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        return res if return_paths else None

    def from_bytes(  # noqa: PLR0913
        self,
        data: BytesIO,
        dataset: Optional[str] = None,
        series_member: str = "latest",
        catalog: Optional[str] = None,
        distribution: str = "parquet",
        show_progress: bool = True,
        return_paths: bool = False,
        chunk_size: int = 5 * 2**20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        file_name: Optional[str] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Optional[list[tuple[bool, str, Optional[str]]]]:
        """Uploads data from an object in memory.

        Args:
            data (str): an object in memory to upload
            dataset (str, optional): Dataset identifier to which the file will be uploaded (for single file only).
                                    If not provided the dataset will be implied from file's name.
            series_member (str, optional): A single date or label. Defaults to 'latest' which will return
                the most recent.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            distribution (str, optional): A distribution type, e.g. a file format or raw
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.
            chunk_size (int, optional): Maximum chunk size.
            from_date (str, optional): start of the data date range contained in the distribution,
                defaults to upload date
            to_date (str, optional): end of the data date range contained in the distribution, defaults to upload date.
            file_name (str, optional): file name to be used for the uploaded file. Defaults to Fusion standard naming.

        Returns:


        """
        catalog = self._use_catalog(catalog)

        fs_fusion = self.get_fusion_filesystem()
        if distribution not in RECOGNIZED_FORMATS + ["raw"]:
            raise ValueError(f"Dataset format {distribution} is not supported")

        is_raw = js.loads(fs_fusion.cat(f"{catalog}/datasets/{dataset}"))["isRawData"]
        local_url_eqiv = path_to_url(
            f"{dataset}__{catalog}__{series_member}.{distribution}", is_raw
        )

        data_map_df = pd.DataFrame(["", local_url_eqiv, file_name]).T
        data_map_df.columns = ["path", "url", "file_name"]  # type: ignore

        res = upload_files(
            fs_fusion,
            data,
            data_map_df,
            parallel=False,
            n_par=1,
            multipart=False,
            chunk_size=chunk_size,
            show_progress=show_progress,
            from_date=from_date,
            to_date=to_date,
        )

        if not all(r[0] for r in res):
            failed_res = [r for r in res if not r[0]]
            msg = f"Not all uploads were successfully completed. The following failed:\n{failed_res}"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)

        return res if return_paths else None

    def listen_to_events(
        self,
        last_event_id: Optional[str] = None,
        catalog: Optional[str] = None,
        url: str = "https://fusion.jpmorgan.com/api/v1/",
    ) -> Union[None, pd.DataFrame]:
        """Run server sent event listener in the background. Retrieve results by running get_events.

        Args:
            last_event_id (str): Last event ID (exclusive).
            catalog (str): catalog.
            url (str): subscription url.
        Returns:
            Union[None, class:`pandas.DataFrame`]: If in_background is True then the function returns no output.
                If in_background is set to False then pandas DataFrame is output upon keyboard termination.
        """

        catalog = self._use_catalog(catalog)
        import asyncio
        import json
        import threading

        from aiohttp_sse_client import client as sse_client

        from .utils import get_client

        kwargs: dict[str, Any] = {}
        if last_event_id:
            kwargs = {"headers": {"Last-Event-ID": last_event_id}}

        async def async_events() -> None:
            """Events sync function.

            Returns:
                None
            """
            timeout = 1e100
            session = await get_client(self.credentials, timeout=timeout)
            async with sse_client.EventSource(
                f"{url}catalogs/{catalog}/notifications/subscribe",
                session=session,
                **kwargs,
            ) as messages:
                try:
                    async for msg in messages:
                        event = json.loads(msg.data)
                        if self.events is None:
                            self.events = pd.DataFrame()
                        else:
                            self.events = pd.concat(
                                [self.events, pd.DataFrame(event)], ignore_index=True
                            )
                except TimeoutError as ex:
                    raise ex from None
                except BaseException:
                    raise

        _ = self.list_catalogs()  # refresh token
        if "headers" in kwargs:
            kwargs["headers"].update(
                {"authorization": f"bearer {self.credentials.bearer_token}"}
            )
        else:
            kwargs["headers"] = {
                "authorization": f"bearer {self.credentials.bearer_token}",
            }
        if "http" in self.credentials.proxies:
            kwargs["proxy"] = self.credentials.proxies["http"]
        elif "https" in self.credentials.proxies:
            kwargs["proxy"] = self.credentials.proxies["https"]
        th = threading.Thread(target=asyncio.run, args=(async_events(),), daemon=True)
        th.start()
        return None

    def get_events(
        self,
        last_event_id: Optional[str] = None,
        catalog: Optional[str] = None,
        in_background: bool = True,
        url: str = "https://fusion.jpmorgan.com/api/v1/",
    ) -> Union[None, pd.DataFrame]:
        """Run server sent event listener and print out the new events. Keyboard terminate to stop.

        Args:
            last_event_id (str): id of the last event.
            catalog (str): catalog.
            in_background (bool): execute event monitoring in the background (default = True).
            url (str): subscription url.
        Returns:
            Union[None, class:`pandas.DataFrame`]: If in_background is True then the function returns no output.
                If in_background is set to False then pandas DataFrame is output upon keyboard termination.
        """

        catalog = self._use_catalog(catalog)
        if not in_background:
            from sseclient import SSEClient

            _ = self.list_catalogs()  # refresh token
            messages = SSEClient(
                session=self.session,
                url=f"{url}catalogs/{catalog}/notifications/subscribe",
                last_id=last_event_id,
                headers={
                    "authorization": f"bearer {self.credentials.bearer_token}",
                },
            )
            lst = []
            try:
                for msg in messages:
                    event = js.loads(msg.data)
                    if event["type"] != "HeartBeatNotification":
                        lst.append(event)
            except KeyboardInterrupt:
                return pd.DataFrame(lst)
            except Exception as e:
                raise e
            finally:
                return None  # noqa: B012, SIM107
        else:
            return self.events

    def list_dataset_lineage(
        self,
        dataset: str,
        catalog: Optional[str] = None,
        output: bool = False,
        max_results: int = -1,
    ) -> pd.DataFrame:
        """List the upstream and downstream lineage of the dataset.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
        Returns:
            class:`pandas.DataFrame`: A dataframe with a row for each resource
        Raises:
            HTTPError: If the dataset is not found in the catalog.
        """
        catalog = self._use_catalog(catalog)

        url_dataset = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}"
        resp_dataset = self.session.get(url_dataset)
        resp_dataset.raise_for_status()

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}/lineage"
        resp = self.session.get(url)
        data = resp.json()
        relations_data = data["relations"]

        restricted_datasets = [
            dataset_metadata["identifier"]
            for dataset_metadata in data["datasets"]
            if dataset_metadata.get("status", None) == "Restricted"
        ]

        data_dict = {}

        for entry in relations_data:
            source_dataset_id = entry["source"]["dataset"]
            source_catalog = entry["source"]["catalog"]
            destination_dataset_id = entry["destination"]["dataset"]
            destination_catalog = entry["destination"]["catalog"]

            if destination_dataset_id == dataset:
                for ds in data["datasets"]:
                    if ds["identifier"] == source_dataset_id and ds.get("status", None) != "Restricted":
                        source_dataset_title = ds["title"]
                    elif ds["identifier"] == source_dataset_id and ds.get("status", None) == "Restricted":
                        source_dataset_title = "Access Restricted"
                data_dict[source_dataset_id] = ("source", source_catalog, source_dataset_title)

            if source_dataset_id == dataset:
                for ds in data["datasets"]:
                    if ds["identifier"] == destination_dataset_id and ds.get("status", None) != "Restricted":
                        destination_dataset_title = ds["title"]
                    elif ds[
                        "identifier"
                        ] == destination_dataset_id and ds.get("status", None) == "Restricted":
                        destination_dataset_title = "Access Restricted"
                data_dict[destination_dataset_id] = ("produced", destination_catalog, destination_dataset_title)

        output_data = {
            "type": [v[0] for v in data_dict.values()],
            "dataset_identifier": list(data_dict.keys()),
            "title": [v[2] for v in data_dict.values()],
            "catalog": [v[1] for v in data_dict.values()],
        }

        lineage_df = pd.DataFrame(output_data)
        lineage_df.loc[
            lineage_df["dataset_identifier"].isin(restricted_datasets),
            ["dataset_identifier", "catalog", "title"],
        ] = "Access Restricted"

        if max_results > -1:
            lineage_df = lineage_df[0:max_results]

        if output:
            pass

        return lineage_df

    def create_dataset_lineage(
        self,
        base_dataset: str,
        source_dataset_catalog_mapping: Union[pd.DataFrame, list[dict[str, str]]],
        catalog: Optional[str] = None,
        return_resp_obj: bool = False,
    ) -> Optional[requests.Response]:
        """Upload lineage to a dataset.

        Args:
            base_dataset (str): A dataset identifier to which you want to add lineage.
            source_dataset_catalog_mapping (Union[pd.DataFrame, list[dict[str]]]): Mapping for the dataset
                identifier(s) and catalog(s) from which to add lineage.
            catalog (Optional[str], optional): Catalog identifier. Defaults to None.
            return_resp_obj (bool, optional): If True then return the response object. Defaults to False.

        Raises:
            ValueError: If source_dataset_catalog_mapping is not a pandas DataFrame or a list of dictionaries
            HTTPError: If the request is unsuccessful.
        
        Examples:
            Creating lineage from a pandas DataFrame.
            >>> data = [{"dataset": "a", "catalog": "a"}, {"dataset": "b", "catalog": "b"}]
            >>> df = pd.DataFrame(data)
            >>> fusion = Fusion()
            >>> fusion.create_dataset_lineage(base_dataset="c", source_dataset_catalog_mapping=df, catalog="c")

            Creating lineage from a list of dictionaries.
            >>> data = [{"dataset": "a", "catalog": "a"}, {"dataset": "b", "catalog": "b"}]
            >>> fusion = Fusion()
            >>> fusion.create_dataset_lineage(base_dataset="c", source_dataset_catalog_mapping=data, catalog="c")
        """
        catalog = self._use_catalog(catalog)

        if isinstance(source_dataset_catalog_mapping, pd.DataFrame):
            dataset_mapping_list = [
                {
                    "dataset": row["dataset"],
                    "catalog": row["catalog"]
                } for _, row in source_dataset_catalog_mapping.iterrows()
            ]
        elif isinstance(source_dataset_catalog_mapping, list):
            dataset_mapping_list = source_dataset_catalog_mapping
        else:
            raise ValueError("source_dataset_catalog_mapping must be a pandas DataFrame or a list of dictionaries.")
        data = {
            "source": dataset_mapping_list
        }

        url = f"{self.root_url}catalogs/{catalog}/datasets/{base_dataset}/lineage"

        resp = self.session.post(url, json=data)

        resp.raise_for_status()

        return resp if return_resp_obj else None
    
    def create_product(
            self,
            product_obj: Union[str, dict[str, Any], Product, pd.Series],
            catalog: Optional[str] = None,
        ) -> requests.Response:
            """Create a new product in the catalog.

            Args:
                product_obj (Union[str, dict, Product, pd.Series]): A Product object, a dictionary, or a pandas Series
                    containing the product metadata.
                catalog (str, optional): A catalog identifier. Defaults to None.

            Returns:
                requests.Response: The response object from the API call.
            """
            catalog = self._use_catalog(catalog)

            if isinstance(product_obj, (str, dict, pd.Series, Product)):
                product = Product.from_object(product_obj)
            else:
                raise ValueError("product_obj must be a Product object, a dictionary, a path to a csv, a json, or a pandas Series.")
            
            releaseDate = product.releaseDate if product.releaseDate else pd.Timestamp("today").strftime("%Y-%m-%d")
            deliveryChannel = product.deliveryChannel if product.deliveryChannel else ["API"]

            product.releaseDate = releaseDate
            product.deliveryChannel = deliveryChannel

            data = product.to_dict()

            url = f"{self.root_url}catalogs/{catalog}/products/{product.identifier}"
            resp: requests.Response = self.session.post(url, json=data)
            resp.raise_for_status()
            return resp
    
    def update_product(
            self,
            product_obj: Union[str, dict, Product, pd.Series],
            catalog: Optional[str] = None,
        ) -> requests.Response:
            """Update an existing product in the catalog.

            Args:
                product_obj (Union[str, dict, Product, pd.Series]): A Product object, a dictionary, or a pandas Series
                    containing the product metadata.
                catalog (str, optional): A catalog identifier. Defaults to None.

            Returns:
                requests.Response: The response object from the API call.
            """
            catalog = self._use_catalog(catalog)

            if isinstance(product_obj, (str, dict, pd.Series, Product)):
                product = Product.from_object(product_obj)
            else:
                raise ValueError("product_obj must be a Product object, a dictionary, a path to a csv, a json, or a pandas Series.")
            
            releaseDate = product.releaseDate if product.releaseDate else pd.Timestamp("today").strftime("%Y-%m-%d")
            deliveryChannel = product.deliveryChannel if product.deliveryChannel else ["API"]

            product.releaseDate = releaseDate
            product.deliveryChannel = deliveryChannel

            data = product.to_dict()

            url = f"{self.root_url}catalogs/{catalog}/products/{product.identifier}"
            resp: requests.Response = self.session.put(url, json=data)
            resp.raise_for_status()
            return resp
    
    def delete_product(
            self,
            product: str,
            catalog: Optional[str] = None,
        ) -> requests.Response:
            """Delete a product from the catalog.

            Args:
                product (str): The identifier of the product to delete.
                catalog (str, optional): A catalog identifier. Defaults to None.

            Returns:
                requests.Response: The response object from the API call.
            """
            catalog = self._use_catalog(catalog)

            url = f"{self.root_url}catalogs/{catalog}/products/{product}"
            resp: requests.Response = self.session.delete(url)
            resp.raise_for_status()
            return resp
    
    def copy_product(
            self: Fusion,
            product: str,
            catalog_from: str,
            catalog_to: str,
            client_to: Fusion | None = None,
    ) -> requests.Response:
        """Copy product from one catalog and/or environment to another by copy.

        Args:
            product (str): Product  identifier.
            catalog_from (str): Catalog identifer from which to copy product.
            catalog_to (str): Catalog  identifier to wich to copy product.
            client_to (Fusion | None, optional): Fusion client object. Defaults to current instance.

        Returns:
            requests.Response: The response object from the API call.
        """
        if client_to is None:
            client_to = self
        product_obj  =  Product.from_catalog(product=product, catalog=catalog_from, client=self)
        return client_to.create_product(product_obj, catalog=catalog_to)
    
    def list_product_dataset_mapping(
        self,
        dataset: str | list[str] | None = None,
        product: str | list[str] | None = None,
        catalog: str | None = None,
    ) -> pd.DataFrame:
        """get the product to dataset linking contained in  a catalog. A product is a grouping of datasets.

        Args:
            dataset (str | list[str] | None, optional): A string or list of strings that are dataset
            identifiers to filter the output. If a list is provided then it will return
            datasets whose identifier matches any of the strings. Defaults to None.
            product (str | list[str] | None, optional): A string or list of strings that are product
            identifiers to filter the output. If a list is provided then it will return
            products whose identifier matches any of the strings. Defaults to None.
            catalog (str | None, optional): A catalog identifier. Defaults to 'common'.

        Returns:
            pd.DataFrame: a dataframe with a row  for each dataset to product mapping.
        """
        catalog = self._use_catalog(catalog)
        url = f"{self.root_url}catalogs/{catalog}/productDatasets"
        mapping_df  = pd.DataFrame(self._call_for_dataframe(url, self.session))

        if dataset:
            if isinstance(dataset, list):
                contains = "|".join(f"{s}" for s in dataset)
                mapping_df = mapping_df[mapping_df["dataset"].str.contains(contains, case=False)]
            if isinstance(dataset, str):
                mapping_df = mapping_df[mapping_df["dataset"].str.contains(dataset, case=False)]
        if product:
            if isinstance(product, list):
                contains = "|".join(f"{s}" for s in product)
                mapping_df = mapping_df[mapping_df["product"].str.contains(contains, case=False)]
            if isinstance(product, str):
                mapping_df = mapping_df[mapping_df["product"].str.contains(product, case=False)]
        return mapping_df

    def create_dataset(
        self,
        dataset_obj: str | dict[str, Any] | Dataset | pd.Series,
        catalog: str | None = None,
    ) -> requests.Response:
        """Uploada dataset via API from a Dataset object.

        Args:
            dataset_obj (str | dict[str, Any] | Dataset | pd.Series): Dataset metadata.
            catalog (str | None, optional): A catalog identifier. Defaults to "common".

        Returns:
            requests.Response: Request response.
        """
        catalog = self._use_catalog(catalog)

        if isinstance(dataset_obj, (str, dict, pd.Series, Dataset)):
            dataset = Dataset.from_object(dataset_obj)
        else:
            raise ValueError("Dataset object must be a dictionary, path to a csv, or a Fusion Dataset object")
        
        dataset.createdDate = dataset.createdDate if dataset.createdDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        dataset.modifiedDate = dataset.modifiedDate if dataset.modifiedDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        data = dataset.to_dict()

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset.identifier}"
        resp: requests.Response = self.session.post(url, json=data)
        return resp
    
    def update_dataset(
        self,
        dataset_obj: str | dict[str, Any] | Dataset | pd.Series,
        catalog: str | None = None,
    ) -> requests.Response:
        """Updates a dataset via API from dataset object.

        Args:
            dataset_obj (str | dict[str, Any] | Dataset | pd.Series): Dataset metadata.
            catalog (str | None, optional): A catalog identifier. Defaults to "common".

        Returns:
            requests.Response: Request response.
        """
        catalog = self._use_catalog(catalog)

        if isinstance(dataset_obj, (str, dict, pd.Series, Dataset)):
            dataset = Dataset.from_object(dataset_obj)
        else:
            raise ValueError("Dataset object must be a dictionary, path to a csv, or a Fusion Dataset object")
        
        dataset.createdDate = dataset.createdDate if dataset.createdDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        dataset.modifiedDate = dataset.modifiedDate if dataset.modifiedDate else pd.Timestamp("today").strftime("%Y-%m-%d")

        data = dataset.to_dict()

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset.identifier}"
        resp: requests.Response = self.session.put(url, json=data)
        return resp
    
    def delete_dataset(
        self,
        dataset: str,
        catalog: str | None = None,
    ) -> requests.Response:
        """Delete a dataset via API from its dataset identifier.

        Args:
            dataset (str): A dataset identifier.
            catalog (str | None, optional): A catalog identifier. Defaults to 'common'.

        Returns:
            requests.Response: Request response.
        """
        catalog = self._use_catalog(catalog)

        url = f"{self.root_url}catalogs/{catalog}/datasets/{dataset}"
        resp: requests.Response = self.session.delete(url)
        return resp
    
    def copy_dataset(
        self,
        dataset: str,
        catalog_from: str,
        catalog_to: str,
        client_to: Fusion | None = None,
    ) -> requests.Response:
        """Copy dataset and its attributes from one catalog and/or environment to another by copy.

        Args:
            dataset (str): A dataset identifier.
            catalog_from (str): A catalog identifier from which to copy dataset.
            catalog_to (str): A catalog identifier to which to copy dataset.
            client_to (Fusion | None, optional): Fusion client object. Defaults to current instance.

        Returns:
            list[requests.Response]: Request response.
        """
        if client_to is None:
            client_to = self
        dataset_obj = Dataset.from_catalog(dataset=dataset, catalog=catalog_from, client=self)
        return client_to.create_dataset(dataset_obj, catalog=catalog_to)

@dataclass
class Product:
    "Product class."

    title: str
    identifier: str
    category: str | list[str] | None = None
    shortAbstract: str = ""
    description: str = ""
    isActive: bool = True
    isRestricted: bool | None = None
    maintainer: str | list[str] | None = None
    region: str | list[str] | None = None
    publisher: str | None = None
    subCategory: str | list[str] | None = None
    tag: str | list[str] | None = None
    deliveryChannel: str | list[str] = field(default_factory=lambda: ["API"])
    theme:  str | None = None
    releaseDate: str | None = None
    language: str = "English"
    status: str = "Available"
    image: str = ""
    logo: str = ""
    dataset: str | list[str] | None = None

    def  __repr__(self: Product) -> str:
        """Return a string representation of the Product object."""
        return (
            f"Product(\n"
            f"  title={self.title!r},\n"
            f"  identifier={self.identifier!r},\n"
            f"  category={self.category!r},\n"
            f"  shortAbstract={self.shortAbstract!r},\n"
            f"  description={self.description!r},\n"
            f"  isActive={self.isActive!r},\n"
            f"  isRestricted={self.isRestricted!r},\n"
            f"  maintainer={self.maintainer!r},\n"
            f"  region={self.region!r},\n"
            f"  publisher={self.publisher!r},\n"
            f"  subCategory={self.subCategory!r},\n"
            f"  tag={self.tag!r},\n"
            f"  deliveryChannel={self.deliveryChannel!r},\n"
            f"  theme={self.theme!r},\n"
            f"  releaseDate={self.releaseDate!r},\n"
            f"  language={self.language!r},\n"
            f"  status={self.status!r},\n"
            f"  image={self.image!r},\n"
            f"  logo={self.logo!r},\n"
            f"  dataset={self.dataset!r}\n"
            f")"
    )

    def __post_init__(self: Product) -> None:
        """Format Product metadata fields after object instantiation."""
        self.title = tidy_string(self.title)
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.shortAbstract = tidy_string(self.shortAbstract)
        self.description = tidy_string(self.description)
        self.category  = (
            self.category if isinstance(self.category, list) or  self.category is None else make_list(self.category)
        )
        self.tag = (
            self.tag if isinstance(self.tag, list) or self.tag is None else make_list(self.tag)
        )
        self.dataset = (
            self.dataset if isinstance(self.dataset, list) or self.dataset is None else make_list(self.dataset)
        )
        self.subCategory = (
            self.subCategory if isinstance(
                self.subCategory, list
            ) or self.subCategory is None else make_list(self.subCategory)
        )
        self.isActive = self.isActive if isinstance(self.isActive, bool) else make_bool(self.isActive)
        self.isRestricted = (
            self.isRestricted if isinstance(
                self.isRestricted, bool
                ) or self.isRestricted is None else make_bool(self.isRestricted)
        )
        self.maintainer = (
            self.maintainer if isinstance(
                self.maintainer, list
            ) or self.maintainer is None else make_list(self.maintainer)
        )
        self.region = (
            self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        )
        self.deliveryChannel = (
            self.deliveryChannel if isinstance(self.deliveryChannel, list) else make_list(self.deliveryChannel)
        )
        self.releaseDate = convert_date_format(self.releaseDate) if self.releaseDate else None

    @classmethod
    def from_series(cls: type[Product], series: pd.Series) -> Product:
        """Create a Product object from a pandas Series."""
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags", "dataset": "datasets"})
        short_abstract = series.get("abstract", "")
        if short_abstract is None:
            short_abstract = series.get("shortabstract", "")
        
        return cls(
            title=series.get("title", None),
            identifier=series.get("identifier", None),
            category=series.get("category", None),
            shortAbstract=short_abstract,
            description=series.get("description", ""),
            theme=series.get("theme", None),
            releaseDate=series.get("releasedate", None),
            isActive=series.get("isactive", True),
            isRestricted=series.get("isrestricted", None),
            maintainer=series.get("maintainer", None),
            region=series.get("region", None),
            publisher=series.get("publisher", None),
            subCategory=series.get("subcategory", None),
            tag=series.get("tags", None),
            deliveryChannel=series.get("deliverychannel", "API"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            dataset=series.get("datasets", None),
        )
    
    @classmethod
    def from_dict(cls: type[Product], data: dict[str, Any]) -> Product:
        """Create a Product object from a dictionary."""
        keys = [f.name for f in fields(cls)]
        data = {k: v for k, v in data.items() if k in keys}
        return cls(**data)

    @classmethod
    def from_csv(cls: type[Product], file_path: str, identifier: str | None = None) -> Product:
        """Create a list of Product objects from a CSV file."""
        data = pd.read_csv(file_path)

        return (
            Product.from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Product.from_series(data.reset_index(drop=True).iloc[0])
        )


    @classmethod
    def from_object(cls: type[Product], product_source: Any) -> Product:
        """Create a Product object from a dictionary."""
        if isinstance(product_source, Product):
            return product_source
        if isinstance(product_source, dict):
            return Product.from_dict(product_source)
        if isinstance(product_source, str):
            if _is_json(product_source):
                return Product.from_dict(js.loads(product_source))
            return Product.from_csv(product_source)
        if isinstance(product_source, pd.Series):
            return Product.from_series(product_source)
        
        raise TypeError(f"Could not resolve the object provided: {product_source}")

    @classmethod
    def from_catalog(cls: type[Product], client: Fusion, product: str, catalog: str) -> Product:
        """Create a Product object from a catalog."""
        list_products = client.session.get(f"{client.root_url}catalogs/{catalog}/products").json()["resources"]
        dict_ = [dict_ for dict_ in list_products if dict_["identifier"] == product][0]
        product_obj = Product.from_dict(dict_)

        return product_obj
    
    def to_dict(self: Product) -> dict[str, Any]:
        """Convert the Product object to a dictionary."""
        product_dict = asdict(self)
        return product_dict


@dataclass
class Dataset:
    """Dataset class."""

    title: str
    identifier: str
    category: str | list[str] | None = None
    description: str = ""
    frequency: str = "Once"
    isInternalOnlyDataset: bool = False
    isThirdPartyData: bool = True
    isRestricted: bool | None = None
    isRawData: bool = False
    maintainer: str | None = "J.P. Morgan Fusion"
    source: str | list[str] | None = None
    region: str | list[str] | None = None
    publisher: str = "J.P. Morgan"
    product:  str | list[str] | None = None
    subCategory: str | list[str] | None = None
    tags: str | list[str] | None = None
    createdDate: str | None = None
    modifiedDate: str | None = None
    deliveryChannel: str | list[str] = field(default_factory=lambda: ["API"])
    language: str = "English"
    status: str = "Available"
    type_: str | None = "Source"
    containerType: str | None = "Snapshot-Full"
    snowflake: str | None = None
    complexity: str | None = None
    isImmutable: bool | None = None
    isMnpi: bool | None = None
    isPci: bool | None = None
    isPii: bool | None = None
    isClient: bool | None = None
    isPublic: bool | None = None
    isInternal: bool | None = None
    isConfidential: bool | None = None
    isHighlyConfidential: bool | None = None
    isActive: bool | None = None

    def  __repr__(self: Dataset) -> str:
        """Format object representation."""
        return (
            f"Dataset(\n"
            f"  title={self.title!r},\n"
            f"  identifier={self.identifier!r},\n"
            f"  category={self.category!r},\n"
            f"  description={self.description!r},\n"
            f"  frequency={self.frequency!r},\n"
            f"  isInternalOnlyDataset={self.isInternalOnlyDataset!r},\n"
            f"  isThirdPartyData={self.isThirdPartyData!r},\n"
            f"  isRestricted={self.isRestricted!r},\n"
            f"  isRawData={self.isRawData!r},\n"
            f"  maintainer={self.maintainer!r},\n"
            f"  source={self.source!r},\n"
            f"  region={self.region!r},\n"
            f"  publisher={self.publisher!r},\n"
            f"  product={self.product!r},\n"
            f"  subCategory={self.subCategory!r},\n"
            f"  tags={self.tags!r},\n"
            f"  createdDate={self.createdDate!r},\n"
            f"  modifiedDate={self.modifiedDate!r},\n"
            f"  deliveryChannel={self.deliveryChannel!r},\n"
            f"  language={self.language!r},\n"
            f"  status={self.status!r},\n"
            f"  type_={self.type_!r},\n"
            f"  containerType={self.containerType!r},\n"
            f"  snowflake={self.snowflake!r},\n"
            f"  complexity={self.complexity!r},\n"
            f"  isImmutable={self.isImmutable!r},\n"
            f"  isMnpi={self.isMnpi!r},\n"
            f"  isPci={self.isPci!r},\n"
            f"  isPii={self.isPii!r},\n"
            f"  isClient={self.isClient!r},\n"
            f"  isPublic={self.isPublic!r},\n"
            f"  isInternal={self.isInternal!r},\n"
            f"  isConfidential={self.isConfidential!r},\n"
            f"  isHighlyConfidential={self.isHighlyConfidential!r},\n"
            f"  isActive={self.isActive!r}\n"
        )
    
    def __post_init__(self: Dataset) -> None:
        """Format Dataset metadata fields after object initialization."""
        self.identifier = tidy_string(self.identifier).upper().replace(" ", "_")
        self.title = tidy_string(self.title)
        self.description = tidy_string(self.description)
        self.category = (
            self.category if isinstance(self.category, list) or self.category is None else make_list(self.category)
        )
        self.deliveryChannel = (
            self.deliveryChannel if isinstance(self.deliveryChannel, list) else make_list(self.deliveryChannel)
        )
        self.source = (
            self.source if isinstance(self.source, list) or self.source is None else make_list(self.source)
        )
        self.region = (
            self.region if isinstance(self.region, list) or self.region is None else make_list(self.region)
        )
        self.product = (
            self.product if isinstance(self.product, list) or self.product is None else make_list(self.product)
        )
        self.subCategory = (
            self.subCategory if isinstance(self.subCategory, list) or self.subCategory is None else make_list(self.subCategory)
        )
        self.tags = (
            self.tags if isinstance(self.tags, list) or self.tags is None else make_list(self.tags)
        )
        self.isInternalOnlyDataset = self.isInternalOnlyDataset if isinstance(self.isInternalOnlyDataset, bool) else make_bool(self.isInternalOnlyDataset)
        self.createdDate = convert_date_format(self.createdDate) if self.createdDate else None
        self.modifiedDate = convert_date_format(self.modifiedDate) if self.modifiedDate else None

    @classmethod
    def from_series(cls: type[Dataset], series: pd.Series) -> Dataset:
        """Create a Dataset object from a pandas Series."""
        series = series.rename(lambda x: x.replace(" ", "").replace("_", "").lower())
        series = series.rename({"tag": "tags"})
        series = series.rename({"type_": "type"})
        series = series.rename({"productId": "product"})

        isInternalOnlyDataset = series.get("isinternalonlydataset", None)
        isInternalOnlyDataset = make_bool(isInternalOnlyDataset) if isInternalOnlyDataset is not None else isInternalOnlyDataset
        isRestricted = series.get("isrestricted", None)
        isRestricted = make_bool(isRestricted) if isRestricted is not None else isRestricted
        isImmutable = series.get("isimmutable", None)
        isImmutable = make_bool(isImmutable) if isImmutable is not None else isImmutable
        isMnpi = series.get("ismnpi", None)
        isMnpi = make_bool(isMnpi) if isMnpi is not None else isMnpi
        isPci = series.get("ispci", None)
        isPci = make_bool(isPci) if isPci is not None else isPci
        isPii = series.get("ispii", None)
        isPii = make_bool(isPii) if isPii is not None else isPii
        isClient = series.get("isclient", None)
        isClient = make_bool(isClient) if isClient is not None else isClient
        isPublic = series.get("ispublic", None)
        isPublic = make_bool(isPublic) if isPublic is not None else isPublic
        isInternal = series.get("isinternal", None)
        isInternal = make_bool(isInternal) if isInternal is not None else isInternal
        isConfidential = series.get("isconfidential", None)
        isConfidential = make_bool(isConfidential) if isConfidential is not None else isConfidential
        isHighlyConfidential = series.get("ishighlyconfidential", None)
        isHighlyConfidential = make_bool(isHighlyConfidential) if isHighlyConfidential is not None else isHighlyConfidential
        isActive = series.get("isactive", None)
        isActive = make_bool(isActive) if isActive is not None else isActive

        dataset = cls(
            identifier=series.get("identifier", None),
            category=series.get("category", None),
            deliveryChannel=series.get("deliverychannel", ["API"]),
            title=series.get("title", None),
            description=series.get("description", ""),
            frequency=series.get("frequency", "Once"),
            isInternalOnlyDataset=isInternalOnlyDataset,
            isThirdPartyData=series.get("isthirdpartydata", True),
            isRestricted=isRestricted,
            isRawData=series.get("israwdata", False),
            maintainer=series.get("maintainer", "J.P. Morgan Fusion"),
            source=series.get("source", None),
            region=series.get("region", None),
            publisher=series.get("publisher", "J.P. Morgan"),
            product=series.get("product", None),
            subCategory=series.get("subcategory", None),
            tags=series.get("tags", None),
            containerType=series.get("containertype", "Snapshot-Full"),
            language=series.get("language", "English"),
            status=series.get("status", "Available"),
            type_=series.get("type", "Source"),
            createdDate=series.get("createddate", None),
            modifiedDate=series.get("modifieddate", None),
            snowflake=series.get("snowflake", None),
            complexity=series.get("complexity", None),
            isImmutable=isImmutable,
            isMnpi=isMnpi,
            isPci=isPci,
            isPii=isPii,
            isClient=isClient,
            isPublic=isPublic,
            isInternal=isInternal,
            isConfidential=isConfidential,
            isHighlyConfidential=isHighlyConfidential,
            isActive=isActive,
        )
        return dataset

    @classmethod
    def from_dict(cls: type[Dataset], data: dict[str, Any]) -> Dataset:
        """Create a Dataset object from a dictionary."""
        keys = [f.name for f in fields(cls)]
        keys = ["type" if key == "type_" else key for key in keys]
        data = {k: v for k, v in data.items() if k in keys}
        if "type" in data:
            data["type_"] = data.pop("type")
        return cls(**data)

    @classmethod
    def from_csv(cls: type[Dataset], file_path: str, identifier: str | None = None) -> Dataset:
        """Create a list of Dataset objects from a CSV file."""
        data = pd.read_csv(file_path)

        return (
            Dataset.from_series(data[data["identifier"] == identifier].reset_index(drop=True).iloc[0])
            if identifier
            else Dataset.from_series(data.reset_index(drop=True).iloc[0])
        )
    
    @classmethod
    def from_object(
        cls: type[Dataset],
        dataset_source: Dataset | dict[str, Any] | str | pd.Series
    ) -> Dataset:
        """Create a Dataset object from a dictionary."""
        if isinstance(dataset_source, Dataset):
            return dataset_source
        if isinstance(dataset_source, dict):
            return Dataset.from_dict(dataset_source)
        if isinstance(dataset_source, str):
            if _is_json(dataset_source):
                return Dataset.from_dict(js.loads(dataset_source))
            return Dataset.from_csv(dataset_source)
        if isinstance(dataset_source, pd.Series):
            return Dataset.from_series(dataset_source)
        
        raise TypeError(f"Could not resolve the object provided: {dataset_source}")
    
    @classmethod
    def from_catalog(
        cls: type[Dataset],
        client: Fusion,
        dataset: str,
        catalog: str
    ) -> Dataset:
        """Create a Dataset object from a catalog."""
        list_datasets = client.session.get(f"{client.root_url}catalogs/{catalog}/datasets").json()["resources"]
        dict_ = [dict_ for dict_ in list_datasets if dict_["identifier"] == dataset][0]
        dataset_obj = Dataset.from_dict(dict_)

        prod_df = client.list_product_dataset_mapping(catalog=catalog)

        if dataset.lower() in list(prod_df.dataset.str.lower()):
            product = [prod_df[prod_df["dataset"].str.lower() == dataset.lower()]["product"].iloc[0]]
            dataset_obj.product = product

        return dataset_obj
    
    def to_dict(self) -> dict[str, Any]:
        """Convert the Dataset object to a dictionary."""
        dataset_dict = asdict(self)
        dataset_dict["type"] = dataset_dict.pop("type_")
        return dataset_dict
