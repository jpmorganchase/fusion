"""Main Fusion module."""

import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
import requests
from joblib import Parallel, delayed
from tabulate import tabulate
from tqdm import tqdm

from .authentication import FusionCredentials, get_default_fs
from .exceptions import APIResponseError
from .fusion_filesystem import FusionHTTPFileSystem
from .utils import (
    cpu_count,
    distribution_to_filename,
    distribution_to_url,
    get_session,
    normalise_dt_param_str,
    read_csv,
    read_parquet,
    stream_single_file_new_session,
    validate_file_names,
    path_to_url,
    upload_files
)

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
        table = response.json()['resources']
        df = pd.DataFrame(table).reset_index(drop=True)
        return df

    def __init__(
        self,
        credentials: Union[str, dict] = 'config/client_credentials.json',
        root_url: str = "https://fusion-api.jpmorgan.com/fusion/v1/",
        download_folder: str = "downloads",
        log_level: int = logging.ERROR,
        fs=None,
    ) -> None:
        """Constructor to instantiate a new Fusion object.

        Args:
            credentials (Union[str, dict], optional): A path to a credentials file or
                a dictionary containing the required keys.
                Defaults to 'config/client_credentials.json'.
            root_url (_type_, optional): The API root URL.
                Defaults to "https://fusion-api.jpmorgan.com/fusion/v1/".
            download_folder (str, optional): The folder path where downloaded data files
                are saved. Defaults to "downloads".
            log_level (int, optional): Set the logging level. Defaults to logging.ERROR.
            fs (fsspec.filesystem): filesystem.
        """
        self._default_catalog = "common"

        self.root_url = root_url
        self.download_folder = download_folder
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        if logger.hasHandlers():
            logger.handlers.clear()
        file_handler = logging.FileHandler(filename="fusion_sdk.log")
        logging.addLevelName(VERBOSE_LVL, "VERBOSE")
        stdout_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d %(name)s:%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        logger.setLevel(log_level)

        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        else:
            self.credentials = FusionCredentials.from_object(credentials)

        self.session = get_session(self.credentials, self.root_url)
        self.fs = fs if fs else get_default_fs()

    def __repr__(self):
        """Object representation to list all available methods."""
        return "Fusion object \nAvailable methods:\n" + tabulate(
            pd.DataFrame(
                [
                    [
                        method_name
                        for method_name in dir(Fusion)
                        if callable(getattr(Fusion, method_name)) and not method_name.startswith("_")
                    ]
                    + [p for p in dir(Fusion) if isinstance(getattr(Fusion, p), property)]
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
    def default_catalog(self, catalog) -> None:
        """Allow the default catalog, which is "common" to be overridden.

        Args:
            catalog (str): The catalog to use as the default

        Returns:
            None
        """
        self._default_catalog = catalog


    def __use_catalog(self, catalog):
        """Determine which catalog to use in an API call.

        Args:
            catalog (str): The catalog value passed as an argument to an API function wrapper.

        Returns:
            str: The catalog to use
        """
        if catalog is None:
            return self.default_catalog
        else:
            return catalog

    def get_fusion_filesystem(self):
        """Creates Fusion Filesystem.

        Returns: Fusion Filesystem

        """
        return FusionHTTPFileSystem(client_kwargs={"root_url": self.root_url, "credentials": self.credentials})

    def list_catalogs(self, output: bool = False) -> pd.DataFrame:
        """Lists the catalogs available to the API account.

        Args:
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each catalog
        """
        url = f'{self.root_url}catalogs/'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def catalog_resources(self, catalog: str = None, output: bool = False) -> pd.DataFrame:
        """List the resources contained within the catalog, for example products and datasets.

        Args:
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
           pandas.DataFrame: A dataframe with a row for each resource within the catalog
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql", maxcolwidths=30))

        return df

    def list_products(
        self,
        contains: Union[str, list] = None,
        id_contains: bool = False,
        catalog: str = None,
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
            pandas.DataFrame: a dataframe with a row for each product
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/products'
        df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f'{s}' for s in contains)
            if id_contains:
                df = df[df['identifier'].str.contains(contains, case=False)]
            else:
                df = df[
                    df['identifier'].str.contains(contains, case=False)
                    | df['description'].str.contains(contains, case=False)
                ]

        df["category"] = df.category.str.join(", ")
        df["region"] = df.region.str.join(", ")
        if not display_all_columns:
            df = df[["identifier", "title", "region", "category", "status", "description"]]

        if max_results > -1:
            df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql", maxcolwidths=30))

        return df

    def list_datasets(
        self,
        contains: Union[str, list] = None,
        id_contains: bool = False,
        catalog: str = None,
        output: bool = False,
        max_results: int = -1,
        display_all_columns: bool = False,
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            contains (Union[str, list], optional): A string or a list of strings that are dataset
                identifiers to filter the datasets list. If a list is provided then it will return
                datasets whose identifier matches any of the strings. Defaults to None.
            id_contains (bool): Filter datasets only where the string(s) are contained in the identifier,
                ignoring description.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            pandas.DataFrame: a dataframe with a row for each dataset.
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets'
        df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f'{s}' for s in contains)
            if id_contains:
                df = df[df['identifier'].str.contains(contains, case=False)]
            else:
                df = df[
                    df['identifier'].str.contains(contains, case=False)
                    | df['description'].str.contains(contains, case=False)
                ]

        if max_results > -1:
            df = df[0:max_results]

        df["category"] = df.category.str.join(", ")
        df["region"] = df.region.str.join(", ")
        if not display_all_columns:
            df = df[
                ["identifier", "title", "region", "category", "coverageStartDate", "coverageEndDate", "description"]
            ]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql", maxcolwidths=30))

        return df

    def dataset_resources(self, dataset: str, catalog: str = None, output: bool = False) -> pd.DataFrame:
        """List the resources available for a dataset, currently this will always be a datasetseries.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each resource
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_dataset_attributes(self, dataset: str, catalog: str = None, output: bool = False, display_all_columns: bool = False) -> pd.DataFrame:
        """Returns the list of attributes that are in the dataset.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            display_all_columns (bool, optional): If True displays all columns returned by the API,
                otherwise only the key columns are displayed

        Returns:
            pandas.DataFrame: A dataframe with a row for each attribute
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/attributes'
        df = Fusion._call_for_dataframe(url, self.session)

        if not display_all_columns:
            df = df[["identifier", "dataType", "isDatasetKey", "description"]]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql", maxcolwidths=30))

        return df

    def list_datasetmembers(
        self, dataset: str, catalog: str = None, output: bool = False, max_results: int = -1
    ) -> pd.DataFrame:
        """List the available members in the dataset series.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.
            max_results (int, optional): Limit the number of rows returned in the dataframe.
                Defaults to -1 which returns all results.

        Returns:
            pandas.DataFrame: a dataframe with a row for each dataset member.
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries'
        df = Fusion._call_for_dataframe(url, self.session)

        if max_results > -1:
            df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def datasetmember_resources(
        self, dataset: str, series: str, catalog: str = None, output: bool = False
    ) -> pd.DataFrame:
        """List the available resources for a datasetseries member.

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each datasetseries member resource.
                Currently, this will always be distributions.
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_distributions(self, dataset: str, series: str, catalog: str = None, output: bool = False) -> pd.DataFrame:
        """List the available distributions (downloadable instances of the dataset with a format type).

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each distribution.
        """
        catalog = self.__use_catalog(catalog)

        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def _resolve_distro_tuples(
        self, dataset: str, dt_str: str = 'latest', dataset_format: str = 'parquet', catalog: str = None
    ):
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
        catalog = self.__use_catalog(catalog)

        datasetseries_list = self.list_datasetmembers(dataset, catalog)

        if datasetseries_list.empty:
            raise APIResponseError(
                f'No data available for dataset {dataset}. '
                f'Check that a valid dataset identifier and date/date range has been set.'
            )

        if dt_str == 'latest':
            dt_str = datasetseries_list.iloc[datasetseries_list['createdDate'].values.argmax()]['identifier']

        parsed_dates = normalise_dt_param_str(dt_str)
        if len(parsed_dates) == 1:
            parsed_dates = (parsed_dates[0], parsed_dates[0])

        if parsed_dates[0]:
            datasetseries_list = datasetseries_list[datasetseries_list['fromDate'] >= parsed_dates[0]]

        if parsed_dates[1]:
            datasetseries_list = datasetseries_list[datasetseries_list['toDate'] <= parsed_dates[1]]

        required_series = list(datasetseries_list['@id'])
        tups = [(catalog, dataset, series, dataset_format) for series in required_series]

        return tups

    def download(
        self,
        dataset: str,
        dt_str: str = 'latest',
        dataset_format: str = 'parquet',
        catalog: str = None,
        n_par: int = None,
        show_progress: bool = True,
        force_download: bool = False,
        download_folder: str = None,
        return_paths: bool = False,
    ):
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

        Returns:

        """
        catalog = self.__use_catalog(catalog)

        n_par = cpu_count(n_par)
        required_series = self._resolve_distro_tuples(dataset, dt_str, dataset_format, catalog)

        if not download_folder:
            download_folder = self.download_folder

        if not self.fs.exists(download_folder):
            self.fs.mkdir(download_folder, create_parents=True)
        download_spec = [
            {
                "credentials": self.credentials,
                "url": distribution_to_url(self.root_url, series[1], series[2], series[3], series[0]),
                "output_file": distribution_to_filename(download_folder, series[1], series[2], series[3], series[0]),
                "overwrite": force_download,
                "fs": self.fs
            }
            for series in required_series
        ]

        if show_progress:
            loop = tqdm(download_spec)
        else:
            loop = download_spec
        logger.log(
            VERBOSE_LVL,
            f'Beginning {len(loop)} downloads in batches of {n_par}',
        )
        res = Parallel(n_jobs=n_par)(delayed(stream_single_file_new_session)(**spec) for spec in loop)

        return res if return_paths else None

    def to_df(
        self,
        dataset: str,
        dt_str: str = 'latest',
        dataset_format: str = 'parquet',
        catalog: str = None,
        n_par: int = None,
        show_progress: bool = True,
        columns: List = None,
        filters: List = None,
        force_download: bool = False,
        download_folder: str = None,
        **kwargs,
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
        Returns:
            pandas.DataFrame: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        catalog = self.__use_catalog(catalog)

        n_par = cpu_count(n_par)
        if not download_folder:
            download_folder = self.download_folder
        download_res = self.download(
            dataset, dt_str, dataset_format, catalog, n_par, show_progress, force_download, download_folder, return_paths=True
        )

        if not all(res[0] for res in download_res):
            failed_res = [res for res in download_res if not res[0]]
            raise Exception(
                f"Not all downloads were successfully completed. "
                f"Re-run to collect missing files. The following failed:\n{failed_res}"
            )

        files = [res[1] for res in download_res]

        pd_read_fn_map = {
            'csv': read_csv,
            'parquet': read_parquet,
            'parq': read_parquet,
            'json': pd.read_json,
        }

        pd_read_default_kwargs: Dict[str, Dict[str, object]] = {
            'csv': {'columns': columns, 'filters': filters},
            'parquet': {'columns': columns, 'filters': filters},
            'json': {'columns': columns, 'filters': filters},
        }

        pd_read_default_kwargs['parq'] = pd_read_default_kwargs['parquet']

        pd_reader = pd_read_fn_map.get(dataset_format)
        pd_read_kwargs = pd_read_default_kwargs.get(dataset_format, {})
        if not pd_reader:
            raise Exception(f'No pandas function to read file in format {dataset_format}')

        pd_read_kwargs.update(kwargs)

        if len(files) == 0:
            raise APIResponseError(
                f"No series members for dataset: {dataset} "
                f"in date or date range: {dt_str} and format: {dataset_format}"
            )
        if dataset_format in ["parquet", "parq"]:
            df = pd_reader(files, **pd_read_kwargs)
        else:
            dataframes = (pd_reader(f, **pd_read_kwargs) for f in files)
            df = pd.concat(dataframes, ignore_index=True)

        return df

    def upload(self,
               path: str,
               dataset: str = None,
               dt_str: str = 'latest',
               catalog: str = None,
               n_par: int = None,
               show_progress: bool = True,
               return_paths: bool = False,
               ):
        """Uploads the requested files/files to Fusion.

        Args:
            path (str): path to a file or a folder with files
            dataset (str, optional): Dataset name to which the file will be uplaoded (for single file only).
                                    If not provided the dataset will be implied from file's name.
            dt_str (str, optional): A single date. Defaults to 'latest' which will return the most recent.
                                    Relevant for a single file upload only. If not provided the dataset will
                                    be implied from file's name.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to all cpus available.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            return_paths (bool, optional): Return paths and success statuses of the downloaded files.

        Returns:


        """
        catalog = self.__use_catalog(catalog)

        if not self.fs.exists(path):
            raise RuntimeError("The provided path does not exist")

        fs_fusion = self.get_fusion_filesystem()
        if self.fs.info(path)["type"] == "directory":
            file_path_lst = self.fs.find(path)
            local_file_validation = validate_file_names(file_path_lst, fs_fusion)
            file_path_lst = [f for flag, f in zip(local_file_validation, file_path_lst) if flag]
            local_url_eqiv = [path_to_url(i) for i in file_path_lst]
        else:
            file_path_lst = [path]
            if not catalog or not dataset:
                local_file_validation = validate_file_names(file_path_lst, fs_fusion)
                file_path_lst = [f for flag, f in zip(local_file_validation, file_path_lst) if flag]
                local_url_eqiv = [path_to_url(i) for i in file_path_lst]
            else:
                file_format = path.split(".")[-1]
                dt_str = dt_str if dt_str != "latest" else pd.Timestamp("today").date().strftime("%Y%m%d")
                dt_str = pd.Timestamp(dt_str).date().strftime("%Y%m%d")
                if catalog not in fs_fusion.ls('') or dataset not in [i.split('/')[-1] for i in
                                                                      fs_fusion.ls(f"{catalog}/datasets")]:
                    msg = f"File file has not been uploaded, one of the catalog: {catalog} " \
                          f"or dataset: {dataset} does not exit."
                    warnings.warn(msg)
                    return [(False, path, Exception(msg))]
                local_url_eqiv = [path_to_url(f"{dataset}__{catalog}__{dt_str}.{file_format}")]

        df = pd.DataFrame([file_path_lst, local_url_eqiv]).T
        df.columns = ["path", "url"]

        if show_progress:
            loop = tqdm(df.iterrows(), total=len(df))
        else:
            loop = df.iterrows()

        n_par = cpu_count(n_par)
        parallel = True if len(df) > 1 else False
        res = upload_files(fs_fusion, self.fs, loop, parallel=parallel, n_par=n_par)

        if not all(r[0] for r in res):
            failed_res = [r for r in res if not r[0]]
            msg = f"Not all uploads were successfully completed. The following failed:\n{failed_res}"
            logger.warning(msg)
            warnings.warn(msg)

        return res if return_paths else None
