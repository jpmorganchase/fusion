"""Main Fusion module."""

import datetime
import json
import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from joblib import Parallel, delayed
from tabulate import tabulate
from tqdm import tqdm

DT_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
DT_YYYY_MM_DD_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
OAUTH_GRANT_TYPE = 'client_credentials'
DEFAULT_CHUNK_SIZE = 2**16
DEFAULT_PARALLELISM = 5
HTTP_SUCCESS = 200
USER_AGENT = 'Mozilla/5.0'
CONTENT_TYPE = 'application/x-www-form-urlencoded'

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


class APIRequestError(Exception):
    """APIRequestError exception wrapper.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIResponseError(Exception):
    """APIResponseError exception wrapper.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIConnectError(Exception):
    """APIConnectError exception wrapper.

    Args:
        Exception : Exception to wrap.
    """

    pass


class UnrecognizedFormatError(Exception):
    """UnrecognizedFormatError exception wrapper.

    Args:
        Exception : Exception to wrap.
    """

    pass


class CredentialError(Exception):
    """CredentialError exception wrapper.

    Args:
        Exception : Exception to wrap.
    """

    pass


def _res_plural(ref_int: int, pluraliser: str = 's') -> str:
    return '' if abs(ref_int) == 1 else pluraliser


def _is_json(data) -> bool:
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def _normalise_dt_param(dt: Union[str, int, datetime.datetime, datetime.date]) -> str:

    if isinstance(dt, (datetime.date, datetime.datetime)):
        return dt.strftime("%Y-%m-%d")

    if isinstance(dt, int):
        dt = str(dt)

    matches = DT_YYYYMMDD_RE.match(dt)

    if matches:
        return "-".join(matches.groups())

    raise ValueError(f"{dt} is not in a recognised data format")


def _normalise_dt_param_str(dt: str) -> tuple:

    date_parts = dt.split(":")

    if not date_parts or len(date_parts) > 2:
        raise ValueError(f"Unable to parse {dt} as either a date or an interval")

    return tuple((_normalise_dt_param(dt_part) if dt_part else dt_part for dt_part in date_parts))


def _distribution_to_filename(
    root_folder: str, dataset: str, datasetseries: str, file_format: str, catalog: str = 'common'
) -> Path:
    if datasetseries[-1] == '/' or datasetseries[-1] == '\\':
        datasetseries = datasetseries[0:-1]
    file_name = f"{dataset}__{catalog}__{datasetseries}.{file_format}"
    return Path(root_folder, file_name)


def _filename_to_distribution(file_name: str) -> tuple:
    dataset, catalog, series_format = Path(file_name).name.split('__')
    datasetseries, file_format = series_format.split('.')
    return (catalog, dataset, datasetseries, file_format)


def _distribution_to_url(
    root_url: str, dataset: str, datasetseries: str, file_format: str, catalog: str = 'common'
) -> str:

    if datasetseries[-1] == '/' or datasetseries[-1] == '\\':
        datasetseries = datasetseries[0:-1]

    return f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries}/distributions/{file_format}"


class FusionCredentials:
    """Class to manage Fusion Creds and OAuth."""

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        resource: str = None,
        auth_url: str = None,
        proxies: str = None,
    ) -> None:
        """Constuctor for Creds mgr.

        Args:
            client_id (str, optional): Client ID as provided by Fusion. Defaults to None.
            client_secret (str, optional): Client Secret as provided by Fusion. Defaults to None.
            resource (str, optional): Fusion resource ID as provided by Fusion. Defaults to None.
            auth_url (str, optional): Auth URL as provided by Fusion. Defaults to None.
            proxies (str, optional): Any proxy servers to hop through. Defaults to None.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource = resource
        self.auth_url = auth_url
        self.proxies = proxies

    @staticmethod
    def generate_credentials_file(
        credentials_file: str = 'config/client_credentials.json',
        client_id: str = None,
        client_secret: str = None,
        resource: str = None,
        auth_url: str = None,
        proxies: str = None,
    ):
        """_summary_.

        Args:
            credentials_file (str, optional): _description_. Defaults to 'config/client_credentials.json'.
            client_id (str, optional): _description_. Defaults to None.
            client_secret (str, optional): _description_. Defaults to None.
            resource (str, optional): _description_. Defaults to None.
            auth_url (str, optional): _description_. Defaults to None.
            proxies (str, optional): _description_. Defaults to None.

        Raises:
            CredentialError: Exception describing creds issue

        Returns:
            _type_: _description_
        """
        if not client_id:
            raise CredentialError('A valid client_id is required')
        if not client_secret:
            raise CredentialError('A valid client secret is required')
        if not resource:
            raise CredentialError('A valid resource is required')
        if not auth_url:
            raise CredentialError('A valid authentication server URL is required')

        data = dict(
            {'client_id': client_id, 'client_secret': client_secret, 'resource': resource, 'auth_url': auth_url}
        )

        if proxies:
            data['proxies'] = proxies
        json_data = json.dumps(data)

        with open(credentials_file, 'w') as credentialsfile:
            credentialsfile.write(json_data)

        credentials = FusionCredentials(client_id, client_secret, resource, auth_url)
        return credentials

    @staticmethod
    def from_dict(credentials: dict):
        """Create Creds object from dict.

        Args:
            credentials (dict): conforming dictionary with creds attributes

        Returns:
            FusionCredentials: creds object
        """
        client_id = credentials['client_id']
        client_secret = credentials['client_secret']
        resource = credentials['resource']
        auth_url = credentials['auth_url']
        proxies = credentials.get('proxies')
        creds = FusionCredentials(client_id, client_secret, resource, auth_url, proxies)
        return creds

    @staticmethod
    def from_file(credentials_file: str = 'config/client.credentials.json'):
        """_summary_.

        Args:
            credentials_file (str, optional): _description_. Defaults to 'config/client.credentials.json'.

        Returns:
            _type_: _description_
        """
        with open(credentials_file, 'r') as credentials:
            data = json.load(credentials)
            credentials = FusionCredentials.from_dict(data)
            return credentials

    @staticmethod
    def from_object(credentials_source: Union[str, dict]):
        """_summary_.

        Args:
            credentials_source (Union[str, dict]): _description_

        Raises:
            CredentialError: _description_

        Returns:
            _type_: _description_
        """
        if isinstance(credentials_source, dict):
            return FusionCredentials.from_dict(credentials_source)
        elif isinstance(credentials_source, str):
            if _is_json(credentials_source):
                return FusionCredentials.from_dict(json.loads(credentials_source))
            else:
                return FusionCredentials.from_file(credentials_source)

        raise CredentialError(f'Could not resolve the credentials provided: {credentials_source}')


class FusionOAuthAdapter(requests.adapters.HTTPAdapter):
    """Fusion OAuth model specific requests adapter."""

    def __init__(self, credentials, proxies={}, refresh_within_seconds=5, *args, **kwargs) -> None:
        """_summary_.

        Args:
            credentials (_type_): _description_
            proxies (dict, optional): _description_. Defaults to {}.
            refresh_within_seconds (int, optional): _description_. Defaults to 5.
        """
        super(FusionOAuthAdapter, self).__init__(*args, **kwargs)

        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        else:
            self.credentials = FusionCredentials.from_object(credentials)

        if proxies:
            self.proxies = proxies
        else:
            self.proxies = self.credentials.proxies

        self.bearer_token_expiry = datetime.datetime.now()
        self.number_token_refreshes = 0
        self.refresh_within_seconds = refresh_within_seconds

    def send(self, request, **kwargs):
        """_summary_.

        Args:
            request (_type_): _description_
        """

        def _refresh_token_data():
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "aud": self.credentials.resource,
            }

            try:
                response = requests.Session().post(self.credentials.auth_url, data=payload)
                response_data = response.json()
                access_token = response_data["access_token"]
                expiry = response_data["expires_in"]
                return access_token, expiry
            except Exception as ex:
                raise Exception(f'Failed to authenticate against OAuth server {ex}')

        token_expires_in = (self.bearer_token_expiry - datetime.datetime.now()).total_seconds()
        if token_expires_in < self.refresh_within_seconds:
            token, expiry = _refresh_token_data()
            self.token = token
            self.bearer_token_expiry = datetime.datetime.now() + timedelta(seconds=int(expiry))
            self.number_token_refreshes += 1
            logger.log(
                VERBOSE_LVL,
                f'Refreshed token {self.number_token_refreshes} time{_res_plural(self.number_token_refreshes)}',
            )

        request.headers.update({'Authorization': f'Bearer {self.token}', 'jpmc-token-provider': 'authe'})
        response = super(FusionOAuthAdapter, self).send(request, **kwargs)
        return response


def _get_canonical_root_url(any_url: str) -> str:
    url_parts = urlparse(any_url)
    root_url = urlunparse((url_parts[0], url_parts[1], '', '', '', ''))
    return root_url


def _get_session(credentials, root_url):
    session = requests.Session()
    auth_handler = FusionOAuthAdapter(credentials)
    if credentials.proxies:
        session.proxies.update(credentials.proxies)
    try:
        mount_url = _get_canonical_root_url(root_url)
    except Exception:
        mount_url = "https://"
    session.mount(mount_url, auth_handler)
    return session


def _stream_single_file_new_session_dry_run(credentials, url: str, output_file: str):
    try:
        _get_session(credentials, url).head(url)
        return (True, output_file, None)
    except Exception as ex:
        return (False, output_file, ex)


def _stream_single_file_new_session(
    credentials,
    url: str,
    output_file: str,
    overwrite: bool = True,
    block_size=DEFAULT_CHUNK_SIZE,
    dry_run: bool = False,
):
    if dry_run:
        return _stream_single_file_new_session_dry_run(credentials, url, output_file)

    if not overwrite and Path(output_file).exists():
        return (True, output_file, None)

    try:
        with _get_session(credentials, url).get(url, stream=True) as r:
            with open(output_file, "wb") as outfile:
                for chunk in r.iter_content(block_size):
                    outfile.write(chunk)
        return (True, output_file, None)
    except Exception as ex:
        return (False, output_file, ex)


def _stream_single_file(session, url: str, output_file: str, blocl_size=DEFAULT_CHUNK_SIZE):
    with session.get(url, stream=True) as r:
        with open(output_file, "wb") as outfile:
            for chunk in r.iter_content(blocl_size):
                outfile.write(chunk)


class Fusion:
    """Core Fusion class for API access."""

    @staticmethod
    def _call_for_dataframe(url: str, session: requests.Session) -> pd.DataFrame:

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
    ) -> None:
        """_summary_.

        Args:
            credentials (Union[str, dict], optional): _description_. Defaults to 'config/client_credentials.json'.
            root_url (_type_, optional): _description_. Defaults to "https://fusion-api.jpmorgan.com/fusion/v1/".
            download_folder (str, optional): _description_. Defaults to "downloads".
            log_level (int, optional): _description_. Defaults to logging.ERROR.
        """
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

        self.session = _get_session(self.credentials, self.root_url)

    def list_catalogs(self, output: bool = False) -> pd.DataFrame:
        """_summary_.

        Args:
            output (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def catalog_resources(self, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.

        Returns:
           pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_products(
        self, contains: Union[str, list] = None, catalog: str = 'common', output: bool = False, max_results: int = -1
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            contains (Union[str, list], optional): _description_. Defaults to None.
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.
            max_results (int, optional): _description_. Defaults to -1.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/products'
        df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f'{s}' for s in contains)
            df = df[df['identifier'].str.contains(contains)]

        df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_datasets(
        self, contains: Union[str, list] = None, catalog: str = 'common', output: bool = False, max_results: int = -1
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            contains (Union[str, list], optional): _description_. Defaults to None.
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.
            max_results (int, optional): _description_. Defaults to -1.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets'
        df = Fusion._call_for_dataframe(url, self.session)

        if contains:
            if isinstance(contains, list):
                contains = "|".join(f'{s}' for s in contains)
            df = df[df['identifier'].str.contains(contains)]

        df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def dataset_resources(self, dataset: str, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_dataset_attributes(self, dataset: str, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/attributes'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_datasetmembers(
        self, dataset: str, catalog: str = 'common', output: bool = False, max_results: int = -1
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.
            max_results (int, optional): _description_. Defaults to -1.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries'
        df = Fusion._call_for_dataframe(url, self.session)

        df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def datasetmember_resources(
        self, dataset: str, series: str, catalog: str = 'common', output: bool = False
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_distributions(
        self, dataset: str, series: str, catalog: str = 'common', output: bool = False
    ) -> pd.DataFrame:
        """_summary_.

        Args:
            catalog (str, optional): _description_. Defaults to 'common'.
            output (bool, optional): _description_. Defaults to False.

        Returns:
            pd.DataFrame: _description_
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def __resolve_distro_tuples(
        self, dataset: str, dt_str: str = 'latest', format: str = 'parquet', catalog: str = 'common'
    ):
        """_summary_.

        Args:
            dt_str (str, optional): _description_. Defaults to 'latest'.
            format (str, optional): _description_. Defaults to 'parquet'.
            catalog (str, optional): _description_. Defaults to 'common'.

        Returns:
            pd.DataFrame: _description_
        """
        parsed_dates = _normalise_dt_param_str(dt_str)
        if len(parsed_dates) == 1:
            parsed_dates = (parsed_dates[0], parsed_dates[0])

        datasetseries_list = self.list_datasetmembers(dataset, catalog)

        if parsed_dates[0]:
            datasetseries_list = datasetseries_list[datasetseries_list['fromDate'] >= parsed_dates[0]]

        if parsed_dates[1]:
            datasetseries_list = datasetseries_list[datasetseries_list['toDate'] <= parsed_dates[1]]

        required_series = list(datasetseries_list['@id'])
        tups = [(catalog, dataset, series, format) for series in required_series]

        return tups

    def download_distribution(
        self,
        dataset: str,
        dt_str: str = 'latest',
        dataset_format: str = 'parquet',
        catalog: str = 'common',
        n_par: int = DEFAULT_PARALLELISM,
        show_progress: bool = True,
        dry_run: bool = True,
        force_download: bool = True,
    ):
        """_summary_.

        Args:
            dt_str (str, optional): _description_. Defaults to 'latest'.
            dataset_format (str, optional): _description_. Defaults to 'parquet'.
            catalog (str, optional): _description_. Defaults to 'common'.
            n_par (int, optional): _descrition_. Defaults to DEFAULT_PARALLELISM.
            show_progress (bool, optional): _description_. Defaults to True.
            dry_run (bool, optional): _description_. Defaults to True.
            force_download (bool, optional): _description_. Defaults to True

        Returns:

        """
        required_series = self.__resolve_distro_tuples(dataset, dt_str, dataset_format, catalog)

        download_spec = [
            (
                self.credentials,
                _distribution_to_url(self.root_url, series[1], series[2], series[3], series[0]),
                _distribution_to_filename(self.download_folder, series[1], series[2], series[3], series[0]),
                force_download,
            )
            for series in required_series
        ]

        if show_progress:
            loop = tqdm(download_spec)
        else:
            loop = download_spec

        res = Parallel(n_jobs=n_par)(delayed(_stream_single_file_new_session)(*spec) for spec in loop)

        return res

    def get_distribution(
        self,
        dataset: str,
        dt_str: str = 'latest',
        dataset_format: str = 'parquet',
        catalog: str = 'common',
        n_par: int = DEFAULT_PARALLELISM,
        show_progress: bool = True,
        dry_run: bool = True,
        columns: List = None,
        force_download: bool = True,
        **kwargs,
    ):
        """Get distribution.

        Args:
            dt_str (str, optional): _description_. Defaults to 'latest'
            dataset_format (str, optional): _description_. Defaults to 'parquet'
            catalog (str, optional): _description_. Defaults to 'common'
            n_par (int, optional): _descrition_. Defaults to DEFAULT_PARALLELISM
            show_progress (bool, optional): _description_. Defaults to True
            dry_run (bool, optional): _description_. Defaults to True
            columns (List, optional): _description_. Defaults to None
            force_download (bool, optional): _description_. Defaults to True

        Returns:
        """
        download_res = self.download_distribution(
            dataset, dt_str, dataset_format, catalog, n_par, show_progress, force_download
        )

        if not all(res[0] for res in download_res):
            raise Exception(f'Not all downloads were successfully completed: {download_res}')

        files = [res[1] for res in download_res]

        pd_read_fn_map = {
            'csv': pd.read_csv,
            'parquet': pd.read_parquet,
            'parq': pd.read_parquet,
            'json': pd.read_json,
        }

        pd_read_default_kwargs: dict[str, dict[str, object]] = {
            'csv': {'sep': ',', 'header': 0, 'low_memory': True},
            'parquet': {'columns': columns},
        }

        pd_read_default_kwargs['parq'] = pd_read_default_kwargs['parquet']

        pd_reader = pd_read_fn_map.get(dataset_format)
        pd_read_kwargs = pd_read_default_kwargs.get(dataset_format, {})
        if not pd_reader:
            raise Exception(f'No pandas function to read file in format {dataset_format}')

        pd_read_kwargs.update(kwargs)
        dataframes = (pd_reader(f, **pd_read_kwargs) for f in files)
        df = pd.concat(dataframes, ignore_index=True)

        return df
