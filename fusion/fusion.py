"""Main Fusion module."""

import datetime
import json
import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse, urlunparse

import pandas as pd
import requests
from joblib import Parallel, delayed
from requests.adapters import HTTPAdapter
from tabulate import tabulate
from tqdm import tqdm
from urllib3.util.retry import Retry

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
    """APIRequestError exception wrapper to handle API request erorrs.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIResponseError(Exception):
    """APIResponseError exception wrapper to handle API response errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class APIConnectError(Exception):
    """APIConnectError exception wrapper to handle API connection errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class UnrecognizedFormatError(Exception):
    """UnrecognizedFormatError exception wrapper to handle format errors.

    Args:
        Exception : Exception to wrap.
    """

    pass


class CredentialError(Exception):
    """CredentialError exception wrapper to handle errors in credentials provided for authentication.

    Args:
        Exception : Exception to wrap.
    """

    pass


def _res_plural(ref_int: int, pluraliser: str = 's') -> str:
    """Private function to return the plural form when the number is more than one.

    Args:
        ref_int (int): The reference integer that determines whether to return a plural suffix.
        pluraliser (str, optional): The plural suffix. Defaults to "s".

    Returns:
        str: The plural suffix to append to a string.
    """
    return '' if abs(ref_int) == 1 else pluraliser


def _is_json(data: str) -> bool:
    """Test whether the content of a string is a JSON object.

    Args:
        data (str): The content to evaluate.

    Returns:
        bool: True if the content of data is JSON, False otherwise.
    """
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def _is_url(url: str) -> bool:
    """Test whether the content of a string is a valid URL.

    Args:
        data (str): The content to evaluate.

    Returns:
        bool: True if the content of data is a URL, False otherwise.
    """
    try:
        urlparse(url)
        return True
    except ValueError:
        return False


def _normalise_dt_param(dt: Union[str, int, datetime.datetime, datetime.date]) -> str:
    """Convert dates into a normalised string representation.

    Args:
        dt (Union[str, int, datetime.datetime, datetime.date]): A date represented in various types.

    Returns:
        str: A normalized date string.
    """
    if isinstance(dt, (datetime.date, datetime.datetime)):
        return dt.strftime("%Y-%m-%d")

    if isinstance(dt, int):
        dt = str(dt)

    if not isinstance(dt, str):
        raise ValueError(f"{dt} is not in a recognised data format")

    matches = DT_YYYY_MM_DD_RE.match(dt)
    if matches:
        yr = matches.group(1)
        mth = matches.group(2).zfill(2)
        day = matches.group(3).zfill(2)
        return f"{yr}-{mth}-{day}"

    matches = DT_YYYYMMDD_RE.match(dt)

    if matches:
        return "-".join(matches.groups())

    raise ValueError(f"{dt} is not in a recognised data format")


def _normalise_dt_param_str(dt: str) -> tuple:
    """Convert a date parameter which may be a single date or a date range into a tuple.

    Args:
        dt (str): Either a single date or a date range separated by a ":".

    Returns:
        tuple: A tuple of dates.
    """
    date_parts = dt.split(":")

    if not date_parts or len(date_parts) > 2:
        raise ValueError(f"Unable to parse {dt} as either a date or an interval")

    return tuple((_normalise_dt_param(dt_part) if dt_part else dt_part for dt_part in date_parts))


def _distribution_to_filename(
    root_folder: str, dataset: str, datasetseries: str, file_format: str, catalog: str = 'common'
) -> Path:
    """Returns a filename representing a dataset distribution.

    Args:
        root_folder (str): The root folder path.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".

    Returns:
        tuple: A tuple of dates.
    """
    if datasetseries[-1] == '/' or datasetseries[-1] == '\\':
        datasetseries = datasetseries[0:-1]
    file_name = f"{dataset}__{catalog}__{datasetseries}.{file_format}"
    return Path(root_folder, file_name)


def _filename_to_distribution(file_name: str) -> tuple:
    """Breaks a filename down into the components that represent a dataset distribution in the catalog.

    Args:
        file_name (str): The filename to decompose.

    Returns:
        tuple: A tuple consisting of catalog id, dataset id, datasetseries instane id, and file format (e.g. CSV).
    """
    dataset, catalog, series_format = Path(file_name).name.split('__')
    datasetseries, file_format = series_format.split('.')
    return (catalog, dataset, datasetseries, file_format)


def _distribution_to_url(
    root_url: str, dataset: str, datasetseries: str, file_format: str, catalog: str = 'common'
) -> str:
    """Returns the API URL to download a dataset distribution.

    Args:
        root_url (str): The base url for the API.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".

    Returns:
        str: A URL for the API distribution endpoint.
    """
    if datasetseries[-1] == '/' or datasetseries[-1] == '\\':
        datasetseries = datasetseries[0:-1]

    return f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{datasetseries}/distributions/{file_format}"


class FusionCredentials:
    """Utility functions to manage credentials."""

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        resource: str = None,
        auth_url: str = None,
        proxies={},
    ) -> None:
        """Constuctor for the FusionCredentials authentication management class.

        Args:
            client_id (str, optional): A valid OAuth client identifier. Defaults to None.
            client_secret (str, optional): A valid OAuth client secret. Defaults to None.
            resource (str, optional): The OAuth audience. Defaults to None.
            auth_url (str, optional): URL for the OAuth authentication server. Defaults to None.
            proxies (dict, optional): Any proxy servers required to route HTTP and HTTPS requests to the internet.
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
        resource: str = "JPMC:URI:RS-93742-Fusion-PROD",
        auth_url: str = "https://authe.jpmorgan.com/as/token.oauth2",
        proxies: Union[str, dict] = None,
    ):
        """Utility function to generate credentials file that can be used for authentication.

        Args:
            credentials_file (str, optional): The path and filename to store the credentials under.
                Path may be absolute or relative to current working directory.
                Defaults to 'config/client_credentials.json'.
            client_id (str, optional): A valid OAuth client identifier. Defaults to None.
            client_secret (str, optional): A valid OAuth client secret. Defaults to None.
            resource (str, optional): The OAuth audience. Defaults to None.
            auth_url (str, optional): URL for the OAuth authentication server. Defaults to None.
            proxies (Union[str, dict], optional): Any proxy servers required to route HTTP and HTTPS
                requests to the internet. Defaults to {}. Keys are http and https. Or specify a single
                URL to set both http and https

        Raises:
            CredentialError: Exception to handle missing values required for authentication.

        Returns:
           FusionCredentials: a credentials object that can be used for authentication.
        """
        if not client_id:
            raise CredentialError('A valid client_id is required')
        if not client_secret:
            raise CredentialError('A valid client secret is required')

        data: Dict[str, Union[str, dict]] = dict(
            {'client_id': client_id, 'client_secret': client_secret, 'resource': resource, 'auth_url': auth_url}
        )

        proxies_resolved = {}
        if proxies:
            if isinstance(proxies, dict):
                raw_proxies_dict = proxies
            elif isinstance(proxies, str):
                if _is_url(proxies):
                    raw_proxies_dict = {'http': proxies, 'https': proxies}
                elif _is_json(proxies):
                    raw_proxies_dict = json.loads(proxies)
            else:
                raise CredentialError(f'A valid proxies param is required, [{proxies}] is not supported.')

            # Now validate and conform proxies dict
            valid_pxy_keys = ['http', 'https', 'http_proxy', 'https_proxy']
            pxy_key_map = {
                'http': 'http',
                'https': 'https',
                'http_proxy': 'http',
                'https_proxy': 'https',
            }
            lcase_dict = {k.lower(): v for k, v in raw_proxies_dict.items()}

            if set(lcase_dict.keys()).intersection(set(valid_pxy_keys)) != set(lcase_dict.keys()):
                raise CredentialError(
                    f'Invalid proxies keys in dict {raw_proxies_dict.keys()}.'
                    f'Only {pxy_key_map.keys()} are accepted and will be mapped as necessary.'
                )
            proxies_resolved = {pxy_key_map[k]: v for k, v in lcase_dict.items()}

        data['proxies'] = proxies_resolved
        json_data = json.dumps(data, indent=4)
        Path(credentials_file).parent.mkdir(parents=True, exist_ok=True)
        with open(credentials_file, 'w') as credentialsfile:
            credentialsfile.write(json_data)

        credentials = FusionCredentials.from_file(credentials_file=credentials_file)
        return credentials

    @staticmethod
    def from_dict(credentials: dict):
        """Create a credentials object from a dictionary.

        Args:
            credentials (dict): A dictionary containing the requried keys: client_id, client_secret,
                resource, auth_url, and optionally proxies

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
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
        """Create a credentils object from a file.

        Args:
            credentials_file (str, optional): Path (absolute or relative) and filename
                to load credentials from. Defaults to 'config/client.credentials.json'.

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        with open(credentials_file, 'r') as credentials:
            data = json.load(credentials)
            credentials = FusionCredentials.from_dict(data)
            return credentials

    @staticmethod
    def from_object(credentials_source: Union[str, dict]):
        """Utility function that will determine how to create a credentials object based on data passed.

        Args:
            credentials_source (Union[str, dict]): A string which could be a filename or a JSON object, or a dictionary.

        Raises:
            CredentialError: Exception raised when the provided credentials is not one of the supported types

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        if isinstance(credentials_source, dict):
            return FusionCredentials.from_dict(credentials_source)
        elif isinstance(credentials_source, str):
            if _is_json(credentials_source):
                return FusionCredentials.from_dict(json.loads(credentials_source))
            else:
                return FusionCredentials.from_file(credentials_source)

        raise CredentialError(f'Could not resolve the credentials provided: {credentials_source}')


class FusionOAuthAdapter(HTTPAdapter):
    """An OAuth adapter to manage authentication and session tokens."""

    def __init__(
        self,
        credentials: Union[FusionCredentials, Union[str, dict]],
        proxies: dict = {},
        refresh_within_seconds: int = 5,
        auth_retries: Union[int, Retry] = None,
        *args,
        **kwargs,
    ) -> None:
        """Class constructor to create a FusionOAuthAdapter object.

        Args:
            credentials (Union[FusionCredentials, Union[str, dict]): Valid user credentials to authenticate.
            proxies (dict, optional): Specify a proxy if required to access the authentication server.
                Defaults to {}.
            refresh_within_seconds (int, optional): When an API call is made with less than the specified
                number of seconds until the access token expires, or after expiry, it will refresh the token.
                Defaults to 5.
            auth_retries (Union[int, Retry]): Number of times to attempt to authenticate to handle connection problems.
                Defaults to None.

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

        if not auth_retries:
            self.auth_retries = Retry(total=5, backoff_factor=0.2)
        else:
            self.auth_retries = Retry.from_int(auth_retries)

    def send(self, request, **kwargs):
        """Function to send a request to the authentication server.

        Args:
            request (requests.Session): A HTTP Session.

        Returns:

        """

        def _refresh_token_data():
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret,
                "aud": self.credentials.resource,
            }

            try:
                s = requests.Session()
                if self.proxies:
                    # mypy does note recognise session.proxies as a dict so fails this line, we'll ignore this chk
                    s.proxies.update(self.proxies)  # type:ignore
                s.mount('http://', HTTPAdapter(max_retries=self.auth_retries))
                response = s.post(self.credentials.auth_url, data=payload)
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
    """Get the full URL for the API endpoint.

    Args:
        any_url (str): A valid URL or URL part

    Returns:
        str: A complete root URL

    """
    url_parts = urlparse(any_url)
    root_url = urlunparse((url_parts[0], url_parts[1], '', '', '', ''))
    return root_url


def _get_session(
    credentials: FusionCredentials, root_url: str, get_retries: Union[int, Retry] = None
) -> requests.Session:
    """Create a new http session and set paramaters.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an acces token
        root_url (str): The URL to call.

    Returns:
        requests.Session(): A HTTP Session object

    """
    if not get_retries:
        get_retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
    else:
        get_retries = Retry.from_int(get_retries)
    session = requests.Session()
    auth_handler = FusionOAuthAdapter(credentials, max_retries=get_retries)
    if credentials.proxies:
        # mypy does note recognise session.proxies as a dict so fails this line, we'll ignore this chk
        session.proxies.update(credentials.proxies)  # type:ignore
    try:
        mount_url = _get_canonical_root_url(root_url)
    except Exception:
        mount_url = "https://"
    session.mount(mount_url, auth_handler)
    return session


def _stream_single_file_new_session_dry_run(credentials, url: str, output_file: str):
    """Function to test that a distribution is available without downloading.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an acces token
        root_url (str): The URL to call.
        output_file: The filename that the data will be saved into.

    Returns:
        requests.Session(): A HTTP Session object

    """
    try:
        resp = _get_session(credentials, url).head(url)
        resp.raise_for_status()
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
) -> tuple:
    """Function to stream a single file from the API to a file on disk.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an acces token
        root_url (str): The URL to call.
        output_file (str): The filename that the data will be saved into.
        overwrite (bool, optional): True if previously downloaded files should be overwritten. Defaults to True.
        block_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE
        dry_run (bool, optional): Test that a file can be downloaded and return the filename without
            downloading the data. Defaults to False.

    Returns:
        tuple: A tuple

    """
    if dry_run:
        return _stream_single_file_new_session_dry_run(credentials, url, output_file)

    output_file_path = Path(output_file)
    if not overwrite and output_file_path.exists():
        return (True, output_file, None)

    tmp_name = output_file_path.with_name(output_file_path.name + ".tmp")
    try:
        with _get_session(credentials, url).get(url, stream=True) as r:
            r.raise_for_status()
            byte_cnt = 0
            with open(tmp_name, "wb") as outfile:
                for chunk in r.iter_content(block_size):
                    byte_cnt += len(chunk)
                    outfile.write(chunk)
        tmp_name.rename(output_file_path)
        tmp_name.unlink(missing_ok=True)
        logger.log(
            VERBOSE_LVL,
            f'Wrote {byte_cnt:,} bytes to {output_file_path}, via {tmp_name}',
        )
        return (True, output_file, None)
    except Exception as ex:
        logger.log(
            VERBOSE_LVL,
            f'Failed to write to {output_file_path}, via {tmp_name}. ex - {ex}',
        )
        return (False, output_file, ex)


def _stream_single_file(session: requests.Session, url: str, output_file: str, block_size=DEFAULT_CHUNK_SIZE):
    """Streams a file downloaded from a URL to a file.

    Args:
        session (FusionCredentials): A HTTP Session
        url (str): The URL to call for the file download.
        output_file (str): The filename that the data will be saved into.
        block_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE

    """
    output_file_path = Path(output_file)
    tmp_name = output_file_path.with_name(output_file_path.name + ".tmp")
    with session.get(url, stream=True) as r:
        r.raise_for_status()
        with open(tmp_name, "wb") as outfile:
            for chunk in r.iter_content(block_size):
                outfile.write(chunk)
    tmp_name.rename(output_file_path)
    tmp_name.unlink(missing_ok=True)


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

    def catalog_resources(self, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """List the resources contained within the catalog, for example products and datasets.

        Args:
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
           pandas.DataFrame: A dataframe with a row for each resource within the catalog
        """
        url = f'{self.root_url}catalogs/{catalog}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_products(
        self,
        contains: Union[str, list] = None,
        id_contains: bool = False,
        catalog: str = 'common',
        output: bool = False,
        max_results: int = -1,
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

        Returns:
            pandas.DataFrame: a dataframe with a row for each product
        """
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

        if max_results > -1:
            df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_datasets(
        self,
        contains: Union[str, list] = None,
        id_contains: bool = False,
        catalog: str = 'common',
        output: bool = False,
        max_results: int = -1,
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

        Returns:
            pandas.DataFrame: a dataframe with a row for each dataset.
        """
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

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def dataset_resources(self, dataset: str, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """List the resources available for a dataset, currently this will always be a datasetseries.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each resource
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_dataset_attributes(self, dataset: str, catalog: str = 'common', output: bool = False) -> pd.DataFrame:
        """Returns the list of attributes that are in the dataset.

        Args:
            dataset (str): A dataset identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each attribute
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/attributes'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_datasetmembers(
        self, dataset: str, catalog: str = 'common', output: bool = False, max_results: int = -1
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
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries'
        df = Fusion._call_for_dataframe(url, self.session)

        if max_results > -1:
            df = df[0:max_results]

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def datasetmember_resources(
        self, dataset: str, series: str, catalog: str = 'common', output: bool = False
    ) -> pd.DataFrame:
        """List the available resources for a datasetseries member.

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each datasetseries member resource.
                Currently this will always be distributions.
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def list_distributions(
        self, dataset: str, series: str, catalog: str = 'common', output: bool = False
    ) -> pd.DataFrame:
        """List the available distributions (downloadable instances of the dataset with a format type).

        Args:
            dataset (str): A dataset identifier
            series (str): The datasetseries identifier
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            output (bool, optional): If True then print the dataframe. Defaults to False.

        Returns:
            pandas.DataFrame: A dataframe with a row for each distribution.
        """
        url = f'{self.root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/{series}/distributions'
        df = Fusion._call_for_dataframe(url, self.session)

        if output:
            print(tabulate(df, headers="keys", tablefmt="psql"))

        return df

    def _resolve_distro_tuples(
        self, dataset: str, dt_str: str = 'latest', dataset_format: str = 'parquet', catalog: str = 'common'
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
        datasetseries_list = self.list_datasetmembers(dataset, catalog)

        if dt_str == 'latest':
            dt_str = datasetseries_list.iloc[datasetseries_list['createdDate'].values.argmax()]['identifier']

        parsed_dates = _normalise_dt_param_str(dt_str)
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
        catalog: str = 'common',
        n_par: int = DEFAULT_PARALLELISM,
        show_progress: bool = True,
        force_download: bool = False,
        download_folder: str = None,
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
                Defaults to DEFAULT_PARALLELISM.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to True.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__

        Returns:

        """
        required_series = self._resolve_distro_tuples(dataset, dt_str, dataset_format, catalog)

        if not download_folder:
            download_folder = self.download_folder
        Path(download_folder).mkdir(parents=True, exist_ok=True)

        download_spec = [
            (
                self.credentials,
                _distribution_to_url(self.root_url, series[1], series[2], series[3], series[0]),
                _distribution_to_filename(download_folder, series[1], series[2], series[3], series[0]),
                force_download,
            )
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
        res = Parallel(n_jobs=n_par)(delayed(_stream_single_file_new_session)(*spec) for spec in loop)

        return res

    def to_df(
        self,
        dataset: str,
        dt_str: str = 'latest',
        dataset_format: str = 'parquet',
        catalog: str = 'common',
        n_par: int = DEFAULT_PARALLELISM,
        show_progress: bool = True,
        columns: List = None,
        force_download: bool = False,
        download_folder: str = None,
        **kwargs,
    ):
        """Gets distributions for a specified date or date range and returns the data as a dataframe.

        Args:
            dataset (str): A dataset identifier
            dt_str (str, optional): Either a single date or a range identified by a start or end date,
                or both separated with a ":". Defaults to 'latest' which will return the most recent
                instance of the dataset.
            dataset_format (str, optional): The file format, e.g. CSV or Parquet. Defaults to 'parquet'.
            catalog (str, optional): A catalog identifier. Defaults to 'common'.
            n_par (int, optional): Specify how many distributions to download in parallel.
                Defaults to DEFAULT_PARALLELISM.
            show_progress (bool, optional): Display a progress bar during data download Defaults to True.
            dry_run (bool, optional): _description_. Defaults to True
            columns (List, optional): _description_. Defaults to None
            force_download (bool, optional): If True then will always download a file even
                if it is already on disk. Defaults to True.
            download_folder (str, optional): The path, absolute or relative, where downloaded files are saved.
                Defaults to download_folder as set in __init__
        Returns:
            pandas.DataFrame: a dataframe containing the requested data.
                If multiple dataset instances are retrieved then these are concatenated first.
        """
        if not download_folder:
            download_folder = self.download_folder
        download_res = self.download(
            dataset, dt_str, dataset_format, catalog, n_par, show_progress, force_download, download_folder
        )

        if not all(res[0] for res in download_res):
            failed_res = [res for res in download_res if not res[0]]
            raise Exception(
                f"Not all downloads were successfully completed. "
                f"Re-run to collect missing files. The following failed:\n{failed_res}"
            )

        files = [res[1] for res in download_res]

        pd_read_fn_map = {
            'csv': pd.read_csv,
            'parquet': pd.read_parquet,
            'parq': pd.read_parquet,
            'json': pd.read_json,
        }

        pd_read_default_kwargs: Dict[str, Dict[str, object]] = {
            'csv': {'sep': ',', 'header': 0, 'low_memory': True},
            'parquet': {'columns': columns},
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
        dataframes = (pd_reader(f, **pd_read_kwargs) for f in files)
        df = pd.concat(dataframes, ignore_index=True)

        return df
