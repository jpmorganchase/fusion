"""Fusion authentication module."""

import datetime
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, Union
from urllib.parse import urlparse
import fsspec
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import CredentialError

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


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
        url (str): The content to evaluate.

    Returns:
        bool: True if the content of data is a URL, False otherwise.
    """
    try:
        urlparse(url)
        return True
    except ValueError:
        return False


def get_default_fs():
    """Retrieve default filesystem.

    Returns: filesystem

    """
    protocol = "file" if "FS_PROTOCOL" not in os.environ.keys() else os.environ["FS_PROTOCOL"]
    if (
            "S3_ENDPOINT" in os.environ.keys()
            and "AWS_ACCESS_KEY_ID" in os.environ.keys()
            and "AWS_SECRET_ACCESS_KEY" in os.environ.keys()
    ):
        endpoint = os.environ["S3_ENDPOINT"]
        fs = fsspec.filesystem(
            "s3",
            client_kwargs={"endpoint_url": f"https://{endpoint}"},
            key=os.environ["AWS_ACCESS_KEY_ID"],
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        )
    else:
        fs = fsspec.filesystem(protocol)
    return fs


class FusionCredentials:
    """Utility functions to manage credentials."""

    def __init__(
            self,
            client_id: str = None,
            client_secret: str = None,
            username: str = None,
            password: str = None,
            resource: str = None,
            auth_url: str = None,
            proxies={},
            grant_type: str = 'client_credentials',
    ) -> None:
        """Constructor for the FusionCredentials authentication management class.

        Args:
            client_id (str, optional): A valid OAuth client identifier. Defaults to None.
            client_secret (str, optional): A valid OAuth client secret. Defaults to None.
            username (str, optional): A valid username. Defaults to None.
            password (str, optional): A valid password for the username. Defaults to None.
            resource (str, optional): The OAuth audience. Defaults to None.
            auth_url (str, optional): URL for the OAuth authentication server. Defaults to None.
            proxies (dict, optional): Any proxy servers required to route HTTP and HTTPS requests to the internet.
            grant_type (str, optional): Allows the grant type to be changed to support different credential types.
                Defaults to client_credentials.
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.resource = resource
        self.auth_url = auth_url
        self.proxies = proxies
        self.grant_type = grant_type

    @staticmethod
    def add_proxies(
            http_proxy: str, https_proxy: str = None, credentials_file: str = 'config/client_credentials.json'
    ) -> None:
        """A function to add proxies to an existing credentials files.
        This function can be called to add proxy addresses to a credential file downloaded from the Fusion website.

        Args:
            http_proxy (str): The HTTP proxy address.
            https_proxy (str): The HTTPS proxy address. If not specified then this will be the
                copied form the HTTP proxy.
            credentials_file (str, optional): The path and filename to store the credentials under.
                Path may be absolute or relative to current working directory.
                Defaults to 'config/client_credentials.json'.

        Returns:
            None
        """

        credentials = FusionCredentials.from_file(credentials_file)
        credentials.proxies['http'] = http_proxy
        if https_proxy is None:
            https_proxy = http_proxy

        credentials.proxies['https'] = https_proxy

        data: Dict[str, Union[str, dict]] = dict(
            {
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'resource': credentials.resource,
                'auth_url': credentials.auth_url,
                'proxies': credentials.proxies,
            }
        )
        json_data = json.dumps(data, indent=4)
        with open(credentials_file, 'w') as credentialsfile:
            credentialsfile.write(json_data)

        return

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

        credentials = FusionCredentials.from_file(file_path=credentials_file)
        return credentials

    @staticmethod
    def from_dict(credentials: dict):
        """Create a credentials object from a dictionary.

            This is the only FusionCredentials creation method that supports the password grant type
            since the username and password should be provided by the user.

        Args:
            credentials (dict): A dictionary containing the requried keys: client_id, client_secret,
                resource, auth_url, and optionally proxies and an OAuth grant type.

        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        if 'grant_type' in credentials:
            grant_type = credentials['grant_type']
        else:
            grant_type = 'client_credentials'

        client_id = credentials['client_id']
        if grant_type == 'client_credentials':
            client_secret = credentials['client_secret']
            username = None
            password = None
        elif grant_type == 'password':
            client_secret = None
            username = credentials['username']
            password = credentials['password']
        else:
            raise CredentialError(f'Unrecognised grant type {grant_type}')

        resource = credentials['resource']
        auth_url = credentials['auth_url']
        proxies = credentials.get('proxies')
        creds = FusionCredentials(
            client_id, client_secret, username, password, resource, auth_url, proxies, grant_type=grant_type
        )
        return creds

    @staticmethod
    def from_file(file_path: str = 'config/client.credentials.json', fs=None, walk_up_dirs=True):
        """Create a credentials object from a file.

        Args:
            file_path (str, optional): Path (absolute or relative) and filename
                to load credentials from. Defaults to 'config/client.credentials.json'.
            fs (fsspec.filesystem): Filesystem to use.
            walk_up_dirs (bool): if true it walks up the directories in search of a config folder
        Returns:
            FusionCredentials: a credentials object that can be used for authentication.
        """
        fs = fs if fs else get_default_fs()
        to_use_file_path = None

        if fs.exists(file_path):  # absolute path case
            logger.log(VERBOSE_LVL, f"Found credentials file at {file_path}")
            to_use_file_path = file_path
        elif fs.exists(os.path.join(fs.info("")["name"], file_path)):  # relative path case
            to_use_file_path = os.path.join(fs.info("")["name"], file_path)
            logger.log(VERBOSE_LVL, f"Found credentials file at {to_use_file_path}")
        else:
            for p in [s.__str__() for s in Path(fs.info("")["name"]).parents]:
                if fs.exists(os.path.join(p, file_path)):
                    to_use_file_path = os.path.join(p, file_path)
                    logger.log(VERBOSE_LVL, f"Found credentials file at {to_use_file_path}")
                    break
        if fs.size(to_use_file_path) > 0:
            try:
                with fs.open(to_use_file_path, 'r') as credentials:
                    data = json.load(credentials)
                    credentials = FusionCredentials.from_dict(data)
                    return credentials
            except Exception as e:
                print(e)
                logger.error(e)
                raise Exception(e)
        else:
            msg = f"{to_use_file_path} is an empty file, make sure to add your credentials to it."
            print(msg)
            logger.error(msg)
            raise IOError(msg)

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
            self.auth_retries = Retry(total=20, backoff_factor=0.2)
        else:
            self.auth_retries = Retry.from_int(auth_retries)

    def send(self, request, **kwargs):
        """Function to send a request to the authentication server.

        Args:
            request (requests.Session): A HTTP Session.

        Returns:

        """

        def _refresh_token_data():
            payload = (
                {
                    "grant_type": "client_credentials",
                    "client_id": self.credentials.client_id,
                    "client_secret": self.credentials.client_secret,
                    "aud": self.credentials.resource,
                }
                if self.credentials.grant_type == 'client_credentials'
                else {
                    "grant_type": "password",
                    "client_id": self.credentials.client_id,
                    "username": self.credentials.username,
                    "password": self.credentials.password,
                    "resource": self.credentials.resource,
                }
            )

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

        request.headers.update({'Authorization': f'Bearer {self.token}'})
        response = super(FusionOAuthAdapter, self).send(request, **kwargs)
        return response