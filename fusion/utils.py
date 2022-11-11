"""Fusion utilities."""

import datetime
import logging
import pyarrow.parquet as pq
import pandas as pd
import re
import requests
import os
import aiohttp
from typing import Union
from pathlib import Path
from pyarrow import csv, json
from io import BytesIO
from urllib.parse import urlparse, urlunparse
from joblib import Parallel, delayed
import hashlib
import base64
import sys
if sys.version_info >= (3, 7):
    from contextlib import nullcontext
else:
    class nullcontext(object):
        """Class for Python 3.6 compatibility."""
        def __init__(self, dummy_resource=None):
            """Constructor.
            """
            self.dummy_resource = dummy_resource

        def __enter__(self):
            """Enter.
            """
            return self.dummy_resource

        def __exit__(self, *args):
            """Exit.
            """
            pass

from urllib3.util.retry import Retry
import multiprocessing as mp
from .authentication import FusionCredentials, FusionOAuthAdapter

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
DT_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
DT_YYYY_MM_DD_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
DEFAULT_CHUNK_SIZE = 2**16
DEFAULT_THREAD_POOL_SIZE = 5


def cpu_count(thread_pool_size: int = None):
    """Determine the number of cpus/threads for parallelization.

    Args:
        thread_pool_size: override argument for number of cpus/threads.

    Returns: number of cpus/threads to use.

    """
    if os.environ.get("NUM_THREADS") is not None:
        return int(os.environ["NUM_THREADS"])
    if thread_pool_size:
        return thread_pool_size
    else:
        if mp.cpu_count():
            thread_pool_size = mp.cpu_count()
        else:
            thread_pool_size = DEFAULT_THREAD_POOL_SIZE
    return thread_pool_size


def _csv_to_table(path: str, fs=None, delimiter: str = None):
    """Reads csv data to pyarrow table.

    Args:
        path (str): path to the file.
        fs: filesystem object.
        delimiter (str): delimiter.

    Returns:
        pyarrow.table pyarrow table with the data.
    """
    parse_options = csv.ParseOptions(delimiter=delimiter)
    with (fs.open(path) if fs else nullcontext(path)) as f:
        return csv.read_csv(f, parse_options=parse_options)


def _json_to_table(path: str, fs=None):
    """Reads json data to pyarrow table.

    Args:
        path: path to json file.
        fs: filesystem.

    Returns:

    """
    with (fs.open(path) if fs else nullcontext(path)) as f:
        return json.read_json(f)


def read_csv(path: str, columns: list = None, filters: list = None, fs=None):
    """Reads csv with possibility of selecting columns and filtering the data.

    Args:
        path (str): path to the csv file.
        columns: list of selected fields.
        filters: filters.
        fs: filesystem object.

    Returns:
        pandas.DataFrame: a dataframe containing the data.

    """
    try:
        try:
            tbl = _csv_to_table(path, fs)
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f'Failed to read {path}, with comma delimiter. {err}',
            )
            raise Exception

        out = BytesIO()
        pq.write_table(tbl, out)
        del tbl
        res = pq.read_table(out, filters=filters, columns=columns).to_pandas()
    except Exception as err:
        logger.log(
            VERBOSE_LVL,
            f"Could not parse {path} properly. " f"Trying with pandas csv reader. {err}",
        )
        try:
            with (fs.open(path) if fs else nullcontext(path)) as f:
                res = pd.read_csv(f, usecols=columns, index_col=False)
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Could not parse {path} properly. " f"Trying with pandas csv reader pandas engine. {err}",
            )
            with (fs.open(path) if fs else nullcontext(path)) as f:
                res = pd.read_table(f, usecols=columns, index_col=False, engine="python", delimiter=None)
    return res


def read_parquet(path: Union[list, str], columns: list = None, filters: list = None, fs=None):
    """Read parquet files(s) to pandas.

    Args:
        path (Union[list, str]): path or a list of paths to parquet files.
        columns (list): list of selected fields.
        filters (list): filters.
        fs: filesystem object.

    Returns:
        pandas.DataFrame: a dataframe containing the data.

    """
    return pq.ParquetDataset(path, use_legacy_dataset=False, filters=filters, filesystem=fs,
                             memory_map=True).read_pandas(columns=columns).to_pandas()


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


def normalise_dt_param_str(dt: str) -> tuple:
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


def distribution_to_filename(
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


def distribution_to_url(
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


async def get_client(credentials, **kwargs):
    """Gets session for async.

    Args:
        credentials: Credentials.
        **kwargs: Kwargs.

    Returns:

    """
    async def on_request_start(session, trace_config_ctx, params):
        payload = (
            {
                "grant_type": "client_credentials",
                "client_id": credentials.client_id,
                "client_secret": credentials.client_secret,
                "aud": credentials.resource,
            }
            if credentials.grant_type == 'client_credentials'
            else {
                "grant_type": "password",
                "client_id": credentials.client_id,
                "username": credentials.username,
                "password": credentials.password,
                "resource": credentials.resource,
            }
        )
        async with aiohttp.ClientSession() as session:
            if credentials.proxies:
                response = await session.post(credentials.auth_url, data=payload, proxy=http_proxy)
            else:
                response = await session.post(credentials.auth_url, data=payload)
            response_data = await response.json()

        access_token = response_data["access_token"]
        params.headers.update({'Authorization': f'Bearer {access_token}'})

    if credentials.proxies:
        if "http" in credentials.proxies.keys():
            http_proxy = credentials.proxies["http"]
        elif "https" in credentials.proxies.keys():
            http_proxy = credentials.proxies["https"]

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start)
    return aiohttp.ClientSession(trace_configs=[trace_config], trust_env=True)


def get_session(
    credentials: FusionCredentials, root_url: str, get_retries: Union[int, Retry] = None
) -> requests.Session:
    """Create a new http session and set parameters.

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
        resp = get_session(credentials, url).head(url)
        resp.raise_for_status()
        return True, output_file, None
    except Exception as ex:
        return False, output_file, ex


def stream_single_file_new_session(
    credentials,
    url: str,
    output_file: str,
    overwrite: bool = True,
    block_size=DEFAULT_CHUNK_SIZE,
    dry_run: bool = False,
    fs=None
) -> tuple:
    """Function to stream a single file from the API to a file on disk.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an acces token
        url (str): The URL to call.
        output_file (str): The filename that the data will be saved into.
        overwrite (bool, optional): True if previously downloaded files should be overwritten. Defaults to True.
        block_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE
        dry_run (bool, optional): Test that a file can be downloaded and return the filename without
            downloading the data. Defaults to False.
        fs (fsspec.filesystem): Filesystem.

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
        with get_session(credentials, url).get(url, stream=True) as r:
            r.raise_for_status()
            byte_cnt = 0
            with fs.open(tmp_name, "wb") as outfile:
                for chunk in r.iter_content(block_size):
                    byte_cnt += len(chunk)
                    outfile.write(chunk)
        tmp_name.rename(output_file_path)
        try:
            tmp_name.unlink()
        except FileNotFoundError:
            pass
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
        return False, output_file, ex


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
    try:
        tmp_name.unlink()
    except FileNotFoundError:
        pass


def validate_file_names(paths, fs_fusion):
    """Validate if the file name format adheres to the standard.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    validation = []
    all_catalogs = fs_fusion.ls('')
    all_datasets = {}
    for f_n in file_names:
        tmp = f_n.split('__')
        if len(tmp) == 3:
            val = tmp[1] in all_catalogs
            if not val:
                validation.append(False)
            else:
                if tmp[1] not in all_datasets.keys():
                    all_datasets[tmp[1]] = [ i.split('/')[-1] for i in fs_fusion.ls(f"{tmp[1]}/datasets")]

                val = tmp[0] in all_datasets[tmp[1]]
                validation.append(val)
        else:
            validation.append(False)

    if not all(validation):
        for i, p in enumerate(paths):
            if not validation[i]:
                logger.warning(f"The file in {p} has a non-compliant name and will not be processed. "
                               f"Please rename the file to dataset__catalog__yyyymmdd.format")
    return validation


def path_to_url(x):
    """Convert file name to fusion url.

    Args:
        x (str): File path.

    Returns (str): Fusion url string.

    """
    catalog, dataset, date, ext = _filename_to_distribution(x.split("/")[-1])
    return "/".join(distribution_to_url("", dataset, date, ext, catalog).split("/")[1:])


def _construct_headers(fs_local, row):
    dt = row["url"].split("/")[-3]
    dt_iso = pd.Timestamp(dt).strftime("%Y-%m-%d")
    headers = {
        "Content-Type": "application/octet-stream",
        "x-jpmc-distribution-created-date": dt_iso,
        "x-jpmc-distribution-from-date": dt_iso,
        "x-jpmc-distribution-to-date": dt_iso,
        "Digest": ""
    }
    with fs_local.open(row["path"], "rb") as file_local:
        hash_md5 = hashlib.md5()
        for chunk in iter(lambda: file_local.read(4096), b""):
            hash_md5.update(chunk)
        headers["Digest"] = "md5=" + base64.b64encode(hash_md5.digest()).decode()
    return headers


def upload_files(fs_fusion, fs_local, loop, parallel=True, n_par=-1):
    """Upload file into Fusion.

    Args:
        fs_fusion: Fusion filesystem.
        fs_local: Local filesystem.
        loop (iterable): Loop of files to iterate through.
        parallel (bool): Is parallel mode enabled.
        n_par (int): Number of subprocesses.

    Returns: List of update statuses.

    """
    def _upload(row):
        kw = {"headers": _construct_headers(fs_local, row)}
        p_url = row["url"]
        try:
            with fs_local.open(row["path"], "rb") as file_local:
                fs_fusion.put(file_local, p_url, chunk_size=100 * 2 ** 20, method="put", **kw)
            return True, row["path"], None
        except Exception as ex:
            logger.log(
                VERBOSE_LVL,
                f'Failed to upload {row["path"]}. ex - {ex}',
            )
            return False, row["path"], ex

    if parallel:
        res = Parallel(n_jobs=n_par)(delayed(_upload)(row) for index, row in loop)
    else:
        res = [_upload(row) for index, row in loop]
    return res
