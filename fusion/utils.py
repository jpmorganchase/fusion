"""Fusion utilities."""

import datetime
from datetime import timedelta
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union
from urllib.parse import urlparse, urlunparse

import aiohttp
import json as js
import pandas as pd
import pyarrow.parquet as pq
import requests
from joblib import Parallel, delayed
from pyarrow import csv, json, unify_schemas
from pyarrow.parquet import filters_to_expression

if sys.version_info >= (3, 7):
    from contextlib import nullcontext
else:

    class nullcontext(object):
        """Class for Python 3.6 compatibility."""

        def __init__(self, dummy_resource=None):
            """Constructor."""
            self.dummy_resource = dummy_resource

        def __enter__(self):
            """Enter."""
            return self.dummy_resource

        def __exit__(self, *args):
            """Exit."""
            pass


import multiprocessing as mp

import fsspec
from urllib3.util.retry import Retry

from .authentication import FusionCredentials, FusionOAuthAdapter, FusionAiohttpSession

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


def csv_to_table(path: str, fs=None, columns: list = None, filters: list = None):
    """Reads csv data to pyarrow table.

    Args:
        path (str): path to the file.
        fs: filesystem object.
        columns: columns to read.
        filters: arrow filters.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    # parse_options = csv.ParseOptions(delimiter=delimiter)
    filters = filters_to_expression(filters) if filters else filters
    with (fs.open(path) if fs else nullcontext(path)) as f:
        tbl = csv.read_csv(f)
        if filters is not None:
            tbl = tbl.filter(filters)
        if columns is not None:
            tbl = tbl.select(columns)
        return tbl


def json_to_table(path: str, fs=None, columns: list = None, filters: list = None):
    """Reads json data to pyarrow table.

    Args:
        path: path to json file.
        fs: filesystem.
        columns: columns to read.
        filters: arrow filters.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    filters = filters_to_expression(filters) if filters else filters
    with (fs.open(path) if fs else nullcontext(path)) as f:
        tbl = json.read_json(f)
        if filters is not None:
            tbl = tbl.filter(filters)
        if columns is not None:
            tbl = tbl.select(columns)
        return tbl


def parquet_to_table(path: str, fs=None, columns: list = None, filters: list = None):
    """Reads parquet data to pyarrow table.

    Args:
        path: path to parquet file.
        fs: filesystem.
        columns: columns to read.
        filters: arrow filters.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """

    if isinstance(path, list):
        schemas = [pq.ParquetDataset(
            p,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
        ).schema for p in path]
    else:
        schemas = [pq.ParquetDataset(
            path,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
        ).schema]

    schema = unify_schemas(schemas)
    return (
        pq.ParquetDataset(
            path,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
            schema=schema
        ).read(columns=columns)
    )


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
            res = csv_to_table(path, fs).to_pandas()
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Failed to read {path}, with comma delimiter. {err}",
            )
            raise Exception

    except Exception as err:
        logger.log(
            VERBOSE_LVL,
            f"Could not parse {path} properly. "
            f"Trying with pandas csv reader. {err}",
        )
        try:
            with (fs.open(path) if fs else nullcontext(path)) as f:
                res = pd.read_csv(f, usecols=columns, index_col=False)
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Could not parse {path} properly. "
                f"Trying with pandas csv reader pandas engine. {err}",
            )
            with (fs.open(path) if fs else nullcontext(path)) as f:
                res = pd.read_table(
                    f, usecols=columns, index_col=False, engine="python", delimiter=None
                )
    return res


def read_json(path: str, columns: list = None, filters: list = None, fs=None):
    """Read json files(s) to pandas.

    Args:
        path (str): path or a list of paths to parquet files.
        columns (list): list of selected fields.
        filters (list): filters.
        fs: filesystem object.

    Returns:
        pandas.DataFrame: a dataframe containing the data.
    """

    try:
        try:
            res = json_to_table(path, fs).to_pandas()
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Failed to read {path}, with arrow reader. {err}",
            )
            raise Exception

    except Exception as err:
        logger.log(
            VERBOSE_LVL,
            f"Could not parse {path} properly. "
            f"Trying with pandas json reader. {err}",
        )
        try:
            with (fs.open(path) if fs else nullcontext(path)) as f:
                res = pd.read_json(f)
        except Exception as err:
            logger.error(
                VERBOSE_LVL,
                f"Could not parse {path} properly. " f"{err}",
            )
            raise Exception(err)
    return res


def read_parquet(
    path: Union[list, str], columns: list = None, filters: list = None, fs=None
):
    """Read parquet files(s) to pandas.

    Args:
        path (Union[list, str]): path or a list of paths to parquet files.
        columns (list): list of selected fields.
        filters (list): filters.
        fs: filesystem object.

    Returns:
        pandas.DataFrame: a dataframe containing the data.

    """

    if isinstance(path, list):
        schemas = [pq.ParquetDataset(
            p,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
        ).schema for p in path]
    else:
        schemas = [pq.ParquetDataset(
            path,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
        ).schema]

    schema = unify_schemas(schemas)
    return (
        pq.ParquetDataset(
            path,
            use_legacy_dataset=False,
            filters=filters,
            filesystem=fs,
            memory_map=True,
            schema=schema
        )
        .read_pandas(columns=columns)
        .to_pandas()
    )


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

    return tuple(
        (_normalise_dt_param(dt_part) if dt_part else dt_part for dt_part in date_parts)
    )


def distribution_to_filename(
    root_folder: str,
    dataset: str,
    datasetseries: str,
    file_format: str,
    catalog: str = "common",
    partitioning: str = None,
) -> str:
    """Returns a filename representing a dataset distribution.

    Args:
        root_folder (str): The root folder path.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        partitioning (str, optional): Partitioning specification.

    Returns:
        str: FQ distro file name
    """
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
        datasetseries = datasetseries[0:-1]

    if partitioning == "hive":
        file_name = f"{dataset}.{file_format}"
    else:
        file_name = f"{dataset}__{catalog}__{datasetseries}.{file_format}"

    sep = "/"
    if "\\" in root_folder:
        sep = "\\"
    return f"{root_folder}{sep}{file_name}"


def _filename_to_distribution(file_name: str) -> tuple:
    """Breaks a filename down into the components that represent a dataset distribution in the catalog.

    Args:
        file_name (str): The filename to decompose.

    Returns:
        tuple: A tuple consisting of catalog id, dataset id, datasetseries instane id, and file format (e.g. CSV).
    """
    dataset, catalog, series_format = Path(file_name).name.split("__")
    datasetseries, file_format = series_format.split(".")
    return catalog, dataset, datasetseries, file_format


def distribution_to_url(
    root_url: str,
    dataset: str,
    datasetseries: str,
    file_format: str,
    catalog: str = "common",
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
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
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
    root_url = urlunparse((url_parts[0], url_parts[1], "", "", "", ""))
    return root_url


async def get_client(credentials, **kwargs):
    """Gets session for async.

    Args:
        credentials: Credentials.
        **kwargs: Kwargs.

    Returns:

    """

    async def on_request_start_token(session, trace_config_ctx, params):
        async def _refresh_token_data():
            payload = (
                {
                    "grant_type": "client_credentials",
                    "client_id": credentials.client_id,
                    "client_secret": credentials.client_secret,
                    "aud": credentials.resource,
                }
                if credentials.grant_type == "client_credentials"
                else {
                    "grant_type": "password",
                    "client_id": credentials.client_id,
                    "username": credentials.username,
                    "password": credentials.password,
                    "resource": credentials.resource,
                }
            )
            async with aiohttp.ClientSession(trust_env=True) as session:
                if credentials.proxies:
                    response = await session.post(
                        credentials.auth_url, data=payload, proxy=http_proxy
                    )
                else:
                    response = await session.post(credentials.auth_url, data=payload)
                response_data = await response.json()

            access_token = response_data["access_token"]
            expiry = response_data["expires_in"]
            return access_token, expiry

        token_expires_in = (
                session.bearer_token_expiry - datetime.datetime.now()
        ).total_seconds()
        if token_expires_in < session.refresh_within_seconds:
            token, expiry = await _refresh_token_data()
            session.token = token
            session.bearer_token_expiry = datetime.datetime.now() + datetime.timedelta(
                seconds=int(expiry)
            )
            session.number_token_refreshes += 1

        params.headers.update({"Authorization": f"Bearer {session.token}"})

    async def on_request_start_fusion_token(session, trace_config_ctx, params):
        async def _refresh_fusion_token_data():
            full_url_lst = str(params.url).split("/")
            url = '/'.join(full_url_lst[:full_url_lst.index("datasets")+2]) + "/authorize/token"
            async with session.get(url) as response:
                response_data = await response.json()
            access_token = response_data["access_token"]
            expiry = response_data["expires_in"]
            return access_token, expiry

        url_lst = params.url.path.split('/')
        fusion_auth_req = "distributions" in url_lst
        if fusion_auth_req:
            catalog = url_lst[url_lst.index("catalogs") + 1]
            dataset = url_lst[url_lst.index("datasets") + 1]
            fusion_token_key = catalog + "_" + dataset
            if fusion_token_key not in session.fusion_token_dict.keys():
                fusion_token, fusion_token_expiry = await _refresh_fusion_token_data()
                session.fusion_token_dict[fusion_token_key] = fusion_token
                session.fusion_token_expiry_dict[fusion_token_key] = datetime.datetime.now() + timedelta(
                    seconds=int(fusion_token_expiry))
                logger.log(VERBOSE_LVL, "Refreshed fusion token")
            else:
                fusion_token_expires_in = (
                        session.fusion_token_expiry_dict[fusion_token_key] - datetime.datetime.now()
                ).total_seconds()
                if fusion_token_expires_in < session.refresh_within_seconds:
                    fusion_token, fusion_token_expiry = await _refresh_fusion_token_data()
                    session.fusion_token_dict[fusion_token_key] = fusion_token
                    session.fusion_token_expiry_dict[fusion_token_key] = datetime.datetime.now() + timedelta(
                        seconds=int(fusion_token_expiry))
                    logger.log(VERBOSE_LVL, "Refreshed fusion token")

            params.headers.update({"Fusion-Authorization": f"Bearer {session.fusion_token_dict[fusion_token_key]}"})

    if credentials.proxies:
        if "http" in credentials.proxies.keys():
            http_proxy = credentials.proxies["http"]
        elif "https" in credentials.proxies.keys():
            http_proxy = credentials.proxies["https"]

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start_token)
    trace_config.on_request_start.append(on_request_start_fusion_token)
    session = FusionAiohttpSession(trace_configs=[trace_config], trust_env=True)
    session.post_init()
    return session


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
        get_retries = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504]
        )
    else:
        get_retries = Retry.from_int(get_retries)
    session = requests.Session()
    if credentials.proxies:
        session.proxies.update(credentials.proxies)  # type:ignore
    try:
        mount_url = _get_canonical_root_url(root_url)
    except Exception:
        mount_url = "https://"
    auth_handler = FusionOAuthAdapter(credentials, max_retries=get_retries, mount_url=mount_url)
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
    fs: fsspec.AbstractFileSystem = fsspec.filesystem("file"),
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

    if not overwrite and fs.exists(output_file):
        return True, output_file, None

    try:
        with get_session(credentials, url).get(url, stream=True) as r:
            r.raise_for_status()
            byte_cnt = 0
            with fs.open(output_file, "wb") as outfile:
                for chunk in r.iter_content(block_size):
                    byte_cnt += len(chunk)
                    outfile.write(chunk)
        logger.log(
            VERBOSE_LVL,
            f"Wrote {byte_cnt:,} bytes to {output_file}",
        )
        return (True, output_file, None)
    except Exception as ex:
        logger.log(
            VERBOSE_LVL,
            f"Failed to write to {output_file}. ex - {ex}",
        )
        return False, output_file, ex


def validate_file_names(paths, fs_fusion):
    """Validate if the file name format adheres to the standard.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    validation = []
    all_catalogs = fs_fusion.ls("")
    all_datasets = {}
    for i, f_n in enumerate(file_names):
        tmp = f_n.split("__")
        if len(tmp) == 3:
            val = tmp[1] in all_catalogs
            if not val:
                validation.append(False)
            else:
                if tmp[1] not in all_datasets.keys():
                    all_datasets[tmp[1]] = [
                        i.split("/")[-1] for i in fs_fusion.ls(f"{tmp[1]}/datasets")
                    ]

                val = tmp[0] in all_datasets[tmp[1]]
                validation.append(val)
        else:
            validation.append(False)
        if not validation and len(tmp) == 3:
            logger.warning(
                f"You might not have access to the catalog {tmp[1]} or dataset {tmp[0]}."
                "Please check you permission on the platform."
            )
        if not validation and len(tmp) != 3:
            logger.warning(
                f"The file in {paths[i]} has a non-compliant name and will not be processed. "
                f"Please rename the file to dataset__catalog__yyyymmdd.format"
            )

    return validation


def is_dataset_raw(paths, fs_fusion):
    """Check if the files correspond to a raw dataset.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    ret = []
    is_raw = {}
    for i, f_n in enumerate(file_names):
        tmp = f_n.split("__")
        if tmp[0] not in is_raw.keys():
            is_raw[tmp[0]] = js.loads(fs_fusion.cat(f"{tmp[1]}/datasets/{tmp[0]}"))["isRawData"]
        ret.append(is_raw[tmp[0]])

    return ret


def path_to_url(x, is_raw=False):
    """Convert file name to fusion url.

    Args:
        x (str): File path.
        is_raw(bool, optional): Is the dataset raw.

    Returns (str): Fusion url string.

    """
    catalog, dataset, date, ext = _filename_to_distribution(x.split("/")[-1])
    ext = "raw" if is_raw else ext
    return "/".join(distribution_to_url("", dataset, date, ext, catalog).split("/")[1:])


def upload_files(
    fs_fusion,
    fs_local,
    loop,
    parallel=True,
    n_par=-1,
    multipart=True,
    chunk_size=5 * 2**20,
):
    """Upload file into Fusion.

    Args:
        fs_fusion: Fusion filesystem.
        fs_local: Local filesystem.
        loop (iterable): Loop of files to iterate through.
        parallel (bool): Is parallel mode enabled.
        n_par (int): Number of subprocesses.
        multipart (bool): Is multipart upload.
        chunk_size (int): Maximum chunk size.

    Returns: List of update statuses.

    """

    def _upload(row):
        p_url = row["url"]
        try:
            mp = multipart and fs_local.size(row["path"]) > chunk_size
            with fs_local.open(row["path"], "rb") as file_local:
                fs_fusion.put(
                    file_local, p_url, chunk_size=chunk_size, method="put", multipart=mp
                )
            return True, row["path"], None
        except Exception as ex:
            logger.log(
                VERBOSE_LVL,
                f'Failed to upload {row["path"]}. ex - {ex}',
            )
            return False, row["path"], str(ex)

    if parallel:
        res = Parallel(n_jobs=n_par)(delayed(_upload)(row) for index, row in loop)
    else:
        res = [_upload(row) for index, row in loop]
    return res
