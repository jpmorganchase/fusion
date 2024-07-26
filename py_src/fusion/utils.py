"""Fusion utilities."""

import contextlib
import json as js
import logging
import math
import multiprocessing as mp
import os
import re
import threading
from collections.abc import Generator
from contextlib import nullcontext
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Optional, Union
from urllib.parse import urlparse, urlunparse

import aiohttp
import fsspec
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from joblib import Parallel, delayed
from pyarrow import csv, json, unify_schemas
from pyarrow.parquet import filters_to_expression
from tqdm import tqdm
from urllib3.util.retry import Retry

from fusion._fusion import FusionCredentials

from .authentication import FusionAiohttpSession, FusionOAuthAdapter
from .types import PyArrowFilterT, WorkerQueueT

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
DT_YYYYMMDD_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")
DT_YYYYMMDDTHHMM_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{4})$")
DT_YYYY_MM_DD_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
DEFAULT_CHUNK_SIZE = 2**16
DEFAULT_THREAD_POOL_SIZE = 5


def get_default_fs() -> fsspec.filesystem:
    """Retrieve default filesystem.

    Returns: filesystem

    """
    protocol = os.environ.get("FS_PROTOCOL", "file")
    if "S3_ENDPOINT" in os.environ and "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
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


@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm) -> Generator[tqdm, None, None]:  # type: ignore
    # pragma: no cover
    """Progress bar sensitive to exceptions during the batch processing.

    Args:
        tqdm_object (tqdm.tqdm): tqdm object.

    Yields: tqdm.tqdm object

    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):  # type: ignore
        """Tqdm execution wrapper."""

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            n = 0
            lst = args[0]._result if hasattr(args[0], "_result") else args[0]
            for i in lst:
                try:
                    if i[0] is True:
                        n += 1
                except Exception as _:  # noqa: F841, PERF203, BLE001
                    n += 1
            tqdm_object.update(n=n)
            return super().__call__(*args, **kwargs)

    old_batch_callback: type[joblib.parallel.BatchCompletionCallBack] = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def cpu_count(thread_pool_size: Optional[int] = None, is_threading: bool = False) -> int:
    """Determine the number of cpus/threads for parallelization.

    Args:
        thread_pool_size (int): override argument for number of cpus/threads.
        is_threading: use threads instead of CPUs

    Returns: number of cpus/threads to use.

    """
    if os.environ.get("NUM_THREADS") is not None:
        return int(os.environ["NUM_THREADS"])
    if thread_pool_size:
        return thread_pool_size
    if is_threading:
        return 10

    thread_pool_size = mp.cpu_count() if mp.cpu_count() else DEFAULT_THREAD_POOL_SIZE
    return thread_pool_size


def csv_to_table(
    path: str,
    fs: Optional[fsspec.filesystem] = None,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
) -> pa.Table:
    """Reads csv data to pyarrow table.

    Args:
        path (str): path to the file.
        fs: filesystem object.
        columns: columns to read.
        filters: arrow filters.

    Returns:
        class:`pyarrow.Table` pyarrow table with the data.
    """
    filters = filters_to_expression(filters) if filters else filters
    with fs.open(path) if fs else nullcontext(path) as f:
        tbl = csv.read_csv(f)
        if filters is not None:
            tbl = tbl.filter(filters)
        if columns is not None:
            tbl = tbl.select(columns)
        return tbl


def json_to_table(
    path: str,
    fs: Optional[fsspec.filesystem] = None,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
) -> pa.Table:
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
    with fs.open(path) if fs else nullcontext(path) as f:
        tbl = json.read_json(f)
        if filters is not None:
            tbl = tbl.filter(filters)
        if columns is not None:
            tbl = tbl.select(columns)
        return tbl


PathLikeT = Union[str, Path]


def parquet_to_table(
    path: Union[PathLikeT, list[PathLikeT]],
    fs: Optional[fsspec.filesystem] = None,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
) -> pa.Table:
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
        schemas = [
            pq.ParquetDataset(
                p,
                use_legacy_dataset=False,
                filters=filters,
                filesystem=fs,
                memory_map=True,
            ).schema
            for p in path
        ]
    else:
        schemas = [
            pq.ParquetDataset(
                path,
                use_legacy_dataset=False,
                filters=filters,
                filesystem=fs,
                memory_map=True,
            ).schema
        ]

    schema = unify_schemas(schemas)
    return pq.ParquetDataset(
        path,
        use_legacy_dataset=False,
        filters=filters,
        filesystem=fs,
        memory_map=True,
        schema=schema,
    ).read(columns=columns)


def read_csv(  # noqa: PLR0912
    path: str,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
    fs: Optional[fsspec.filesystem] = None,
    dataframe_type: str = "pandas",
) -> Union[pd.DataFrame, pa.Table]:
    """Reads csv with possibility of selecting columns and filtering the data.

    Args:
        path (str): path to the csv file.
        columns: list of selected fields.
        filters: filters.
        fs: filesystem object.
        dataframe_type (str, optional): Datafame type pandas or polars

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.

    """
    try:
        try:
            res = csv_to_table(path, fs, columns=columns, filters=filters)

            if dataframe_type == "pandas":
                res = res.to_pandas()
            elif dataframe_type == "polars":
                import polars as pl

                res = pl.from_arrow(res)
            else:
                raise ValueError(f"Unknown DataFrame type {dataframe_type}")
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Failed to read {path}, with comma delimiter.",
                exc_info=True,
            )
            raise Exception from err

    except Exception:  # noqa: BLE001
        logger.log(
            VERBOSE_LVL,
            f"Could not parse {path} properly. Trying with pandas csv reader.",
            exc_info=True,
        )
        try:  # pragma: no cover
            with fs.open(path) if fs else nullcontext(path) as f:
                if dataframe_type == "pandas":
                    res = pd.read_csv(f, usecols=columns, index_col=False)
                elif dataframe_type == "polars":
                    import polars as pl

                    res = pl.read_csv(f, columns=columns)
                else:
                    raise ValueError(f"Unknown DataFrame type {dataframe_type}")

        except Exception as err:  # noqa: BLE001
            logger.log(
                VERBOSE_LVL,
                f"Could not parse {path} properly. " f"Trying with pandas csv reader pandas engine.",
                exc_info=True,
            )
            with fs.open(path) if fs else nullcontext(path) as f:
                if dataframe_type == "pandas":
                    res = pd.read_table(  # pragma: no cover
                        f,
                        usecols=columns,
                        index_col=False,
                        engine="python",
                        delimiter=None,
                    )
                else:
                    raise ValueError(f"Unknown DataFrame type {dataframe_type}") from err
    return res


def read_json(
    path: str,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
    fs: Optional[fsspec.filesystem] = None,
    dataframe_type: str = "pandas",
) -> Union[pd.DataFrame, pa.Table]:
    """Read json files(s) to pandas.

    Args:
        path (str): path or a list of paths to parquet files.
        columns (list): list of selected fields.
        filters (list): filters.
        fs: filesystem object.
        dataframe_type (str, optional): Datafame type pandas or polars

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.
    """

    try:
        try:
            res = json_to_table(path, fs, columns=columns, filters=filters)
            if dataframe_type == "pandas":
                res = res.to_pandas()
            elif dataframe_type == "polars":
                import polars as pl

                res = pl.from_arrow(res)
            else:
                raise ValueError(f"Unknown DataFrame type {dataframe_type}")
        except Exception as err:
            logger.log(
                VERBOSE_LVL,
                f"Failed to read {path}, with arrow reader. {err}",
            )
            raise err

    except Exception:  # noqa: BLE001
        logger.log(
            VERBOSE_LVL,
            f"Could not parse {path} properly. " f"Trying with pandas json reader.",
            exc_info=True,
        )
        try:  # pragma: no cover
            with fs.open(path) if fs else nullcontext(path) as f:
                if dataframe_type == "pandas":
                    res = pd.read_json(f)
                elif dataframe_type == "polars":
                    import polars as pl

                    res = pl.read_json(f)
                else:
                    raise ValueError(f"Unknown DataFrame type {dataframe_type}")

        except Exception as err:
            logger.log(VERBOSE_LVL, f"Could not parse {path} properly. ", exc_info=True)
            raise err  # pragma: no cover
    return res


def read_parquet(
    path: PathLikeT,
    columns: Optional[list[str]] = None,
    filters: Optional[PyArrowFilterT] = None,
    fs: Optional[fsspec.filesystem] = None,
    dataframe_type: str = "pandas",
) -> Union[pd.DataFrame, pa.Table]:
    """Read parquet files(s) to pandas.

    Args:
        path (PathLikeT): path or a list of paths to parquet files.
        columns (list): list of selected fields.
        filters (list): filters.
        fs: filesystem object.
        dataframe_type (str, optional): Datafame type pandas or polars

    Returns:
        Union[pandas.DataFrame, polars.DataFrame]: a dataframe containing the data.

    """

    tbl = parquet_to_table(path, columns=columns, filters=filters, fs=fs)
    if dataframe_type == "pandas":
        return tbl.to_pandas()
    if dataframe_type == "polars":
        import polars as pl

        return pl.from_arrow(tbl)
    else:
        raise ValueError(f"Unknown DataFrame type {dataframe_type}")


def _normalise_dt_param(dt: Union[str, int, datetime, date]) -> str:
    """Convert dates into a normalised string representation.

    Args:
        dt (Union[str, int, datetime, date]): A date represented in various types.

    Returns:
        str: A normalized date string.
    """
    if isinstance(dt, (date, datetime)):
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

    matches = DT_YYYYMMDDTHHMM_RE.match(dt)

    if matches:
        return "-".join(matches.groups())

    raise ValueError(f"{dt} is not in a recognised data format")


def normalise_dt_param_str(dt: str) -> tuple[str, ...]:
    """Convert a date parameter which may be a single date or a date range into a tuple.

    Args:
        dt (str): Either a single date or a date range separated by a ":".

    Returns:
        tuple: A tuple of dates.
    """
    date_parts = dt.split(":")
    max_date_seg_cnt = 2
    if not date_parts or len(date_parts) > max_date_seg_cnt:
        raise ValueError(f"Unable to parse {dt} as either a date or an interval")

    return tuple(_normalise_dt_param(dt_part) if dt_part else dt_part for dt_part in date_parts)


def distribution_to_filename(
    root_folder: str,
    dataset: str,
    datasetseries: str,
    file_format: str,
    catalog: str = "common",
    partitioning: Optional[str] = None,
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


def _filename_to_distribution(file_name: str) -> tuple[str, str, str, str]:
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
    is_download: bool = False,
) -> str:
    """Returns the API URL to download a dataset distribution.

    Args:
        root_url (str): The base url for the API.
        dataset (str): The dataset identifier.
        datasetseries (str): The datasetseries instance identifier.
        file_format (str): The file type, e.g. CSV or Parquet
        catalog (str, optional): The data catalog containing the dataset. Defaults to "common".
        is_download (bool, optional): Is url for download

    Returns:
        str: A URL for the API distribution endpoint.
    """
    if datasetseries[-1] == "/" or datasetseries[-1] == "\\":
        datasetseries = datasetseries[0:-1]

    if datasetseries == "sample":
        return f"{root_url}catalogs/{catalog}/datasets/{dataset}/sample/distributions/csv"
    if is_download:
        return (
            f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/"
            f"{datasetseries}/distributions/{file_format}/operationType/download"
        )
    return (
        f"{root_url}catalogs/{catalog}/datasets/{dataset}/datasetseries/" f"{datasetseries}/distributions/{file_format}"
    )


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


async def get_client(credentials: FusionCredentials, **kwargs: Any) -> FusionAiohttpSession:  # noqa: PLR0915
    """Gets session for async.

    Args:
        credentials: Credentials.
        **kwargs: Kwargs.

    Returns:

    """

    async def on_request_start_token(session: Any, _trace_config_ctx: Any, params: Any) -> None:
        if params.url:
            params.headers.update(session.credentials.get_fusion_token_headers(str(params.url)))

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_start.append(on_request_start_token)

    if "timeout" in kwargs:
        timeout = aiohttp.ClientTimeout(total=kwargs["timeout"])
    else:
        timeout = aiohttp.ClientTimeout(total=60 * 60)  # default 60min timeout
    session = FusionAiohttpSession(trace_configs=[trace_config], trust_env=True, timeout=timeout)
    session.post_init(credentials=credentials)
    return session


def get_session(
    credentials: FusionCredentials, root_url: str, get_retries: Optional[Union[int, Retry]] = None
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
    if credentials.proxies:
        session.proxies.update(credentials.proxies)
    try:
        mount_url = _get_canonical_root_url(root_url)
    except Exception:  # noqa: BLE001
        mount_url = "https://"
    auth_handler = FusionOAuthAdapter(credentials, max_retries=get_retries, mount_url=mount_url)
    session.mount(mount_url, auth_handler)
    return session


def _stream_single_file_new_session_dry_run(
    credentials: FusionCredentials, url: str, output_file: str
) -> tuple[bool, str, Optional[str]]:
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
        return (True, output_file, None)
    except BaseException as ex:  # noqa: BLE001
        logger.log(VERBOSE_LVL, f"Failed to download url {url} to {output_file}", exc_info=True)
        return (False, output_file, str(ex))


def stream_single_file_new_session_chunks(  # noqa: PLR0913
    session: requests.Session,
    url: str,
    output_file: fsspec.spec.AbstractBufferedFile,
    start: int,
    end: int,
    lock: threading.Lock,
    results: list[tuple[bool, str, Optional[str]]],
    idx: int,
    overwrite: bool = True,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> int:
    """Function to stream a single file from the API to a file on disk.

    Args:
        session (class `requests.Session`): HTTP session.
        url (str): The URL to call.
        output_file: The file handle for the target write file.
        start (int): Start byte.
        end(int): End byte.
        lock (Threading.Lock): Lock.
        results (list): Results list.
        idx (int): Results list index.
        overwrite (bool, optional): True if previously downloaded files should be overwritten. Defaults to True.
        fs (fsspec.filesystem): Filesystem.

    Returns:
        int: Exit status

    """
    if fs is None:
        fs = fsspec.filesystem("file")
    if not overwrite and fs.exists(output_file):
        results[idx] = True, output_file, None
        return 0

    try:
        url = url + f"?downloadRange=bytes={start}-{end-1}"
        with session.get(url, stream=False) as r:
            r.raise_for_status()
            with lock:
                output_file.seek(start)
                output_file.write(r.content)

        logger.log(
            VERBOSE_LVL,
            f"Wrote {start} - {end} bytes to {output_file}",
        )
        results[idx] = (True, output_file, None)
        return 0
    except Exception as ex:  # noqa: BLE001
        logger.log(
            VERBOSE_LVL,
            f"Failed to write to {output_file}.",
            exc_info=True,
        )
        results[idx] = (False, output_file, str(ex))
        return 1


def _worker(
    queue: WorkerQueueT,
    session: requests.Session,
    url: str,
    output_file: str,
    lock: threading.Lock,
    results: list[tuple[bool, str, Optional[str]]],
) -> None:
    while True:
        idx, start, end = queue.get()
        if idx == -1 and start == -1 and end == -1:
            break
        stream_single_file_new_session_chunks(session, url, output_file, start, end, lock, results, idx)
        queue.task_done()


def download_single_file_threading(
    credentials: FusionCredentials,
    url: str,
    output_file: fsspec.spec.AbstractBufferedFile,
    chunk_size: int = 5 * 2**20,
    fs: Optional[fsspec.AbstractFileSystem] = None,
    max_threads: int = 10,
) -> list[tuple[bool, str, Optional[str]]]:
    """Download single file using range requests.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an access token
        url (str): The URL to call.
        output_file (str): The filename that the data will be saved into.
        chunk_size (int): Chunk size for parallelization.
        fs (fsspec.filesystem): Filesystem.
        max_threads (int, optional): Number of threads to use. Defaults to 10.

    Returns: List[Tuple]

    """
    if fs is None:
        fs = fsspec.filesystem("file")
    session = get_session(credentials, url)
    header = session.head(url).headers
    content_length = int(header["Content-Length"])
    n_chunks = int(math.ceil(content_length / chunk_size))
    starts = [i * chunk_size for i in range(n_chunks)]
    ends = [min((i + 1) * chunk_size, content_length) for i in range(n_chunks)]
    lock = Lock()
    output_file_h: fsspec.spec.AbstractBufferedFile = fs.open(output_file, "wb")
    results = [None] * n_chunks
    queue: WorkerQueueT = Queue(max_threads)
    threads = []
    for _ in range(max_threads):
        t = Thread(target=_worker, args=(queue, session, url, output_file_h, lock, results))
        t.start()
        threads.append(t)

    for idx, (start, end) in enumerate(zip(starts, ends)):
        queue.put((idx, start, end))

    queue.join()

    for _ in range(max_threads):
        queue.put((-1, -1, -1))
    for t in threads:
        t.join()

    output_file_h.close()
    return results  # type: ignore


def stream_single_file_new_session(
    credentials: FusionCredentials,
    url: str,
    output_file: str,
    overwrite: bool = True,
    block_size: int = DEFAULT_CHUNK_SIZE,
    dry_run: bool = False,
    fs: Optional[fsspec.AbstractFileSystem] = None,
) -> tuple[bool, str, Optional[str]]:
    """Function to stream a single file from the API to a file on disk.

    Args:
        credentials (FusionCredentials): Valid user credentials to provide an access token
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
    if fs is None:
        fs = fsspec.filesystem("file")
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
    except Exception as ex:  # noqa: BLE001
        logger.log(
            VERBOSE_LVL,
            f"Failed to write to {output_file}.",
            exc_info=True,
        )
        return (False, output_file, str(ex))


def validate_file_names(paths: list[str], fs_fusion: fsspec.AbstractFileSystem) -> list[bool]:
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
    file_seg_cnt = 3
    for i, f_n in enumerate(file_names):
        tmp = f_n.split("__")
        if len(tmp) == file_seg_cnt:
            val = tmp[1] in all_catalogs
            if not val:
                validation.append(False)
            else:
                if tmp[1] not in all_datasets:
                    all_datasets[tmp[1]] = [i.split("/")[-1] for i in fs_fusion.ls(f"{tmp[1]}/datasets")]

                val = tmp[0] in all_datasets[tmp[1]]
                validation.append(val)
        else:
            validation.append(False)
        if not validation[-1] and len(tmp) == file_seg_cnt:
            logger.warning(
                f"You might not have access to the catalog {tmp[1]} or dataset {tmp[0]}."
                "Please check you permission on the platform."
            )
        if not validation[-1] and len(tmp) != file_seg_cnt:
            logger.warning(
                f"The file in {paths[i]} has a non-compliant name and will not be processed. "
                f"Please rename the file to dataset__catalog__yyyymmdd.format"
            )

    return validation


def is_dataset_raw(paths: list[str], fs_fusion: fsspec.AbstractFileSystem) -> list[bool]:
    """Check if the files correspond to a raw dataset.

    Args:
        paths (list): List of file paths.
        fs_fusion: Fusion filesystem.

    Returns (list): List of booleans.

    """
    file_names = [i.split("/")[-1].split(".")[0] for i in paths]
    ret = []
    is_raw = {}
    for _i, f_n in enumerate(file_names):
        tmp = f_n.split("__")
        if tmp[0] not in is_raw:
            is_raw[tmp[0]] = js.loads(fs_fusion.cat(f"{tmp[1]}/datasets/{tmp[0]}"))["isRawData"]
        ret.append(is_raw[tmp[0]])

    return ret


def path_to_url(x: str, is_raw: bool = False, is_download: bool = False) -> str:
    """Convert file name to fusion url.

    Args:
        x (str): File path.
        is_raw(bool, optional): Is the dataset raw.
        is_download(bool, optional): Is the url for download.

    Returns (str): Fusion url string.

    """
    catalog, dataset, date, ext = _filename_to_distribution(x.split("/")[-1])
    ext = "raw" if is_raw else ext
    return "/".join(distribution_to_url("", dataset, date, ext, catalog, is_download).split("/")[1:])


def upload_files(  # noqa: PLR0913
    fs_fusion: fsspec.AbstractFileSystem,
    fs_local: fsspec.AbstractFileSystem,
    loop: pd.DataFrame,
    parallel: bool = True,
    n_par: int = -1,
    multipart: bool = True,
    chunk_size: int = 5 * 2**20,
    show_progress: bool = True,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> list[tuple[bool, str, Optional[str]]]:
    """Upload file into Fusion.

    Args:
        fs_fusion: Fusion filesystem.
        fs_local: Local filesystem.
        loop (pd.DataFrame): DataFrame of files to iterate through.
        parallel (bool): Is parallel mode enabled.
        n_par (int): Number of subprocesses.
        multipart (bool): Is multipart upload.
        chunk_size (int): Maximum chunk size.
        show_progress (bool): Show progress bar
        from_date (str, optional): earliest date of data contained in distribution.
        to_date (str, optional): latest date of data contained in distribution.

    Returns: List of update statuses.

    """

    def _upload(p_url: str, path: str) -> tuple[bool, str, Optional[str]]:
        try:
            mp = multipart and fs_local.size(path) > chunk_size

            if isinstance(fs_local, BytesIO):
                fs_fusion.put(
                    fs_local,
                    p_url,
                    chunk_size=chunk_size,
                    method="put",
                    multipart=mp,
                    from_date=from_date,
                    to_date=to_date,
                )
            else:
                with fs_local.open(path, "rb") as file_local:
                    fs_fusion.put(
                        file_local,
                        p_url,
                        chunk_size=chunk_size,
                        method="put",
                        multipart=mp,
                        from_date=from_date,
                        to_date=to_date,
                    )
            return (True, path, None)
        except Exception as ex:  # noqa: BLE001
            logger.log(
                VERBOSE_LVL,
                f"Failed to upload {path}.",
                exc_info=True,
            )
            return (False, path, str(ex))

    if parallel:
        if show_progress:
            with tqdm_joblib(tqdm(total=len(loop))) as _:
                res = Parallel(n_jobs=n_par, backend="threading")(
                    delayed(_upload)(row["url"], row["path"]) for _, row in loop.iterrows()
                )
        else:
            res = Parallel(n_jobs=n_par, backend="threading")(
                delayed(_upload)(row["url"], row["path"]) for _, row in loop.iterrows()
            )
    else:
        res = [None] * len(loop)
        if show_progress:
            with tqdm(total=len(loop)) as p:
                for i, (_, row) in enumerate(loop.iterrows()):
                    r = _upload(row["url"], row["path"])
                    res[i] = r
                    if r[0] is True:
                        p.update(1)
        else:
            res = [_upload(row["url"], row["path"]) for _, row in loop.iterrows()]

    return res  # type: ignore
