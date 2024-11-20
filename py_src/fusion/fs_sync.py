"""Fusion fsync."""

import base64
import hashlib
import json
import logging
import sys
import time
import warnings
from os.path import relpath
from pathlib import Path
from typing import Optional

import fsspec
import pandas as pd
from joblib import Parallel, delayed

from .utils import (
    cpu_count,
    distribution_to_filename,
    is_dataset_raw,
    joblib_progress,
    path_to_url,
    upload_files,
    validate_file_names,
)

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
DEFAULT_CHUNK_SIZE = 2**16


def _url_to_path(x: str) -> str:
    file_name = distribution_to_filename("", x.split("/")[2], x.split("/")[4], x.split("/")[6], x.split("/")[0])
    return f"{x.split('/')[0]}/{x.split('/')[2]}/{x.split('/')[4]}/{file_name}"


def _download(
    fs_fusion: fsspec.filesystem,
    fs_local: fsspec.filesystem,
    df: pd.DataFrame,
    n_par: int,
    show_progress: bool = True,
    local_path: str = "",
) -> list[tuple[bool, str, Optional[str]]]:
    if len(df) > 0:
        if show_progress:
            with joblib_progress("Downloading", total=len(df)):
                res = Parallel(n_jobs=n_par)(
                    delayed(fs_fusion.download)(fs_local, row["url"], local_path + row["path_fusion"])
                    for i, row in df.iterrows()
                )
        else:
            res = Parallel(n_jobs=n_par)(
                delayed(fs_fusion.download)(fs_local, row["url"], local_path + row["path_fusion"])
                for i, row in df.iterrows()
            )
    else:
        return []

    return res  # type: ignore


def _upload(
    fs_fusion: fsspec.filesystem,
    fs_local: fsspec.filesystem,
    df: pd.DataFrame,
    n_par: int,
    show_progress: bool = True,
    local_path: str = "",
) -> list[tuple[bool, str, Optional[str]]]:
    upload_df = df.rename(columns={"path_local": "path"})
    upload_df["path"] = [Path(local_path) / p for p in upload_df["path"]]
    parallel = len(df) > 1
    res = upload_files(
        fs_fusion,
        fs_local,
        upload_df,
        parallel=parallel,
        n_par=n_par,
        multipart=True,
        show_progress=show_progress,
    )

    return res


def _generate_sha256_token(path: str, fs: fsspec.filesystem, chunk_size: int = 5 * 2**20) -> str:
    hash_sha256 = hashlib.sha256()
    hash_sha256_chunk = hashlib.sha256()

    chunk_count = 0
    with fs.open(path, "rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            hash_sha256_chunk = hashlib.sha256()
            hash_sha256_chunk.update(chunk)
            hash_sha256.update(hash_sha256_chunk.digest())
            chunk_count += 1

    if chunk_count > 1:
        return base64.b64encode(hash_sha256.digest()).decode()
    else:
        return base64.b64encode(hash_sha256_chunk.digest()).decode()


def _get_fusion_df(
    fs_fusion: fsspec.filesystem,
    datasets_lst: list[str],
    catalog: str,
    flatten: bool = False,
    dataset_format: Optional[str] = None,
) -> pd.DataFrame:
    df_lst = []
    for dataset in datasets_lst:
        changes = fs_fusion.info(f"{catalog}/datasets/{dataset}")["changes"]["datasets"]
        if len(changes) > 0:
            changes = changes[0]["distributions"]
            keys = [i["key"].replace(".", "/").split("/") for i in changes]
            keys = ["/".join([k[0], k[1], k[2], k[-1]]) for k in keys]
            urls = [catalog + "/datasets/" + k for k in keys]
            urls = [i.replace("distribution", "distributions") for i in urls]
            urls = [
                ("/".join(i.split("/")[:3] + ["datasetseries"] + i.split("/")[3:]) if "datasetseries" not in i else i)
                for i in urls
            ]
            sz = [int(i["values"][1]) for i in changes]
            md = [i["values"][2].split("SHA-256=")[-1][:44] for i in changes]
            keys = [_url_to_path(i) for i in urls]

            if flatten:
                keys = ["/".join(k.split("/")[:2] + k.split("/")[-1:]) for k in keys]

            info_df = pd.DataFrame([keys, urls, sz, md]).T
            info_df.columns = pd.Index(["path", "url", "size", "sha256"])
            if dataset_format and len(info_df) > 0:
                info_df = info_df[info_df.url.str.split("/").str[-1] == dataset_format]
            df_lst.append(info_df)
        else:
            df_lst.append(pd.DataFrame(columns=["path", "url", "size", "sha256"]))

    return pd.concat(df_lst)


def _get_local_state(
    fs_local: fsspec.filesystem,
    fs_fusion: fsspec.filesystem,
    datasets: list[str],
    catalog: str,
    dataset_format: Optional[str] = None,
    local_state: Optional[pd.DataFrame] = None,
    local_path: str = "",
) -> pd.DataFrame:
    local_files = []
    local_files_rel = []
    local_dirs = [f"{local_path}{catalog}/{i}" for i in datasets] if len(datasets) > 0 else [local_path + catalog]

    for local_dir in local_dirs:
        if not fs_local.exists(local_dir):
            fs_local.mkdir(local_dir, exist_ok=True, create_parents=True)

        local_files_temp = fs_local.find(local_dir)
        local_rel_path = [i[i.find(local_dir) :] for i in local_files_temp]
        local_file_validation = validate_file_names(local_rel_path, fs_fusion)
        local_files += [f for flag, f in zip(local_file_validation, local_files_temp) if flag]
        local_files_rel += [
            Path(local_dir, relpath(loc_file, local_dir).replace("\\", "/").replace(local_path, ""))
            for loc_file in local_files_temp
        ]

    local_mtime = [fs_local.info(x)["mtime"] for x in local_files]
    is_raw_lst = is_dataset_raw(local_files, fs_fusion)
    local_url_eqiv = [path_to_url(i, r) for i, r in zip(local_files, is_raw_lst)]
    df_local = pd.DataFrame([local_files_rel, local_url_eqiv, local_mtime, local_files]).T
    df_local.columns = pd.Index(["path", "url", "mtime", "local_path"])

    if local_state is not None and len(local_state) > 0:
        df_join = df_local.merge(local_state, on="path", how="left", suffixes=("", "_prev"))
        df_join.loc[df_join["mtime"] != df_join["mtime_prev"], "sha256"] = [
            _generate_sha256_token(x, fs_local) for x in df_join[df_join["mtime"] != df_join["mtime_prev"]].local_path
        ]
        df_local = df_join[["path", "url", "mtime", "sha256"]]
    else:
        df_local["sha256"] = [_generate_sha256_token(x, fs_local) for x in local_files]

    if dataset_format and len(df_local) > 0:
        df_local = df_local[df_local.url.str.split("/").str[-1] == dataset_format]

    df_local = df_local.sort_values("path").drop_duplicates()
    return df_local


def _synchronize(  # noqa: PLR0913
    fs_fusion: fsspec.filesystem,
    fs_local: fsspec.filesystem,
    df_local: pd.DataFrame,
    df_fusion: pd.DataFrame,
    direction: str = "upload",
    n_par: Optional[int] = None,
    show_progress: bool = True,
    local_path: str = "",
) -> list[tuple[bool, str, Optional[str]]]:
    """Synchronize two filesystems."""

    n_par = cpu_count(n_par)
    if direction == "upload":
        if len(df_local) == 0:
            msg = "No dataset members available for upload for your dataset selection."
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            res = []
        else:
            join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="left")
            join_df = join_df[join_df["sha256_local"] != join_df["sha256_fusion"]]
            res = _upload(
                fs_fusion,
                fs_local,
                join_df,
                n_par,
                show_progress=show_progress,
                local_path=local_path,
            )
    elif direction == "download":
        if len(df_fusion) == 0:
            msg = "No dataset members available for download for your dataset selection."
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            res = []
        else:
            join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="right")
            join_df = join_df[join_df["sha256_local"] != join_df["sha256_fusion"]]
            res = _download(
                fs_fusion,
                fs_local,
                join_df,
                n_par,
                show_progress=show_progress,
                local_path=local_path,
            )
    else:
        raise ValueError("Unknown direction of operation.")
    return res


def fsync(  # noqa: PLR0913
    fs_fusion: fsspec.filesystem,
    fs_local: fsspec.filesystem,
    products: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    catalog: Optional[str] = None,
    direction: str = "upload",
    flatten: bool = False,
    dataset_format: Optional[str] = None,
    n_par: Optional[int] = None,
    show_progress: bool = True,
    local_path: str = "",
    log_level: int = logging.ERROR,
    log_path: str = ".",
) -> None:
    """Synchronisation between the local filesystem and Fusion.

    Args:
        fs_fusion (fsspec.filesystem): Fusion filesystem.
        fs_local (fsspec.filesystem): Local filesystem.
        products (list): List of products.
        datasets (list): List of datasets.
        catalog (str): Fusion catalog.
        direction (str): Direction of synchronisation: upload/download.
        flatten (bool): Flatten the folder structure.
        dataset_format (str): Dataset format for upload/download.
        n_par (int, optional): Specify how many distributions to download in parallel. Defaults to all.
        show_progress (bool): Display a progress bar during data download Defaults to True.
        local_path (str): path to files in the local filesystem, e.g., "s3a://my_bucket/"
        log_level (int): Logging level. Error level by default.
        log_path (str): The folder path where the log is stored. Defaults to ".".

    Returns:

    """

    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(filename="{}/{}".format(log_path, "fusion_fsync.log"))
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

    catalog = catalog if catalog else "common"
    datasets = datasets if datasets else []
    products = products if products else []

    assert len(products) > 0 or len(datasets) > 0, "At least one list products or datasets should be non-empty."
    assert direction in [
        "upload",
        "download",
    ], "The direction must be either upload or download."

    if len(local_path) > 0 and local_path[-1] != "/":
        local_path += "/"

    for product in products:
        res = json.loads(fs_fusion.cat(f"{catalog}/products/{product}").decode())
        datasets += [r["identifier"] for r in res["resources"]]

    assert len(datasets) > 0, "The supplied products did not contain any datasets."

    local_state = pd.DataFrame()
    fusion_state = pd.DataFrame()
    while True:
        try:
            local_state_temp = _get_local_state(
                fs_local,
                fs_fusion,
                datasets,
                catalog,
                dataset_format,
                local_state,
                local_path,
            )
            fusion_state_temp = _get_fusion_df(fs_fusion, datasets, catalog, flatten, dataset_format)
            if not local_state_temp.equals(local_state) or not fusion_state_temp.equals(fusion_state):
                res = _synchronize(
                    fs_fusion,
                    fs_local,
                    local_state_temp,
                    fusion_state_temp,
                    direction,
                    n_par,
                    show_progress,
                    local_path,
                )
                if len(res) == 0 or all(i[0] for i in res):
                    local_state = local_state_temp
                    fusion_state = fusion_state_temp

                if not all(r[0] for r in res):
                    failed_res = [r for r in res if not r[0]]
                    msg = f"Not all {direction}s were successfully completed. The following failed:\n{failed_res}"
                    errs = [r for r in res if not r[2]]
                    logger.warning(msg)
                    logger.warning(errs)
                    warnings.warn(msg, stacklevel=2)

            else:
                logger.info("All synced, sleeping")
                time.sleep(10)

        except KeyboardInterrupt:  # noqa: PERF203
            if input("Type exit to exit: ") != "exit":
                continue
            break

        except Exception as _:
            logger.error("Exception thrown", exc_info=True)
            continue
