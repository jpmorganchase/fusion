import pandas as pd
import numpy as np
import os
from os.path import splitext, relpath
from pathlib import Path
import hashlib
import base64
from .utils import distribution_to_filename, path_to_url, validate_file_names

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import logging


logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
SYNC_PATH = ""
ALLOWED_FORMATS = ["csv", "parquet", "pq", "gz", "zip", "json"]
DEFAULT_CHUNK_SIZE = 2**16


def rename_to_convention(fs_local, dir_local):
    local_rel_path = ["/".join(i.split("/")[i.split("/").index(dir_local):]) for i in fs_local.find(dir_local)]
    file_names = [i.split("/")[-1].split(".")[0] for i in local_rel_path]
    local_ext = [splitext(i)[1][1:] for i in local_rel_path]
    for p, f_n, ext in zip(local_rel_path, file_names, local_ext):
        tmp = p.split("/")
        des_name = tmp[1] + "__" + tmp[0] + "__" + tmp[2]
        if des_name != f_n:
            new_path = os.path.join("/".join(p.split("/")[:-1]), des_name + "." + ext)
            logger.log(VERBOSE_LVL, f"Renaming {p} to {new_path}")
            fs_local.mv(p , new_path)


def validate_uniqueness_of_format(rel_paths):
    file_paths_no_file = np.asarray(["/".join(i.split("/")[:-1]) for i in rel_paths])
    local_ext = np.asarray([splitext(i)[1][1:] for i in rel_paths])
    counts_paths = [(file_paths_no_file == i).sum() for i in file_paths_no_file]
    validation = []
    for i, c in enumerate(counts_paths):
        if c > 1:
            idx = np.where(file_paths_no_file == file_paths_no_file[i])[0]
            exts = np.asarray(local_ext[idx])
            counts_exts = np.array([(exts == k).sum() for k in exts])
            if any(counts_exts > 1):
                validation.append(False)
            else:
                validation.append(True)
        else:
            validation.append(True)
    return validation


def url_to_path(x):
    file_name = distribution_to_filename("", x.split('/')[2], x.split('/')[4], x.split('/')[6], x.split("/")[0])
    return f"{x.split('/')[0]}/{x.split('/')[2]}/{x.split('/')[4]}/{file_name}"


def download(fs_fusion, fs_local, df):
    def _download(row):
        p_path = url_to_path(row["url"])
        if not fs_local.exists(p_path):
            try:
                fs_local.mkdir(Path(p_path).parent, exist_ok=True, create_parents=True)
            except Exception as ex:
                logger.info(f"Path {p_path} exists already", ex)
        try:
            fs_fusion.get(row["url"], p_path, chunk_size=DEFAULT_CHUNK_SIZE)
            return True, p_path, None
        except Exception as ex:
            logger.log(
                VERBOSE_LVL,
                f'Failed to write to {p_path}. ex - {ex}',
            )
            try:
                msg = ex.message
            except:
                msg = ""
            return False, p_path, msg

    # create local directories based on paths...
    if len(df) > 1:
        res = Parallel(n_jobs=-1)(delayed(_download)(row) for index, row in tqdm(df.iterrows(), total=len(df)))
    else:
        res = [_download(row) for index, row in tqdm(df.iterrows(), total=len(df))]

    return res


def upload(fs_fusion, fs_local, join_df):
    pass


def _generate_md5_token(path, fs):
    hash_md5 = hashlib.md5()
    with fs.open(path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_md5_chunk = hashlib.md5()
            hash_md5_chunk.update(chunk)
            hash_md5.update(chunk)

    return base64.b64encode(hash_md5.digest()).decode()


def _get_fusion_df(fs_fusion, datasets_lst, catalog):
    df_lst = []
    for dataset in datasets_lst:
        changes = fs_fusion.info(f"{catalog}/datasets/{dataset}")["changes"]["datasets"][0]["distributions"]
        urls = [catalog + "/datasets/" + "/".join(i["key"].split(".")) for i in changes]
        urls = [i.replace("distribution", "distributions") for i in urls]
        urls = ["/".join(i.split("/")[:3] + ["datasetseries"] + i.split("/")[3:]) if "datasetseries" not in i else i for i in urls]
        ts = [pd.Timestamp(i["values"][0]).timestamp() for i in changes]
        sz = [int(i["values"][1]) for i in changes]
        md = [i["values"][2].split("md5=")[-1] for i in changes]
        keys = [url_to_path(i) for i in urls]
        df = pd.DataFrame([keys, urls, sz, ts, md]).T
        df.columns = ["path", "url", "size", "mtime", "md5"]
        df_lst.append(df)

    return pd.concat(df_lst)


def synchronize(fs_fusion, fs_local, datasets, catalog, direction: str = "upload"):
    """Synchronize two filesystems."""
    # check overall catalog level checksum

    local_files = []
    local_files_rel = []
    local_dirs = [f"{catalog}/{i}" for i in datasets] if len(datasets)>0 else [catalog]
    # fusion_dirs = [f"{catalog}/datasets/{i}" for i in datasets] if len(datasets) > 0 else [catalog]

    for local_dir in local_dirs:
        if not fs_local.exists(local_dir):
            fs_local.mkdir(local_dir, exist_ok=True, create_parents=True)

        local_files = fs_local.find(local_dir)
        # validate file names if not rename?
        local_rel_path = [i[i.find(local_dir):] for i in local_files]
        local_file_validation = validate_file_names(local_rel_path, fs_fusion)
        local_files += [f for flag, f in zip(local_file_validation, local_files) if flag]
        local_files_rel += [os.path.join(local_dir, relpath(i, local_dir)).replace("\\", "/") for i in local_files]
    # datasets_lst = list(set([i.split("/")[-1].split("__")[0] for i in local_files_rel]))

    local_mtime = [fs_local.info(x)["mtime"] for x in local_files]
    local_md5 = [_generate_md5_token(x, fs_local) for x in local_files] # TODO: generate md5
    # local_ext = [splitext(i)[1][1:] for i in local_files]
    # local_datasets = [i.split("\\")[0] for i in local_files_rel]
    # local_dates = [i.split("\\")[1] for i in local_files_rel]
    local_url_eqiv = [path_to_url(i) for i in local_files]
    df_local = pd.DataFrame([local_files_rel, local_url_eqiv, local_mtime, local_md5]).T
    df_local.columns = ["path", "url", "mtime", "md5"]

    df_fusion = _get_fusion_df(fs_fusion, datasets, catalog)

    if direction == "upload":
        join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="left")
        join_df = join_df[join_df["md5_local"] != join_df["md5_fusion"]]
        res = upload(fs_fusion, fs_local, join_df)
    elif direction == "download":
        join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="right")
        join_df = join_df[join_df["md5_local"] != join_df["md5_fusion"]]
        res = download(fs_fusion, fs_local, join_df)
    else:
        raise ValueError("Unknown direction of operation.")
    return res
