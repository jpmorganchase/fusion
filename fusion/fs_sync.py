import pandas as pd
import numpy as np
import os
from os.path import splitext, relpath
from pathlib import Path
from .utils import distribution_to_filename, path_to_url, distribution_to_url, validate_file_names

from tqdm.auto import tqdm
from joblib import Parallel, delayed
import logging


logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
SYNC_PATH = ""
ALLOWED_FORMATS = ["csv", "parquet", "pq", "gz", "zip", "json"]


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
            fs_local.mkdir(Path(p_path).parent)
        try:
            with fs_local.open(p_path, "wb") as file_local:
                pass
                # fs_fusion.get(row["url"], file_local)
            return True, p_path, None
        except Exception as ex:
            logger.log(
                VERBOSE_LVL,
                f'Failed to write to {p_path}. ex - {ex}',
            )
            return False, p_path, ex

    # create local directories based on paths...
    if len(df) > 1:
        res = Parallel(n_jobs=-1)(delayed(_download)(row) for index, row in tqdm(df.iterrows()))
    else:
        res = [_download(row) for index, row in tqdm(df.iterrows())]

    return res


def upload(fs_fusion, fs_local, join_df):
    pass


def _get_fusion_df(fs_fusion, datasets_lst, catalog):
    df_lst = []
    for dataset in datasets_lst:
        changes = fs_fusion.info("")["changes"]["datasets"][0]["distributions"]
        keys = [catalog + "/datasets/" + "/".join(i["key"].split(".")) for i in changes]
        ts = [pd.Timestamp(i["values"][0]) for i in changes]
        sz = [int(i["values"][1]) for i in changes]
        md = [i["values"][2].split("md5=")[-1] for i in changes]
        urls = [path_to_url(i) for i in keys]
        df = pd.DataFrame([keys, urls, sz, ts, md]).T
        df.columns = ["path", "url", "size", "mtime", "md5"]
        df_lst.append(df)

    return pd.concat(df_lst)


def synchronize(fs_fusion, fs_local, dir_fusion: str, dir_local: str = "", direction: str = "upload"):
    """Synchronize two filesystems."""
    # check overall catalog level checksum

    local_files = fs_local.find(dir_local)
    # validate file names if not rename?
    local_rel_path = ["/".join(i.split("/")[i.split("/").index(dir_local):]) for i in local_files]
    local_file_validation = validate_file_names(local_rel_path, fs_fusion)
    local_files = [f for flag, f in zip(local_file_validation, local_files) if flag]

    local_files_rel = [os.path.join(dir_local, relpath(i, dir_local).replace("\\", "/")) for i in local_files]
    local_mtime = [fs_local.info(x)["mtime"] for x in local_files]
    local_md5 = [fs_local.ukey(x) for x in local_files]
    # local_ext = [splitext(i)[1][1:] for i in local_files]
    # local_datasets = [i.split("\\")[0] for i in local_files_rel]
    # local_dates = [i.split("\\")[1] for i in local_files_rel]
    local_url_eqiv = [path_to_url(i) for i in local_files]
    df_local = pd.DataFrame([local_files_rel, local_url_eqiv, local_mtime, local_md5]).T
    df_local.columns = ["path", "url", "mtime", "md5"]

    local_max_mtime = df_local.mtime.max() # TODO: this will go to request to fusion
    #fusion_df = pd.read_parquet(fs_fusion.cat(SYNC_PATH)) # TODO: specify path
    df_fusion = pd.read_csv("sample_fusion_checksum.csv")
    # fusion_files = fusion_df["url"] # TODO: change
    # If upload
    if direction == "upload":
        join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="left")
        join_df = join_df[join_df["mtime"] != join_df["mtime"]]
        res = upload(fs_fusion, fs_local, join_df)
    # If download
    elif direction == "download":
        join_df = df_local.merge(df_fusion, on="url", suffixes=("_local", "_fusion"), how="right")
        join_df = join_df[join_df["mtime_local"] != join_df["mtime_fusion"]]
        res = download(fs_fusion, fs_local, join_df)
    else:
        raise ValueError("Unknown direction of operation.")
    return res
