"""Fusion FileSystem."""

import base64
import hashlib
import io
import logging
from collections.abc import AsyncGenerator, Generator
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import quote, urljoin

import fsspec
import pandas as pd
import requests
from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.implementations.http import HTTPFile, HTTPFileSystem, sync, sync_wrapper
from fsspec.utils import nullcontext

from fusion._fusion import FusionCredentials

from .utils import get_client

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25


class FusionHTTPFileSystem(HTTPFileSystem):  # type: ignore
    """Fusion HTTP filesystem."""

    def __init__(
        self,
        credentials: Optional[Union[str, FusionCredentials]] = "config/client_credentials.json",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Same signature as the fsspec HTTPFileSystem.

        Args:
            credentials: Credentials.
            *args: Args.
            **kwargs: Kwargs.
        """

        if "get_client" not in kwargs:
            kwargs["get_client"] = get_client
        if "client_kwargs" not in kwargs:
            if isinstance(credentials, FusionCredentials):
                self.credentials = credentials
            elif isinstance(credentials, str):
                self.credentials = FusionCredentials.from_file(Path(credentials))
            kwargs["client_kwargs"] = {
                "credentials": self.credentials,
                "root_url": "https://fusion.jpmorgan.com/api/v1/",
            }
        elif "client_kwargs" in kwargs and isinstance(kwargs["client_kwargs"]["credentials"], FusionCredentials):
            self.credentials = kwargs["client_kwargs"]["credentials"]
        else:
            raise ValueError("Credentials not provided")

        if self.credentials.proxies:
            if "http" in self.credentials.proxies:
                kwargs["proxy"] = self.credentials.proxies["http"]
            elif "https" in self.credentials.proxies:
                kwargs["proxy"] = self.credentials.proxies["https"]

        if "headers" not in kwargs:
            kwargs["headers"] = {"Accept-Encoding": "identity"}

        super().__init__(*args, **kwargs)

    async def _async_raise_not_found_for_status(self, response: Any, url: str) -> None:
        """Raises FileNotFoundError for 404s, otherwise uses raise_for_status."""
        if response.status == requests.codes.not_found:  # noqa: PLR2004
            self._raise_not_found_for_status(response, url)
        else:
            real_reason = ""
            try:
                real_reason = await response.text()
                response.reason = real_reason
            finally:
                self._raise_not_found_for_status(response, url)

    async def _decorate_url_a(self, url: str) -> str:
        url = urljoin(f'{self.client_kwargs["root_url"]}catalogs/', url) if "http" not in url else url
        return url

    def _decorate_url(self, url: str) -> str:
        url = urljoin(f'{self.client_kwargs["root_url"]}catalogs/', url) if "http" not in url else url
        url = url[:-1] if url[-1] == "/" else url
        return url

    async def _isdir(self, path: str) -> bool:
        path = self._decorate_url(path)
        try:
            ret: dict[str, str] = await self._info(path)
            return ret["type"] == "directory"
        except BaseException:  # pragma: no cover
            logger.exception(VERBOSE_LVL, f"Artificial error")
            return False

    async def _changes(self, url: str) -> dict[Any, Any]:
        url = self._decorate_url(url)
        try:
            session = await self.set_session()
            async with session.get(url, **self.kwargs) as r:
                self._raise_not_found_for_status(r, url)
                try:
                    out: dict[Any, Any] = await r.json()
                except BaseException:
                    logger.exception(VERBOSE_LVL, f"{url} cannot be parsed to json")
                    out = {}
            return out
        except Exception as ex:
            logger.log(VERBOSE_LVL, f"Artificial error, {ex}")
            raise ex

    async def _ls_real(self, url: str, detail: bool = True, **kwargs: Any) -> Any:
        # ignoring URL-encoded arguments
        clean_url = url
        if "http" not in url:
            url = f'{self.client_kwargs["root_url"]}catalogs/' + url
        kw = self.kwargs.copy()
        kw.update(kwargs)
        session = await self.set_session()
        is_file = False
        size = None
        url = url if url[-1] != "/" else url[:-1]
        url_parts = url.split("/")
        if url_parts[-2] == "distributions":
            async with session.head(url + "/operationType/download", **self.kwargs) as r:
                self._raise_not_found_for_status(r, url)
                out = [
                    url.split("/")[6] + "-" + url.split("/")[8] + "-" + url.split("/")[10] + "." + url.split("/")[-1]
                ]

                size = int(r.headers["Content-Length"])
                is_file = True
        else:
            async with session.get(url, **self.kwargs) as r:
                self._raise_not_found_for_status(r, url)
                out = await r.json()

        if not is_file:
            out = [urljoin(clean_url + "/", x["identifier"]) for x in out["resources"]]  # type: ignore

        if detail:
            if not is_file:
                return [
                    {
                        "name": u,
                        "size": None,
                        "type": ("directory" if not (u.endswith(("csv", "parquet"))) else "file"),
                    }
                    for u in out
                ]
            else:
                return [
                    {
                        "name": out[0],
                        "size": size,
                        "type": "file",
                    }
                ]
        else:
            return out

    def info(self, path: str, **kwargs: Any) -> Any:
        """Return info.

        Args:
            path: Path.
            **kwargs: Kwargs.

        Returns:

        """
        path = self._decorate_url(path)
        kwargs["keep_protocol"] = True
        res = super().ls(path, detail=True, **kwargs)
        if res[0]["type"] != "file":
            res = super().info(path, **kwargs)
            if path.split("/")[-2] == "datasets":
                target = path.split("/")[-1]
                args = ["/".join(path.split("/")[:-1]) + f"/changes?datasets={quote(target)}"]
                res["changes"] = sync(super().loop, self._changes, *args)
        split_path = path.split("/")
        if len(split_path) > 1 and split_path[-2] == "distributions":
            res = res[0]

        return res

    async def _ls(self, url: str, detail: bool = True, **kwargs: Any) -> Any:
        url = await self._decorate_url_a(url)
        return await super()._ls(url, detail, **kwargs)

    def ls(self, url: str, detail: bool = False, **kwargs: Any) -> Any:
        """List resources.

        Args:
            url: Url.
            detail: Detail.
            **kwargs: Kwargs.

        Returns:

        """
        url = self._decorate_url(url)
        ret = super().ls(url, detail=detail, **kwargs)
        keep_protocol = kwargs.pop("keep_protocol", False)
        if detail:
            if not keep_protocol:
                for k in ret:
                    k["name"] = k["name"].split(f'{self.client_kwargs["root_url"]}catalogs/')[-1]

        elif not keep_protocol:
            return [x.split(f'{self.client_kwargs["root_url"]}catalogs/')[-1] for x in ret]

        return ret

    def exists(self, url: str, detail: bool = True, **kwargs: Any) -> Any:
        """Check existence.

        Args:
            url: Url.
            detail: Detail.
            **kwargs: Kwargs.

        Returns:

        """
        url = self._decorate_url(url)
        return super().exists(url, detail, **kwargs)

    def isfile(self, path: str) -> Union[bool, Any]:
        """Is path a file.

        Args:
            path: Path.

        Returns:

        """
        path = self._decorate_url(path)
        return super().isfile(path)

    def cat(self, url: str, start: Optional[int] = None, end: Optional[int] = None, **kwargs: Any) -> Any:
        """Fetch paths' contents.

        Args:
            url: Url.
            start: Start.
            end: End.
            **kwargs: Kwargs.

        Returns:

        """
        url = self._decorate_url(url)
        return super().cat(url, start=start, end=end, **kwargs)

    def get(
        self,
        rpath: str,
        lpath: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        **kwargs: Any,
    ) -> Any:
        """Copy file(s) to local.

        Args:
            rpath: Rpath.
            lpath: Lpath.
            chunk_size: Chunk size.
            callback: Callback function.
            **kwargs: Kwargs.

        Returns:

        """
        rpath = self._decorate_url(rpath)
        return super().get(rpath, lpath, chunk_size=chunk_size, callback=callback, **kwargs)

    async def _put_file(
        self,
        lpath: Union[str, io.IOBase],
        rpath: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        method: str = "post",
        multipart: bool = False,
        **kwargs: Any,
    ) -> None:
        async def put_data() -> AsyncGenerator[dict[Any, Any], None]:
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")  # noqa: SIM115, PTH123, ASYNC101
                use_seek = True

            with context as f:
                if use_seek:
                    callback.set_size(f.seek(0, 2))
                    f.seek(0)
                else:
                    callback.set_size(getattr(f, "size", None))

                chunk = f.read(chunk_size)
                i = 0
                while chunk:
                    kw = self.kwargs.copy()
                    url = rpath + f"/operations/upload?operationId={operation_id}&partNumber={i+1}"
                    kw.update({"headers": kwargs["chunk_headers_lst"][i]})
                    async with meth(url=url, data=chunk, **kw) as resp:
                        await self._async_raise_not_found_for_status(resp, rpath)
                        yield await resp.json()
                    i += 1
                    callback.relative_update(len(chunk))
                    chunk = f.read(chunk_size)

        session = await self.set_session()

        method = method.lower()
        if method not in ("post", "put"):
            raise ValueError(f"method has to be either 'post' or 'put', not: {method!r}")

        headers = kwargs["headers"]

        meth = getattr(session, method)
        if not multipart:
            kw = self.kwargs.copy()
            kw.update({"headers": headers})
            if isinstance(lpath, io.BytesIO):
                lpath.seek(0)
            async with meth(rpath, data=lpath.read(), **kw) as resp:  # type: ignore
                await self._async_raise_not_found_for_status(resp, rpath)
        else:
            async with session.post(rpath + "/operationType/upload") as resp:
                await self._async_raise_not_found_for_status(resp, rpath)
                operation_id = await resp.json()

            operation_id = operation_id["operationId"]
            resps = [resp async for resp in put_data()]
            kw = self.kwargs.copy()
            kw.update({"headers": headers})
            async with session.post(
                url=rpath + f"/operations/upload?operationId={operation_id}",
                json={"parts": resps},
                **kw,
            ) as resp:
                self._raise_not_found_for_status(resp, rpath + f"/operations/upload?operationId={operation_id}")

    @staticmethod
    def _construct_headers(
        file_local: Any, dt_from: str, dt_to: str, dt_created: str, chunk_size: int = 5 * 2**20, multipart: bool = False
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        headers = {
            "Content-Type": "application/octet-stream",
            "x-jpmc-distribution-created-date": dt_created,
            "x-jpmc-distribution-from-date": dt_from,
            "x-jpmc-distribution-to-date": dt_to,
            "Digest": "",  # to be changed to x-jpmc-digest
        }
        headers["Content-Type"] = "application/json" if multipart else headers["Content-Type"]
        headers_chunks = {"Content-Type": "application/octet-stream", "Digest": ""}

        headers_chunk_lst = []
        hash_sha256 = hashlib.sha256()
        if isinstance(file_local, io.BytesIO):
            file_local.seek(0)
        for chunk in iter(lambda: file_local.read(chunk_size), b""):
            hash_sha256_chunk = hashlib.sha256()
            hash_sha256_chunk.update(chunk)
            hash_sha256.update(hash_sha256_chunk.digest())
            headers_chunks = deepcopy(headers_chunks)
            headers_chunks["Digest"] = "SHA-256=" + base64.b64encode(hash_sha256_chunk.digest()).decode()
            headers_chunk_lst.append(headers_chunks)

        file_local.seek(0)
        if multipart:
            headers["Digest"] = "SHA-256=" + base64.b64encode(hash_sha256.digest()).decode()
        else:
            headers["Digest"] = "SHA-256=" + base64.b64encode(hash_sha256_chunk.digest()).decode()

        return headers, headers_chunk_lst

    def _cloud_copy(  # noqa: PLR0913, PLR0915
        self,
        lpath: Any,
        rpath: Any,
        dt_from: str,
        dt_to: str,
        dt_created: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        method: str = "put",
    ) -> None:
        async def _get_operation_id() -> dict[str, Any]:
            session = await self.set_session()
            async with session.post(rpath + "/operationType/upload", **self.kwargs) as r:
                await self._async_raise_not_found_for_status(r, rpath + "/operationType/upload")
                res: dict[str, Any] = await r.json()
                return res

        async def _finish_operation(operation_id: Any, kw: Any) -> None:
            session = await self.set_session()
            async with session.post(
                url=rpath + f"/operations/upload?operationId={operation_id}",
                json={"parts": resps},
                **kw,
            ) as r:
                await self._async_raise_not_found_for_status(
                    r, rpath + f"/operations/upload?operationId={operation_id}"
                )

        def put_data() -> Generator[dict[str, Any], None, None]:
            async def _meth(url: Any, kw: Any) -> None:
                session = await self.set_session()
                meth = getattr(session, method)
                retry_num = 3
                ex_cnt = 0
                last_ex = None
                while ex_cnt < retry_num:
                    async with meth(url=url, data=chunk, **kw) as resp:
                        try:
                            await self._async_raise_not_found_for_status(resp, url)
                            return await resp.json()  # type: ignore
                        except Exception as ex:  # noqa: BLE001
                            ex_cnt += 1
                            last_ex = ex

                raise Exception(f"Failed to upload file: {last_ex}, failed after {ex_cnt} exceptions.")

            context = nullcontext(lpath)

            with context as f:
                callback.get_size(getattr(f, "size", None))

            chunk = f.read(chunk_size)
            i = 0
            headers_chunks = {"Content-Type": "application/octet-stream", "Digest": ""}
            while chunk:
                hash_sha256_chunk = hashlib.sha256()
                hash_sha256_chunk.update(chunk)
                hash_sha256_lst[0].update(hash_sha256_chunk.digest())
                headers_chunks = deepcopy(headers_chunks)
                headers_chunks["Digest"] = "SHA-256=" + base64.b64encode(hash_sha256_chunk.digest()).decode()
                kw = self.kwargs.copy()
                kw.update({"headers": headers_chunks})
                url = rpath + f"/operations/upload?operationId={operation_id}&partNumber={i+1}"
                yield sync(self.loop, _meth, url, kw)
                i += 1
                callback.relative_update(len(chunk))
                chunk = f.read(chunk_size)

        method = method.lower()
        if method not in ("put", "post"):
            raise ValueError(f"method has to be either 'post' or 'put', not {method!r}")
        hash_sha256 = hashlib.sha256()
        hash_sha256_lst = [hash_sha256]
        headers = {
            "Content-Type": "application/json",
            "x-jpmc-distribution-created-date": dt_created,
            "x-jpmc-distribution-from-date": dt_from,
            "x-jpmc-distribution-to-date": dt_to,
            "Digest": "",  # to be changed to x-jpmc-digest
        }
        lpath.seek(0)
        kw = self.kwargs.copy()
        kw.update({"headers": headers})
        operation_id = sync(self.loop, _get_operation_id)["operationId"]
        resps = list(put_data())

        hash_sha256 = hash_sha256_lst[0]
        headers["Digest"] = "SHA-256=" + base64.b64encode(hash_sha256.digest()).decode()
        kw = self.kwargs.copy()
        kw.update({"headers": headers})
        sync(self.loop, _finish_operation, operation_id, kw)

    def put(  # noqa: PLR0913
        self,
        lpath: str,
        rpath: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        method: str = "put",
        multipart: bool = False,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Copy file(s) from local.

        Args:
            lpath: Lpath.
            rpath: Rpath.
            chunk_size: Chunk size.
            callback: Callback function.
            method: Method: put/post.
            multipart: Flag which indicated whether it's a multipart uplaod.
            from_date: earliest date of data in upload file
            to_date: latest date of data in upload file
            **kwargs: Kwargs.

        Returns:

        """

        if from_date is None or to_date is None:
            dt_from = pd.Timestamp.now().strftime("%Y-%m-%d")
            dt_to = "2199-12-31"
        else:
            dt_from = pd.Timestamp(from_date).strftime("%Y-%m-%d")
            dt_to = pd.Timestamp(to_date).strftime("%Y-%m-%d")

        dt_created = pd.Timestamp.now().strftime("%Y-%m-%d")
        rpath = self._decorate_url(rpath)
        if type(lpath).__name__ in ["S3File"]:
            return self._cloud_copy(lpath, rpath, dt_from, dt_to, dt_created, chunk_size, callback, method)
        headers, chunk_headers_lst = self._construct_headers(lpath, dt_from, dt_to, dt_created, chunk_size, multipart)
        kwargs.update({"headers": headers})
        if multipart:
            kwargs.update({"chunk_headers_lst": chunk_headers_lst})
            args = [lpath, rpath, chunk_size, callback, method, multipart]
        else:
            args = [lpath, rpath, None, callback, method, multipart]

        return sync(super().loop, self._put_file, *args, **kwargs)

    def find(self, path: str, maxdepth: Optional[int] = None, withdirs: bool = False, **kwargs: Any) -> Any:
        """Find all file in a folder.

        Args:
            path: Path.
            maxdepth: Max depth.
            withdirs: With dirs, default to False.
            **kwargs: Kwargs.

        Returns:

        """
        path = self._decorate_url(path)
        return super().find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs)

    def glob(self, path: str, **kwargs: Any) -> Any:
        """Glob.

        Args:
            path: Path.
            **kwargs: Kwargs.

        Returns:

        """

        return super().glob(path, **kwargs)

    def open(
        self,
        path: str,
        mode: str = "rb",
        **kwargs: Any,
    ) -> fsspec.spec.AbstractBufferedFile:
        """Open.

        Args:
            path: Path.
            mode: Defaults to rb.
            **kwargs: Kwargs.

        Returns:

        """

        path = self._decorate_url(path)
        return super().open(path, mode, **kwargs)

    def _open(  # noqa: PLR0913
        self,
        path: str,
        mode: str = "rb",
        block_size: Optional[int] = None,
        _autocommit: Optional[bool] = None,
        cache_type: None = None,
        cache_options: None = None,
        size: Optional[None] = None,
        **kwargs: Any,
    ) -> fsspec.spec.AbstractBufferedFile:
        """Make a file-like object.

        Args:
            path (str): Full URL with protocol
            mode (str): must be "rb"
            block_size (int): Bytes to download in one request; use instance value if None. If
            zero, will return a streaming Requests file-like instance.
            autocommit (bool):
            cache_type ():
            cache_options ():
            size ():
            **kwargs ():

        Returns:

        """
        if mode != "rb":
            raise NotImplementedError
        block_size = block_size if block_size is not None else self.block_size
        kw = self.kwargs.copy()
        kw["asynchronous"] = self.asynchronous
        kw.update(kwargs)
        size = size or self.info(path, **kwargs)["size"]
        session = sync(self.loop, self.set_session)
        if block_size and size:
            return FusionFile(
                self,
                path,
                session=session,
                block_size=block_size,
                mode=mode,
                size=size,
                cache_type=cache_type or self.cache_type,
                cache_options=cache_options or self.cache_options,
                loop=self.loop,
                **kw,
            )
        else:
            raise NotImplementedError


class FusionFile(HTTPFile):  # type: ignore
    """Fusion File."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init."""
        super().__init__(*args, **kwargs)

    async def async_fetch_range(self, start: int, end: int) -> bytes:
        """Download a block of data.

        The expectation is that the server returns only the requested bytes,
        with HTTP code 206. If this is not the case, we first check the headers,
        and then stream the output - if the data size is bigger than we
        requested, an exception is raised.
        """
        logger.debug(f"Fetch range for {self}: {start}-{end}")
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        url = self.url + f"/operationType/download?downloadRange=bytes={start}-{end-1}"
        logger.debug(str(url))
        r = await self.session.get(self.fs.encode_url(url), headers=headers, **kwargs)
        async with r:
            if r.status == requests.codes.range_not_satisfiable:  # noqa: PLR2004
                # range request outside file
                return b""
            r.raise_for_status()
            if r.status == requests.codes.partial_content:  # noqa: PLR2004
                # partial content, as expected
                out: bytes = await r.read()
            elif int(r.headers.get("Content-Length", 0)) <= end - start:
                out = await r.read()
            else:
                cl = 0
                out_arr = []
                while True:
                    chunk = await r.content.read(2**20)
                    # data size unknown, let's read until we have enough
                    if chunk:
                        out_arr.append(chunk)
                        cl += len(chunk)
                        if cl > end - start:
                            break
                    else:
                        break
                out = b"".join(out_arr)[: end - start]
            return out

    _fetch_range = sync_wrapper(async_fetch_range)
