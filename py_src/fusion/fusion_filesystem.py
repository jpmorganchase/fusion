"""Fusion FileSystem."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
from collections.abc import AsyncGenerator, Generator
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast
from urllib.parse import quote, urljoin

import aiohttp
import fsspec
import fsspec.asyn
import pandas as pd
import requests
from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.implementations.http import HTTPFile, HTTPFileSystem, sync, sync_wrapper
from fsspec.utils import nullcontext

from fusion.credentials import FusionCredentials
from fusion.exceptions import APIResponseError

from .utils import _merge_responses, append_query_params, cpu_count, get_client, get_default_fs, get_session

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator, Mapping

logger = logging.getLogger(__name__)
VERBOSE_LVL = 25
DEFAULT_CHUNK_SIZE = 5 * 2**20
MULTIPART_CHECKSUM_PARTS = 2
APPLICATION_JSON_MIMETYPE = "application/json"
APPLICATION_OCTET_STREAM_MIMETYPE = "application/octet-stream"
UPLOAD_OPERATION_PATH = "/operationType/upload"
SHA256_DIGEST_PREFIX = "SHA-256="
HTTP_STATUS_EXCEPTIONS = (FileNotFoundError, requests.HTTPError, aiohttp.ClientResponseError)
ISDIR_FALSE_EXCEPTIONS = HTTP_STATUS_EXCEPTIONS + (aiohttp.ClientError, OSError, RuntimeError, ValueError)


def _build_sha256_digest(digest: bytes) -> str:
    return SHA256_DIGEST_PREFIX + base64.b64encode(digest).decode()


def _base64_crc(checksum: int, size: int) -> str:
    return base64.b64encode(checksum.to_bytes(size, byteorder="big")).decode("ascii")


def _base64_hash(data: bytes, algorithm: str) -> str:
    return base64.b64encode(hashlib.new(algorithm, data).digest()).decode("ascii")


class FusionHTTPFileSystem(HTTPFileSystem):  # type: ignore
    """Fusion HTTP filesystem."""

    def __init__(
        self,
        credentials: str | FusionCredentials | None = "config/client_credentials.json",
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
        self.sync_session = get_session(self.credentials, kwargs["client_kwargs"].get("root_url"))
        super().__init__(*args, **kwargs)

    def _extract_token_from_response(self, response: Any, token_header: str = "x-jpmc-next-token") -> Any:
        """Get pagination token from response headers if available"""
        if response and hasattr(response, "headers") and token_header in response.headers:
            token = response.headers[token_header]
            return token
        return None

    @staticmethod
    def _is_glue_delivery_channel(delivery_channel: str | list[str] | None = None) -> bool:
        channels: list[str] = []
        if isinstance(delivery_channel, str):
            channels.append(delivery_channel)
        elif isinstance(delivery_channel, list):
            channels.extend(str(channel) for channel in delivery_channel)
        return any(channel.strip().lower() == "glue" for channel in channels)

    def _should_skip_checksum_validation(
        self,
        headers: Mapping[str, Any],
        delivery_channel: str | list[str] | None = None,
    ) -> bool:
        expected_checksum = headers.get("x-jpmc-checksum")
        checksum_algorithm = headers.get("x-jpmc-checksum-algorithm")
        has_any_checksum_header = bool(expected_checksum or checksum_algorithm)
        return self._is_glue_delivery_channel(delivery_channel) and not has_any_checksum_header

    def _raise_not_found_for_status(self, response: Any, url: str) -> None:
        try:
            super()._raise_not_found_for_status(response, url)
        except HTTP_STATUS_EXCEPTIONS as ex:
            status_code = getattr(response, "status", None)
            message = f"Error when accessing {url}"
            raise APIResponseError(ex, message=message, status_code=status_code) from ex

    async def _async_raise_not_found_for_status(self, response: Any, url: str) -> None:
        """Raises FileNotFoundError for 404s, otherwise uses raise_for_status."""

        try:
            if response.status == requests.codes.not_found:  # noqa: PLR2004
                self._raise_not_found_for_status(response, url)
            else:
                real_reason = ""
                try:
                    real_reason = await response.text()
                    response.reason = real_reason
                finally:
                    self._raise_not_found_for_status(response, url)
        except HTTP_STATUS_EXCEPTIONS as ex:
            status_code = getattr(response, "status", None)
            message = f"Error when accessing {url}"
            raise APIResponseError(ex, message=message, status_code=status_code) from ex

    def _check_session_open(self) -> bool:
        # Check that _session is active. Expects that if _session is populated with .set_session, result
        # result was already awaited
        if self._session is None:
            return False
        return not self._session.closed

    async def _async_startup(self) -> None:
        await self.set_session()
        if not self._check_session_open():
            raise RuntimeError("FusionFS session closed before operation")

    async def _decorate_url_a(self, url: str) -> str:
        url = urljoin(f"{self.client_kwargs['root_url']}catalogs/", url) if "http" not in url else url
        url = url[:-1] if url[-1] == "/" else url
        return url

    def _decorate_url(self, url: str) -> str:
        url = urljoin(f"{self.client_kwargs['root_url']}catalogs/", url) if "http" not in url else url
        url = url[:-1] if url[-1] == "/" else url
        return url

    async def _isdir(self, path: str) -> bool:
        path = self._decorate_url(path)
        try:
            ret: dict[str, str] = await self._info(path)
            return ret["type"] == "directory"
        except ISDIR_FALSE_EXCEPTIONS:  # pragma: no cover
            logger.log(VERBOSE_LVL, "Artificial error", exc_info=True)
            return False

    async def _changes(self, url: str) -> dict[Any, Any]:
        """Get from given url.

        Currently called within the context of the /changes api endpoint.

        Args:
            url: str

        Returns:
            Dict containing json-ified return from endpoint.
        """
        url = self._decorate_url(url)
        all_responses = []
        next_token = None
        session = await self.set_session()

        call_kwargs = {k: v for k, v in self.kwargs.items() if k != "headers"}
        headers = self.kwargs.get("headers", {}).copy() if "headers" in self.kwargs else {}

        while True:
            if next_token:
                headers["x-jpmc-next-token"] = next_token
            call_kwargs["headers"] = headers
            async with session.get(url, **call_kwargs) as r:
                self._raise_not_found_for_status(r, url)
                try:
                    out: dict[Any, Any] = await r.json()
                except (aiohttp.ContentTypeError, json.JSONDecodeError, UnicodeDecodeError):
                    logger.log(VERBOSE_LVL, "%s cannot be parsed to json", url, exc_info=True)
                    out = {}
                all_responses.append(out)
                next_token = r.headers.get("x-jpmc-next-token")
                if not next_token:
                    break
        return _merge_responses(all_responses)

    async def _ls_real(self, url: str, detail: bool = False, **kwargs: Any) -> Any:
        clean_url = url
        url = self._normalize_ls_url(url)
        kw = self._build_ls_kwargs(kwargs)
        session = await self.set_session()
        if self._is_distribution_url(url):
            file_out, size = await self._ls_distribution(url, session)
            return self._format_ls_output(clean_url, file_out, detail, is_file=True, size=size)
        collection_out = await self._ls_collection(url, kw, session)
        return self._format_ls_output(clean_url, collection_out, detail, is_file=False)

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
            kwargs.pop("keep_protocol", None)
            res = super().info(path, **kwargs)
            if path.split("/")[-2] == "datasets":
                target = path.split("/")[-1]
                args = ["/".join(path.split("/")[:-1]) + f"/changes?datasets={quote(target)}"]
                res["changes"] = sync(super().loop, self._changes, *args)
        split_path = path.split("/")
        if len(split_path) > 1 and split_path[-2] == "distributions":
            res = res[0]

        return res

    async def _info(self, path: str, **kwargs: Any) -> Any:
        await self._async_startup()
        path = self._decorate_url(path)
        kwargs["keep_protocol"] = True
        res = await super()._ls(path, detail=True, **kwargs)
        if res[0]["type"] != "file":
            kwargs.pop("keep_protocol", None)
            res = await super()._info(path, **kwargs)
            if path.split("/")[-2] == "datasets":
                target = path.split("/")[-1]
                args = ["/".join(path.split("/")[:-1]) + f"/changes?datasets={quote(target)}"]
                res["changes"] = await self._changes(*args)
            if res["size"] is None and (res["mimetype"] == APPLICATION_JSON_MIMETYPE) and (res["type"] == "file"):
                res["size"] = 0
                res.pop("mimetype", None)
                res["type"] = "directory"
        split_path = path.split("/")
        if len(split_path) > 1 and split_path[-2] == "distributions":
            res = res[0]
        return res

    def ls(self, url: str, detail: bool = False, **kwargs: Any) -> Any:
        """List resources with pagination support.

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
                    k["name"] = k["name"].split(f"{self.client_kwargs['root_url']}catalogs/")[-1]
        elif not keep_protocol:
            return [x.split(f"{self.client_kwargs['root_url']}catalogs/")[-1] for x in ret]

        return ret

    async def _ls(self, url: str, detail: bool = False, **kwargs: Any) -> Any:
        await self._async_startup()
        url = await self._decorate_url_a(url)
        ret = await super()._ls(url, detail, **kwargs)
        keep_protocol = kwargs.pop("keep_protocol", False)
        if detail:
            if not keep_protocol:
                for k in ret:
                    k["name"] = k["name"].split(f"{self.client_kwargs['root_url']}catalogs/")[-1]
        elif not keep_protocol:
            return [x.split(f"{self.client_kwargs['root_url']}catalogs/")[-1] for x in ret]

        return ret

    def exists(self, url: str, **kwargs: Any) -> Any:
        """Check existence.

        Args:
            url: Url.
            detail: Detail.
            **kwargs: Kwargs.

        Returns:

        """
        url = self._decorate_url(url)
        return super().exists(url, **kwargs)

    async def _exists(self, url: str, **kwargs: Any) -> Any:
        await self._async_startup()
        url = self._decorate_url(url)
        out = await super()._exists(url, **kwargs)
        return out

    def isfile(self, path: str) -> bool | Any:
        """Is path a file.

        Args:
            path: Path.

        Returns:

        """
        path = self._decorate_url(path)
        return super().isfile(path)

    @staticmethod
    def _merge_all_data(all_data: dict[str, Any] | None, response_dict: dict[str, Any]) -> dict[str, Any]:
        if all_data is None:
            return response_dict
        if isinstance(response_dict, dict):
            return FusionHTTPFileSystem._merge_resource_list(all_data, response_dict.get("resources", []))
        if isinstance(response_dict, list):
            return FusionHTTPFileSystem._merge_resource_list(
                all_data,
                FusionHTTPFileSystem._flatten_resources(response_dict),
            )
        return FusionHTTPFileSystem._merge_resource_list(all_data, [response_dict])

    @staticmethod
    def _flatten_resources(response_list: list[Any]) -> list[Any]:
        flat_resources = []
        for item in response_list:
            if isinstance(item, dict) and "resources" in item:
                flat_resources.extend(item["resources"])
            else:
                flat_resources.append(item)
        return flat_resources

    @staticmethod
    def _merge_resource_list(all_data: dict[str, Any], resources: list[Any]) -> dict[str, Any]:
        if "resources" in all_data:
            all_data["resources"].extend(resources)
            return all_data
        return {"resources": resources}

    def cat(
        self,
        url: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fetch paths' contents with pagination support.

        Args:
            url: Url.
            start: Start.
            end: End.
            **kwargs: Kwargs.

        Returns:

        """
        url = self._decorate_url(url)
        kw = kwargs.copy()
        headers = kw.get("headers", {}).copy()
        kw["headers"] = headers

        session = self.sync_session
        range_start = start if start is not None else 0
        range_end = end if end is not None else 2**63 - 1

        fusion_file = FusionFile(self, url, session=session, **kw)
        all_bytes = bytearray()
        all_data = None

        while True:
            out, resp_headers = fusion_file._fetch_range_with_headers(range_start, range_end)
            try:
                # Try to decode as JSON (API response)
                response_dict = json.loads(out.decode("utf-8"))
                all_data = self._merge_all_data(all_data, response_dict)
                is_json = True
            except json.JSONDecodeError:
                # Not JSON, treat as file content
                all_bytes += out
                is_json = False

            next_token = resp_headers.get("x-jpmc-next-token")
            if not next_token:
                break
            headers["x-jpmc-next-token"] = next_token
            kw["headers"] = headers

        if is_json:
            return json.dumps(all_data, separators=(",", ":")).encode("utf-8")
        else:
            return bytes(all_bytes)

    async def _cat(
        self,
        url: str,
        start: int | None = None,
        end: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Fetch paths' contents with pagination support (async).

        Args:
            url: Url.
            start: Start.
            end: End.
            **kwargs: Kwargs.

        Returns:

        """
        await self._async_startup()
        url = self._decorate_url(url)
        kw = kwargs.copy()
        headers = kw.get("headers", {}).copy()
        kw["headers"] = headers

        session = await self.set_session()
        range_start = start if start is not None else 0
        range_end = end if end is not None else 2**63 - 1
        fusion_file = FusionFile(self, url, session=session, **kw)
        all_bytes = bytearray()
        all_data = None

        while True:
            out, resp_headers = await fusion_file._async_fetch_range_with_headers(range_start, range_end)
            try:
                # Try to decode as JSON (API response)
                response_dict = json.loads(out.decode("utf-8"))
                all_data = self._merge_all_data(all_data, response_dict)
                is_json = True
            except json.JSONDecodeError:
                # Not JSON, treat as file content
                all_bytes += out
                is_json = False

            next_token = resp_headers.get("x-jpmc-next-token")
            if not next_token:
                break
            headers["x-jpmc-next-token"] = next_token
            kw["headers"] = headers

        if is_json:
            return json.dumps(all_data, separators=(",", ":")).encode("utf-8")
        else:
            return bytes(all_bytes)

    async def _stream_file(self, url: str, chunk_size: int = 100) -> AsyncGenerator[bytes, None]:
        """Return an async stream to file at the given url.

        Args:
            url (str): File url. Appends Fusion.root_url if http prefix not present.
            chunk_size (int, optional): Size for each chunk in async stream. Defaults to 100.

        Returns:
            AsyncGenerator[bytes, None]: Async generator object.

        Yields:
            Iterator[AsyncGenerator[bytes, None]]: Next set of bytes read from the file at given url.
        """
        await self._async_startup()
        url = self._decorate_url(url)
        f = await self.open_async(url, "rb")
        async with f:
            while True:
                chunk = await f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def _fetch_range(
        self,
        session: aiohttp.ClientSession,
        url: str,
        start: int,
        end: int,
        output_file: fsspec.spec.AbstractBufferedFile,
    ) -> Any:
        """Fetch a range of bytes from a URL and write it to a file.

        Args:
            url (str): URL to fetch.
            start (int): Start byte.
            end (int): End byte.
            output_file (fsspec.spec.AbstractBufferedFile): File to write to.

        Returns:
            None: None.
        """

        range_url = append_query_params(url, {"downloadRange": f"bytes={start}-{end - 1}"})

        async def fetch() -> None:
            async with session.get(range_url, **self.kwargs) as response:
                if response.status in [200, 206]:
                    chunk = await response.read()
                    output_file.seek(start)
                    output_file.write(chunk)
                    logger.log(
                        VERBOSE_LVL,
                        "Wrote %s - %s bytes to %s" % (start, end, output_file.path),  # noqa: UP031
                    )
                else:
                    response.raise_for_status()

        retries = 5
        for attempt in range(retries):
            try:
                await fetch()
                return
            except Exception as ex:  # noqa: BLE001, PERF203
                if attempt < retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.log(
                        VERBOSE_LVL,
                        f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...",  # disable: W1202, C0209
                        exc_info=True,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.log(
                        VERBOSE_LVL,
                        f"Failed to write to {output_file.path}.",
                        exc_info=True,
                    )
                    raise ex

    async def _download_single_file_async(
        self,
        url: str,
        output_file: fsspec.spec.AbstractBufferedFile,
        file_size: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        n_threads: int = 10,
    ) -> list[tuple[bool, str, str | None]]:
        """Download a single file using asynchronous range requests.

        Args:
            url (str): _description_
            output_file (fsspec.spec.AbstractBufferedFile): _description_
            results (list[tuple[bool, str, Optional[str]]]): _description_
            file_size (int): _description_
            chunk_size (int, optional): _description_. Defaults to DEFAULT_CHUNK_SIZE.
            n_threads (int, optional): _description_. Defaults to 10.

        Returns:
            list[tuple[bool, str, Optional[str]]]: Return array.
        """

        coros = []
        session = await self.set_session()
        for start in range(0, file_size, chunk_size):
            end = min(start + chunk_size, file_size)
            task = self._fetch_range(session, url, start, end, output_file)
            coros.append(task)

        # Execute the tasks concurrently
        on_error = "raise"
        out = await fsspec.asyn._run_coros_in_chunks(coros, batch_size=n_threads, nofiles=True, return_exceptions=True)
        output_file.close()
        if on_error == "raise":
            ex = next(filter(fsspec.asyn.is_exception, out), False)
            if ex:
                return False, output_file.path, str(ex)  # type: ignore

        return True, output_file.path, None  # type: ignore

    async def _download_single_file_async_with_checksum(
        self,
        url: str,
        output_path: str,
        file_size: int,
        expected_checksum: str,
        checksum_algorithm: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        n_threads: int = 10,
    ) -> tuple[bool, str, str | None]:
        """Download a single file using asynchronous range requests with checksum validation.

        Downloads data to memory buffer, validates checksum, then writes to disk only if valid.
        This ensures no corrupted files are ever written to disk.

        Args:
            url (str): URL to download from.
            output_path (str): The file path where data will be saved.
            file_size (int): Size of the file to download.
            expected_checksum (str): Expected checksum value from headers.
            checksum_algorithm (str): Checksum algorithm from headers.
            chunk_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE.
            n_threads (int, optional): Number of concurrent download threads. Defaults to 10.

        Returns:
            tuple: (success, path, error)
        """
        try:
            data_chunks: dict[int, bytes] = {}
            coros = []
            session = await self.set_session()

            for start in range(0, file_size, chunk_size):
                end = min(start + chunk_size, file_size)
                task = self._fetch_range_to_memory(session, url, start, end, data_chunks)
                coros.append(task)

            out = await fsspec.asyn._run_coros_in_chunks(
                coros, batch_size=n_threads, nofiles=True, return_exceptions=True
            )

            ex = next(filter(fsspec.asyn.is_exception, out), False)
            if ex:
                return False, output_path, str(ex)

            file_data = bytearray(file_size)
            for start_pos, chunk_data in data_chunks.items():
                end_pos = start_pos + len(chunk_data)
                file_data[start_pos:end_pos] = chunk_data

            is_multipart = False
            base_checksum = expected_checksum

            if "-" in expected_checksum:
                parts = expected_checksum.rsplit("-", 1)
                if len(parts) == MULTIPART_CHECKSUM_PARTS and parts[1].isdigit():
                    base_checksum = parts[0]
                    is_multipart = True

            computed_checksum = self._compute_checksum_from_data(
                bytes(file_data), checksum_algorithm, is_multipart=is_multipart
            )

            if not computed_checksum:
                error_msg = f"Could not compute checksum using algorithm {checksum_algorithm}"
                logger.warning(error_msg)
                return False, output_path, error_msg

            checksum_to_compare = base_checksum if is_multipart else expected_checksum
            if computed_checksum != checksum_to_compare:
                error_msg = (
                    "Checksum validation failed. File may be corrupted or incomplete. "
                    "Please retry the download and if still failing contact the dataset owner."
                )
                logger.warning(error_msg)
                return False, output_path, error_msg

            await asyncio.to_thread(Path(output_path).parent.mkdir, parents=True, exist_ok=True)

            def write_file_data() -> None:
                with Path(output_path).open("wb") as f:
                    f.write(file_data)

            await asyncio.to_thread(write_file_data)

            logger.log(VERBOSE_LVL, f"Multi-threaded checksum validation successful, wrote to {output_path}")
            return True, output_path, None

        except Exception as ex:  # noqa: BLE001
            return False, output_path, str(ex)

    async def _fetch_range_to_memory(
        self,
        session: aiohttp.ClientSession,
        url: str,
        start: int,
        end: int,
        data_chunks: dict[int, bytes],
    ) -> None:
        """Fetch a range of bytes from a URL and store it in memory.

        Args:
            session: The aiohttp session to use.
            url (str): URL to fetch.
            start (int): Start byte.
            end (int): End byte.
            data_chunks (dict): Dictionary to store chunks keyed by start position.

        Returns:
            None: None.
        """

        range_url = append_query_params(url, {"downloadRange": f"bytes={start}-{end - 1}"})

        async def fetch() -> None:
            async with session.get(range_url, **self.kwargs) as response:
                if response.status in [200, 206]:
                    chunk = await response.read()
                    data_chunks[start] = chunk
                    logger.log(
                        VERBOSE_LVL,
                        f"Fetched bytes {start}-{end} ({len(chunk)} bytes) to memory",
                    )
                else:
                    response.raise_for_status()

        retries = 5
        for attempt in range(retries):
            try:
                await fetch()
                return
            except Exception as ex:  # noqa: BLE001, PERF203
                if attempt < retries - 1:
                    wait_time = 2**attempt  # Exponential backoff
                    logger.log(
                        VERBOSE_LVL,
                        f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...",
                        exc_info=True,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.log(
                        VERBOSE_LVL,
                        f"Failed to fetch range {start}-{end}.",
                        exc_info=True,
                    )
                    raise ex

    def stream_single_file(
        self,
        url: str,
        output_path: str,
        lfs: fsspec.AbstractFileSystem,
        block_size: int = DEFAULT_CHUNK_SIZE,
        delivery_channel: str | list[str] | None = None,
    ) -> tuple[bool, str, str | None]:
        """Function to stream a single file from the API to a file on disk.

        Args:
            url (str): The URL to call.
            output_path (str): The file path where the data will be saved.
            lfs (fsspec.AbstractFileSystem): The filesystem to use for file operations.
            block_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE

        Returns:
            tuple: A tuple

        """
        session = self.sync_session
        get_file_kwargs = self.kwargs.copy()
        get_file_kwargs.pop("proxy", None)

        try:
            with session.head(url, **get_file_kwargs) as r:
                r.raise_for_status()

                expected_checksum = r.headers.get("x-jpmc-checksum")
                checksum_algorithm = r.headers.get("x-jpmc-checksum-algorithm")

                if expected_checksum and checksum_algorithm:
                    return self.stream_single_file_with_checksum_validation(
                        url, output_path, lfs, expected_checksum, checksum_algorithm, block_size
                    )
                if self._should_skip_checksum_validation(r.headers, delivery_channel=delivery_channel):
                    logger.log(
                        VERBOSE_LVL,
                        "Skipping checksum validation for Glue-delivered download "
                        "because checksum headers are absent: %s",
                        url,
                    )
                    return self._stream_single_file_without_checksum(url, output_path, lfs, block_size)
                raise ValueError("Checksum validation is required but missing checksum information.")
        except Exception as ex:  # noqa: BLE001
            return False, output_path, str(ex)

    def _stream_single_file_without_checksum(
        self,
        url: str,
        output_path: str,
        lfs: fsspec.AbstractFileSystem,
        block_size: int = DEFAULT_CHUNK_SIZE,
    ) -> tuple[bool, str, str | None]:
        """Stream a single file to disk without checksum validation."""
        session = self.sync_session
        get_file_kwargs = self.kwargs.copy()
        get_file_kwargs.pop("proxy", None)

        try:
            if not lfs.exists(Path(output_path).parent):
                lfs.mkdir(Path(output_path).parent, create_parents=True)

            with session.get(url, **get_file_kwargs) as response:
                response.raise_for_status()
                with lfs.open(output_path, "wb") as file_obj:
                    for chunk in response.iter_content(block_size):
                        if chunk:
                            file_obj.write(chunk)

            logger.log(VERBOSE_LVL, "Checksum validation skipped, wrote to %s", output_path)
            return True, output_path, None
        except Exception as ex:  # noqa: BLE001
            return False, output_path, str(ex)

    def stream_single_file_with_checksum_validation(  # noqa: PLR0915
        self,
        url: str,
        output_path: str,
        lfs: fsspec.AbstractFileSystem,
        expected_checksum: str,
        checksum_algorithm: str,
        block_size: int = DEFAULT_CHUNK_SIZE,
    ) -> tuple[bool, str, str | None]:
        """Function to stream a single file with in-memory checksum validation.

        Streams data to memory buffer, validates checksum, then writes to disk only if valid.
        This ensures no corrupted files are ever written to disk.

        Args:
            url (str): The URL to call.
            output_path (str): The file path where data will be saved.
            lfs (fsspec.AbstractFileSystem): The filesystem to use for file operations.
            expected_checksum (str): Expected checksum value from headers.
            checksum_algorithm (str): Checksum algorithm from headers.
            block_size (int, optional): The chunk size to download data. Defaults to DEFAULT_CHUNK_SIZE.

        Returns:
            tuple: (success, path, error)
        """

        retries = 5
        for attempt in range(retries):
            try:
                success, path, error = self._stream_and_validate_checksum(
                    url,
                    output_path,
                    lfs,
                    expected_checksum,
                    checksum_algorithm,
                    block_size,
                )
                if success or attempt == retries - 1:
                    return success, path, error
            except Exception as ex:  # noqa: BLE001, PERF203
                if attempt == retries - 1:
                    error_msg = f"Failed after {retries} attempts: {ex}"
                    logger.log(VERBOSE_LVL, error_msg, exc_info=True)
                    return False, output_path, error_msg
            self._sleep_before_retry(attempt)
        return False, output_path, "Unknown error occurred"

    def _compute_checksum_from_data(self, data: bytes, algorithm: str, is_multipart: bool = False) -> str:  # noqa: PLR0911, PLR0912
        """Compute checksum from raw data bytes in AWS S3 compatible format.

        Supports AWS S3 checksum algorithms (returns base64-encoded values):
        - CRC32: AWS S3 checksumCRC32 (base64-encoded 4-byte value)
        - CRC32C: AWS S3 checksumCRC32C (base64-encoded 4-byte value)
        - SHA-256: AWS S3 checksumSHA256 (base64-encoded 32-byte hash)
        - SHA-1: AWS S3 checksumSHA1 (base64-encoded 20-byte hash)
        - MD5: AWS S3 Content-MD5 (base64-encoded 16-byte hash)
        - CRC64NVME: Custom algorithm (base64-encoded 8-byte value)

        Args:
            data: Raw bytes to compute checksum for
            algorithm: Algorithm name from x-jpmc-checksum-algorithm header
            is_multipart: Whether to apply multipart checksum computation (double hash). Defaults to False.

        Returns:
            Base64-encoded checksum string
        """
        crc_algorithms = {
            "CRC32": ("crc32", 4),
            "CRC32C": ("crc32c", 4),
            "CRC64NVME": ("crc64nvme", 8),
        }
        if algorithm in crc_algorithms:
            checksum_name, size = crc_algorithms[algorithm]
            return self._compute_crc_checksum(data, checksum_name, size, is_multipart)
        if algorithm == "SHA-256":
            return self._compute_hash_checksum(data, "sha256", is_multipart)
        if algorithm == "SHA-1":
            return self._compute_hash_checksum(data, "sha1", is_multipart)  # NOSONAR(S4790) legacy API checksum
        if algorithm == "MD5":
            return self._compute_hash_checksum(data, "md5", is_multipart)  # NOSONAR(S4790) legacy API checksum
        raise ValueError(f"Unsupported checksum algorithm: {algorithm}")

    def download(  # noqa: PLR0913
        self,
        lfs: fsspec.AbstractFileSystem,
        rpath: str | Path,
        lpath: str | Path,
        chunk_size: int = 5 * 2**20,
        overwrite: bool = True,
        preserve_original_name: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Download file(s) from remote to local.

        Args:
            lfs (fsspec.AbstractFileSystem): Local filesystem.
            rpath (Union[str, Path]): Remote path.
            lpath (Union[str, Path]): Local path.
            chunk_size (int, optional): Chunk size. Defaults to 5 * 2**20.
            overwrite (bool, optional): True if previously downloaded files should be overwritten. Defaults to True.
            preserve_original_name (bool, optional): True if the original name should be preserved. Defaults to False.
            **kwargs (Any): Kwargs.

        Returns:
            Any: Return value.
        """
        if not overwrite and lfs.exists(lpath) and not preserve_original_name:
            return True, lpath, None

        async def get_headers() -> Any:
            session = await self.set_session()
            async with session.head(rpath, **self.kwargs) as r:
                r.raise_for_status()
                return r.headers

        try:
            headers = sync(self.loop, get_headers)
            if not overwrite and lfs.exists(lpath):
                return True, lpath, None
        except Exception as ex:  # noqa: BLE001
            headers = {}
            logger.info(f"Failed to get headers for {rpath}", exc_info=ex)

        rpath = self._decorate_url(rpath) if isinstance(rpath, str) else rpath

        is_local_fs = type(lfs).__name__ == "LocalFileSystem"

        result = self.get(
            str(rpath),
            lpath,
            chunk_size=chunk_size,
            headers=headers,
            is_local_fs=is_local_fs,
            lfs=lfs,
            **kwargs,
        )

        return result

    def get(  # noqa: PLR0912  # disable: W0221
        self,
        rpath: str | io.IOBase,
        lpath: str | io.IOBase | fsspec.spec.AbstractBufferedFile | Path,
        chunk_size: int = 5 * 2**20,
        **kwargs: Any,
    ) -> Any:
        """Copy file(s) to local.

        Args:
            rpath: Rpath. Download url.
            lpath: Lpath. Destination path or opened file handle.
            chunk_size: Chunk size.
            **kwargs: Kwargs.

        Returns:

        """
        lfs = kwargs.pop("lfs", None)
        delivery_channel = kwargs.pop("delivery_channel", None)
        lfs, lpath_str = self._resolve_local_target(lpath, lfs)
        rpath = self._decorate_url(rpath) if isinstance(rpath, str) else rpath
        n_threads = cpu_count(is_threading=True)
        if not self._can_parallel_download(n_threads, kwargs.get("is_local_fs", False)):
            return self.stream_single_file(
                str(rpath),
                lpath_str,
                lfs,
                block_size=chunk_size,
                delivery_channel=delivery_channel,
            )
        try:
            return self._download_multi_threaded(
                str(rpath),
                lpath_str,
                lfs,
                chunk_size,
                n_threads,
                delivery_channel=delivery_channel,
            )
        except Exception as ex:  # noqa: BLE001
            logger.info(f"Failed to get content length for multi-threaded download: {ex}")
            return self.stream_single_file(
                str(rpath),
                lpath_str,
                lfs,
                block_size=chunk_size,
                delivery_channel=delivery_channel,
            )

    def _normalize_ls_url(self, url: str) -> str:
        if "http" not in url:
            url = f"{self.client_kwargs['root_url']}catalogs/" + url
        return url[:-1] if url.endswith("/") else url

    def _build_ls_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kw = cast(dict[str, Any], self.kwargs.copy())
        kw.update(kwargs)
        kw.pop("keep_protocol", None)
        headers = cast(dict[str, str], kw.get("headers", {})).copy()
        kw["headers"] = headers
        return kw

    @staticmethod
    def _is_distribution_url(url: str) -> bool:
        url_parts = url.split("/")
        return len(url_parts) > 1 and url_parts[-2] == "distributions"

    async def _ls_distribution(
        self,
        url: str,
        session: aiohttp.ClientSession,
    ) -> tuple[list[str], int]:
        async with session.head(url + "/operationType/download", **self.kwargs) as response:
            self._raise_not_found_for_status(response, url)
            parts = url.split("/")
            name = parts[6] + "-" + parts[8] + "-" + parts[10] + "." + parts[-1]
            return [name], int(response.headers["Content-Length"])

    async def _ls_collection(
        self,
        url: str,
        kw: dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> dict[str, Any]:
        async with session.get(url, **kw) as response:
            self._raise_not_found_for_status(response, url)
            out = cast(dict[str, Any], await response.json())
            resources = out.get("resources", [])
            next_token = cast(Optional[str], self._extract_token_from_response(response))
        if next_token:
            out["resources"] = await self._fetch_paginated_resources(url, kw, session, resources, next_token)
        return out

    async def _fetch_paginated_resources(
        self,
        url: str,
        kw: dict[str, Any],
        session: aiohttp.ClientSession,
        resources: list[Any],
        next_token: str,
    ) -> list[Any]:
        all_resources = list(resources)
        current_token = next_token
        while current_token:
            headers = kw.get("headers", {}).copy()
            headers["x-jpmc-next-token"] = current_token
            kw["headers"] = headers
            async with session.get(url, **kw) as response:
                self._last_async_response = response
                self._raise_not_found_for_status(response, url)
                more_out = await response.json()
                all_resources.extend(more_out.get("resources", []))
                current_token = self._extract_token_from_response(response)
        return all_resources

    def _format_ls_output(
        self,
        clean_url: str,
        out: list[str] | dict[str, Any],
        detail: bool,
        *,
        is_file: bool,
        size: int | None = None,
    ) -> Any:
        if not is_file:
            collection = cast(dict[str, Any], out)
            resources = [urljoin(clean_url + "/", item["identifier"]) for item in collection["resources"]]
            if not detail:
                return resources
            return [
                {
                    "name": resource,
                    "size": None,
                    "type": ("directory" if not resource.endswith(("csv", "parquet")) else "file"),
                }
                for resource in resources
            ]
        file_list = cast(list[str], out)
        if not detail:
            return file_list
        return [{"name": file_list[0], "size": size, "type": "file"}]

    def _stream_and_validate_checksum(
        self,
        url: str,
        output_path: str,
        lfs: fsspec.AbstractFileSystem,
        expected_checksum: str,
        checksum_algorithm: str,
        block_size: int,
    ) -> tuple[bool, str, str | None]:
        try:
            file_data = self._download_file_data(url, block_size)
            validation_error = self._validate_checksum(file_data, expected_checksum, checksum_algorithm)
            if validation_error is not None:
                logger.warning(validation_error)
                return False, output_path, validation_error
            self._write_downloaded_file(lfs, output_path, file_data)
            logger.log(VERBOSE_LVL, "Checksum validation successful, wrote to %s", output_path)
            return True, output_path, None
        except Exception as ex:  # noqa: BLE001
            return False, output_path, str(ex)

    def _download_file_data(self, url: str, block_size: int) -> bytes:
        data_buffer = io.BytesIO()
        session = self.sync_session
        get_file_kwargs = self.kwargs.copy()
        get_file_kwargs.pop("proxy", None)
        with session.get(url, **get_file_kwargs) as response:
            response.raise_for_status()
            byte_cnt = 0
            for chunk in response.iter_content(block_size):
                if chunk:
                    byte_cnt += len(chunk)
                    data_buffer.write(chunk)
        logger.log(VERBOSE_LVL, "Wrote %d bytes to %s", byte_cnt, url)
        return data_buffer.getvalue()

    def _validate_checksum(self, file_data: bytes, expected_checksum: str, checksum_algorithm: str) -> str | None:
        checksum_to_compare, is_multipart = self._parse_checksum_expectation(expected_checksum)
        computed_checksum = self._compute_checksum_from_data(file_data, checksum_algorithm, is_multipart=is_multipart)
        if not computed_checksum:
            return f"Could not compute checksum using algorithm {checksum_algorithm}"
        if computed_checksum != checksum_to_compare:
            return (
                "Checksum validation failed. File may be corrupted or incomplete. "
                "Please retry the download and if still failing contact the dataset owner."
            )
        return None

    @staticmethod
    def _parse_checksum_expectation(expected_checksum: str) -> tuple[str, bool]:
        if "-" not in expected_checksum:
            return expected_checksum, False
        parts = expected_checksum.rsplit("-", 1)
        if len(parts) == MULTIPART_CHECKSUM_PARTS and parts[1].isdigit():
            return parts[0], True
        return expected_checksum, False

    @staticmethod
    def _write_downloaded_file(lfs: fsspec.AbstractFileSystem, output_path: str, file_data: bytes) -> None:
        parent = Path(output_path).parent
        if not lfs.exists(parent):
            lfs.mkdir(parent, create_parents=True)
        with lfs.open(output_path, "wb") as file_obj:
            file_obj.write(file_data)

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        wait_time = 2**attempt
        logger.log(
            VERBOSE_LVL,
            "Attempt %d failed, retrying in %d seconds...",
            attempt + 1,
            wait_time,
            exc_info=True,
        )
        time.sleep(wait_time)

    def _compute_crc_checksum(self, data: bytes, checksum_name: str, size: int, is_multipart: bool) -> str:
        from awscrt import checksums as aws_checksums

        checksum_fn = getattr(aws_checksums, checksum_name)
        checksum = checksum_fn(data)
        checksum_bytes = checksum.to_bytes(size, byteorder="big")
        if is_multipart:
            return _base64_crc(checksum_fn(checksum_bytes), size)
        return base64.b64encode(checksum_bytes).decode("ascii")

    @staticmethod
    def _compute_hash_checksum(data: bytes, algorithm: str, is_multipart: bool) -> str:
        digest = hashlib.new(algorithm, data).digest()
        if is_multipart:
            digest = hashlib.new(algorithm, digest).digest()
        return base64.b64encode(digest).decode("ascii")

    @staticmethod
    def _can_parallel_download(n_threads: int, is_local_fs: bool) -> bool:
        return n_threads > 1 and is_local_fs

    @staticmethod
    def _resolve_local_target(
        lpath: str | io.IOBase | fsspec.spec.AbstractBufferedFile | Path,
        lfs: fsspec.AbstractFileSystem | None,
    ) -> tuple[fsspec.AbstractFileSystem, str]:
        if isinstance(lpath, (str, Path)):
            return lfs or get_default_fs(), str(lpath)
        if hasattr(lpath, "path"):
            return (lfs or get_default_fs()), lpath.path
        return get_default_fs(), str(lpath)

    def _download_multi_threaded(
        self,
        rpath: str,
        lpath_str: str,
        lfs: fsspec.AbstractFileSystem,
        chunk_size: int,
        n_threads: int,
        delivery_channel: str | list[str] | None = None,
    ) -> tuple[bool, str, str | None]:
        file_size, expected_checksum, checksum_algorithm = cast(
            tuple[Optional[int], Optional[str], Optional[str]],
            sync(self.loop, self._get_content_length_and_checksum, rpath),
        )
        if not file_size:
            return self.stream_single_file(
                rpath,
                lpath_str,
                lfs,
                block_size=chunk_size,
                delivery_channel=delivery_channel,
            )
        download_path = rpath if "operationType/download" in rpath else rpath + "/operationType/download"
        if not expected_checksum or not checksum_algorithm:
            headers = {
                "Content-Length": str(file_size),
                "x-jpmc-checksum": expected_checksum,
                "x-jpmc-checksum-algorithm": checksum_algorithm,
            }
            if self._should_skip_checksum_validation(headers, delivery_channel=delivery_channel):
                logger.log(
                    VERBOSE_LVL,
                    "Falling back to streamed download without checksum validation "
                    "for Glue-delivered download because checksum headers are absent: %s",
                    rpath,
                )
                return self.stream_single_file(
                    rpath,
                    lpath_str,
                    lfs,
                    block_size=chunk_size,
                    delivery_channel=delivery_channel,
                )
            return False, lpath_str, "Checksum validation is required but missing checksum information."
        download_result = cast(
            tuple[bool, str, Optional[str]],
            sync(
                self.loop,
                self._download_single_file_async_with_checksum,
                download_path,
                lpath_str,
                file_size,
                expected_checksum,
                checksum_algorithm,
                chunk_size,
                n_threads,
            ),
        )
        return download_result

    async def _get_content_length_and_checksum(
        self,
        rpath: str | io.IOBase,
    ) -> tuple[int | None, str | None, str | None]:
        session = await self.set_session()
        async with session.head(rpath, **self.kwargs) as response:
            response.raise_for_status()
            file_size = int(response.headers.get("Content-Length", 0)) or None
            expected_checksum = response.headers.get("x-jpmc-checksum")
            checksum_algorithm = response.headers.get("x-jpmc-checksum-algorithm")
            return file_size, expected_checksum, checksum_algorithm

    @staticmethod
    def _update_kwargs(
        kw: dict[str, Any], headers: dict[str, str], additional_headers: dict[str, str] | None
    ) -> dict[str, Any]:
        if "File-Name" in headers:  # noqa: PLR0915
            kw.setdefault("headers", {})
            kw["headers"]["File-Name"] = headers["File-Name"]
        if additional_headers:
            kw["headers"].update(additional_headers)
        return kw

    async def _put_file(  # noqa: PLR0915, PLR0913
        self,
        lpath: str | io.IOBase | fsspec.spec.AbstractBufferedFile,
        rpath: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        method: str = "post",
        multipart: bool = False,
        additional_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        async def put_data() -> AsyncGenerator[dict[Any, Any], None]:
            # Support passing arbitrary file-like objects
            # and use them instead of streams.
            if isinstance(lpath, io.IOBase):
                context = nullcontext(lpath)
                use_seek = False  # might not support seeking
            else:
                context = open(lpath, "rb")  # noqa: SIM115, PTH123, ASYNC101, ASYNC230
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
                    url = rpath + f"/operations/upload?operationId={operation_id}&partNumber={i + 1}"
                    kw.update({"headers": kwargs["chunk_headers_lst"][i]})
                    kw = FusionHTTPFileSystem._update_kwargs(kw, headers, additional_headers)
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
            if additional_headers:
                kw["headers"].update(additional_headers)
            if isinstance(lpath, io.BytesIO):
                lpath.seek(0)
            async with meth(rpath, data=lpath.read(), **kw) as resp:  # type: ignore
                await self._async_raise_not_found_for_status(resp, rpath)
        else:
            kw = self.kwargs.copy()
            kw = FusionHTTPFileSystem._update_kwargs(kw, headers, additional_headers)

            async with session.post(rpath + UPLOAD_OPERATION_PATH, **kw) as resp:
                await self._async_raise_not_found_for_status(resp, rpath)
                operation_id = await resp.json()

            operation_id = operation_id["operationId"]
            put_data_iterable: AsyncGenerator[dict[Any, Any], None] = put_data()
            resps = [resp async for resp in put_data_iterable]
            kw = self.kwargs.copy()
            kw.update({"headers": headers})
            kw = FusionHTTPFileSystem._update_kwargs(kw, headers, additional_headers)
            async with session.post(
                url=rpath + f"/operations/upload?operationId={operation_id}",
                json={"parts": resps},
                **kw,
            ) as resp:
                self._raise_not_found_for_status(resp, rpath + f"/operations/upload?operationId={operation_id}")

    @staticmethod
    def _construct_headers(
        file_local: Any,
        dt_from: str,
        dt_to: str,
        dt_created: str,
        chunk_size: int = 5 * 2**20,
        multipart: bool = False,
        file_name: str | None = None,
    ) -> tuple[dict[str, str], list[dict[str, str]]]:
        headers = {
            "Content-Type": APPLICATION_OCTET_STREAM_MIMETYPE,
            "x-jpmc-distribution-created-date": dt_created,
            "x-jpmc-distribution-from-date": dt_from,
            "x-jpmc-distribution-to-date": dt_to,
            "Digest": "",  # to be changed to x-jpmc-digest
        }
        if file_name:
            headers["File-Name"] = file_name

        headers["Content-Type"] = APPLICATION_JSON_MIMETYPE if multipart else headers["Content-Type"]
        headers_chunks = {"Content-Type": APPLICATION_OCTET_STREAM_MIMETYPE, "Digest": ""}

        headers_chunk_lst = []
        hash_sha256 = hashlib.sha256()
        if isinstance(file_local, io.BytesIO):
            file_local.seek(0)
        for chunk in iter(lambda: file_local.read(chunk_size), b""):
            hash_sha256_chunk = hashlib.sha256()
            hash_sha256_chunk.update(chunk)
            hash_sha256.update(hash_sha256_chunk.digest())
            headers_chunks = deepcopy(headers_chunks)
            headers_chunks["Digest"] = _build_sha256_digest(hash_sha256_chunk.digest())
            headers_chunk_lst.append(headers_chunks)

        file_local.seek(0)
        if multipart:
            headers["Digest"] = _build_sha256_digest(hash_sha256.digest())
        else:
            headers["Digest"] = _build_sha256_digest(hash_sha256_chunk.digest())

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
        file_name: str | None = None,
        additional_headers: dict[str, str] | None = None,
    ) -> None:
        async def _get_operation_id(kw: dict[str, str]) -> dict[str, Any]:
            session = await self.set_session()
            async with session.post(rpath + UPLOAD_OPERATION_PATH, **kw) as r:
                await self._async_raise_not_found_for_status(r, rpath + UPLOAD_OPERATION_PATH)
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
                            # wait 3 seconds before retrying
                            await asyncio.sleep(3 * (ex_cnt + 1))
                            logger.debug(f"Failed to upload file: {ex}")
                            ex_cnt += 1
                            last_ex = ex

                raise Exception(f"Failed to upload file: {last_ex}, failed after {ex_cnt} exceptions. {last_ex}")

            context = nullcontext(lpath)

            with context as f:
                callback.get_size(getattr(f, "size", None))

            chunk = f.read(chunk_size)
            i = 0
            headers_chunks = {"Content-Type": APPLICATION_OCTET_STREAM_MIMETYPE, "Digest": ""}
            while chunk:
                hash_sha256_chunk = hashlib.sha256()
                hash_sha256_chunk.update(chunk)
                hash_sha256_lst[0].update(hash_sha256_chunk.digest())
                headers_chunks = deepcopy(headers_chunks)
                headers_chunks["Digest"] = _build_sha256_digest(hash_sha256_chunk.digest())
                kw = self.kwargs.copy()
                kw.update({"headers": headers_chunks})
                kw = FusionHTTPFileSystem._update_kwargs(kw, headers, additional_headers)
                url = rpath + f"/operations/upload?operationId={operation_id}&partNumber={i + 1}"
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
            "Content-Type": APPLICATION_JSON_MIMETYPE,
            "x-jpmc-distribution-created-date": dt_created,
            "x-jpmc-distribution-from-date": dt_from,
            "x-jpmc-distribution-to-date": dt_to,
            "Digest": "",  # to be changed to x-jpmc-digest
        }
        if file_name:
            headers["File-Name"] = file_name

        if additional_headers:
            headers.update(additional_headers)

        lpath.seek(0)

        kw_op = self.kwargs.copy()
        kw_op = FusionHTTPFileSystem._update_kwargs(kw_op, headers, additional_headers)

        operation_id = sync(self.loop, _get_operation_id, kw_op)["operationId"]
        resps = list(put_data())
        hash_sha256 = hash_sha256_lst[0]
        headers["Digest"] = _build_sha256_digest(hash_sha256.digest())
        kw = self.kwargs.copy()
        kw.update({"headers": headers})
        kw = FusionHTTPFileSystem._update_kwargs(kw, headers, additional_headers)
        sync(self.loop, _finish_operation, operation_id, kw)

    def put(  # noqa: PLR0913
        self,
        lpath: str,
        rpath: str,
        chunk_size: int = 5 * 2**20,
        callback: fsspec.callbacks.Callback = _DEFAULT_CALLBACK,
        method: str = "put",
        multipart: bool = False,
        from_date: str | None = None,
        to_date: str | None = None,
        file_name: str | None = None,
        additional_headers: dict[str, str] | None = None,
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
            file_name: Name of the file.
            additional_headers: Additional headers.
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
            return self._cloud_copy(
                lpath, rpath, dt_from, dt_to, dt_created, chunk_size, callback, method, file_name, additional_headers
            )
        headers, chunk_headers_lst = self._construct_headers(
            lpath, dt_from, dt_to, dt_created, chunk_size, multipart, file_name
        )
        kwargs.update({"headers": headers})
        if multipart:
            kwargs.update({"chunk_headers_lst": chunk_headers_lst})
            args = [lpath, rpath, chunk_size, callback, method, multipart, additional_headers]
        else:
            args = [lpath, rpath, None, callback, method, multipart, additional_headers]

        return sync(super().loop, self._put_file, *args, **kwargs)

    def find(self, path: str, maxdepth: int | None = None, withdirs: bool = False, **kwargs: Any) -> Any:
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

    async def _find(self, path: str, maxdepth: int | None = None, withdirs: bool = False, **kwargs: Any) -> Any:
        await self._async_startup()
        path = self._decorate_url(path)
        out = await super()._find(path, maxdepth=maxdepth, withdirs=withdirs, **kwargs)
        return out

    def glob(self, path: str, **kwargs: Any) -> Any:
        """Glob.

        Args:
            path: Path.
            **kwargs: Kwargs.

        Returns:

        """

        return super().glob(path, **kwargs)

    async def _glob(self, path: str, **kwargs: Any) -> Any:
        out = await super()._glob(path, **kwargs)
        return out

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
        block_size: int | None = None,
        _autocommit: bool | None = None,
        cache_type: None = None,
        cache_options: None = None,
        size: int | None = None,
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
        url = append_query_params(f"{self.url}/operationType/download", {"downloadRange": f"bytes={start}-{end - 1}"})
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

    async def _async_fetch_range_with_headers(self, start: int, end: int) -> tuple[bytes, dict[str, Any]]:
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        r = await self.session.get(self.fs.encode_url(self.url), headers=headers, **kwargs)
        async with r:
            r.raise_for_status()
            out = await r.read()
            return out, r.headers

    def _fetch_range_with_headers(self, start: int, end: int) -> tuple[bytes, dict[str, Any]]:
        kwargs = self.kwargs.copy()
        headers = kwargs.pop("headers", {}).copy()
        headers["Range"] = f"bytes={start}-{end - 1}"
        with self.session.get(self.fs.encode_url(self.url), headers=headers, **kwargs) as r:
            r.raise_for_status()
            out = r.content
            return out, r.headers
