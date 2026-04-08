"""HTTP Connectivity to Fusion Embeddings API"""

from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import warnings
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiohttp
import requests
import yarl
from aiohttp import ClientTimeout
from aiohttp import client_exceptions as aiohttp_exceptions
from opensearchpy._async.compat import get_running_loop
from opensearchpy._async.http_aiohttp import AIOHttpConnection
from opensearchpy.compat import reraise_exceptions, string_types, urlencode
from opensearchpy.connection.base import Connection
from opensearchpy.exceptions import (
    ConnectionError as OpenSearchConnectionError,
)
from opensearchpy.exceptions import (
    ConnectionTimeout,
    ImproperlyConfigured,
    SSLError,
)
from opensearchpy.metrics import Metrics, MetricsNone

from fusion.credentials import FusionCredentials
from fusion.embeddings_utils import (
    _modify_post_haystack,
    _modify_post_response_langchain,
    _retrieve_index_name_from_bulk_body,
    is_index_creation_request,
    transform_index_creation_body,
)
from fusion.utils import create_secure_ssl_context, get_client, get_session

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from fusion.authentication import FusionAiohttpSession


HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300
VERIFY_CERTS_DEFAULT = object()
SSL_SHOW_WARN_DEFAULT = object()

logger = logging.getLogger(__name__)


def _load_credentials(credentials: FusionCredentials | str | None) -> FusionCredentials:
    if isinstance(credentials, FusionCredentials):
        return credentials
    if isinstance(credentials, str):
        return FusionCredentials.from_file(Path(credentials))
    raise ValueError("credentials must be a path to a credentials file or FusionCredentials object")


def _resolve_url_prefix(catalog: str, knowledge_base: str | list[str] | None) -> tuple[str, str]:
    url_prefix = f"dataspaces/{catalog}/datasets/{knowledge_base}/indexes/"
    multi_dataset_url_prefix = f"dataspaces/{catalog}/indexes/"
    if isinstance(knowledge_base, list):
        return multi_dataset_url_prefix, multi_dataset_url_prefix
    return url_prefix, multi_dataset_url_prefix


def _normalize_sync_http_auth(http_auth: Any) -> Any:
    if http_auth is None:
        return None
    if isinstance(http_auth, (tuple, list)):
        return tuple(http_auth)
    if isinstance(http_auth, string_types):
        if isinstance(http_auth, bytes):
            return tuple(http_auth.decode("utf-8").split(":", 1))
        return tuple(http_auth.split(":", 1))
    return http_auth


def _normalize_async_http_auth(http_auth: Any) -> Any:
    if http_auth is None:
        return None
    if isinstance(http_auth, (tuple, list)):
        return aiohttp.BasicAuth(login=http_auth[0], password=http_auth[1])
    if isinstance(http_auth, string_types):
        auth_string = http_auth.decode("utf-8") if isinstance(http_auth, bytes) else http_auth
        login, password = auth_string.split(":", 1)
        return aiohttp.BasicAuth(login=login, password=password)
    return http_auth


def _normalize_method(method: str) -> str:
    return "POST" if method.lower() == "put" else method


def _mock_info_response() -> tuple[int, dict[str, Any], str]:
    logger.warning("Returning mock OpenSearch info response for client compatibility")
    return 200, {}, json.dumps({"version": {"number": "2.11.0"}})


def _is_root_info_request(method: str, url: str, base_url: str) -> bool:
    return method.lower() == "get" and (url.endswith("/") or url == base_url.rstrip("/"))


def _skip_unsupported_endpoint(url: str) -> tuple[int, dict[str, Any], str] | None:
    if "_refresh" in url:
        return 200, {}, ""
    return None


def _skip_multi_kb_write(
    knowledge_base: str | list[str] | None,
    method: str,
    body: bytes | None,
) -> tuple[int, dict[str, Any], str] | None:
    if body and isinstance(knowledge_base, list) and method.lower() == "post" and "query" not in body.decode("utf-8"):
        return 200, {}, ""
    return None


def _prepare_embeddings_request_body(
    url: str,
    method: str,
    body: bytes | None,
    knowledge_base: str | list[str] | None,
) -> bytes | None:
    if body and is_index_creation_request(url, method, body):
        logger.debug("Transforming index creation body to match API expectations")
        body = transform_index_creation_body(body)
    return _modify_post_haystack(knowledge_base=knowledge_base, body=body, method=method)


def _get_short_circuit_response(
    base_url: str,
    knowledge_base: str | list[str] | None,
    method: str,
    url: str,
    body: bytes | None,
) -> tuple[int, dict[str, Any], str] | None:
    if _is_root_info_request(method, url, base_url):
        return _mock_info_response()
    return _skip_unsupported_endpoint(url) or _skip_multi_kb_write(knowledge_base, method, body)


def _resolve_async_ssl_defaults(verify_certs: Any, ssl_show_warn: Any) -> tuple[bool, bool]:
    resolved_verify_certs = True if verify_certs is VERIFY_CERTS_DEFAULT else verify_certs
    resolved_ssl_show_warn = True if ssl_show_warn is SSL_SHOW_WARN_DEFAULT else ssl_show_warn
    return resolved_verify_certs, resolved_ssl_show_warn


def _warn_if_ssl_context_overrides(
    ssl_context: Any,
    verify_certs: Any,
    ssl_show_warn: Any,
    ca_certs: Any,
    client_cert: Any,
    client_key: Any,
    ssl_version: Any,
) -> None:
    if ssl_context and (
        (verify_certs is not VERIFY_CERTS_DEFAULT)
        or (ssl_show_warn is not SSL_SHOW_WARN_DEFAULT)
        or ca_certs
        or client_cert
        or client_key
        or ssl_version
    ):
        warnings.warn("When using `ssl_context`, all other SSL related kwargs are ignored", stacklevel=2)


def _load_verify_locations(ssl_context: ssl.SSLContext, ca_certs: str) -> None:
    ca_path = Path(ca_certs)
    if ca_path.is_file():
        ssl_context.load_verify_locations(cafile=ca_certs)
        return
    if ca_path.is_dir():
        ssl_context.load_verify_locations(capath=ca_certs)
        return
    raise ImproperlyConfigured("ca_certs parameter is not a path")


def _validate_client_cert_path(path: Any, label: str) -> None:
    if path and not Path(path).is_file():
        raise ImproperlyConfigured(f"{label} is not a path to a file")


class FusionEmbeddingsConnection(Connection):  # type: ignore[misc]
    """
    Class responsible for maintaining HTTP connection to the Fusion Embedding API using OpenSearch. This class is a
    customized version of the `RequestsHttpConnection` class from the `opensearchpy` library, tailored to work with an
    internal vector database.

    The `FusionEmbeddingsConnection` class provides the following enhancements:
    - Establishes and manages HTTP connections to the Fusion Embedding API.
    - Integrates with the Fusion API for authentication and session management.
    - Provides methods for modifying and validating URLs specific to the Fusion Embedding API.

    Args:
        host (str, optional): Hostname of the server. Defaults to "localhost".
        port (int, optional): Port number of the server. Defaults to None.
        http_auth (str or tuple, optional): HTTP auth information as either ':' separated string or a tuple.
            Any value will be passed into requests as `auth`.
        use_ssl (bool, optional): Use SSL for the connection if `True`.
        verify_certs (bool, optional): Whether to verify SSL certificates.
        ssl_show_warn (bool, optional): Show warning when verify certs is disabled.
        ca_certs (str, optional): Path to CA bundle. Defaults to configured OpenSSL bundles from environment variables
            and then certifi before falling back to the standard requests bundle to improve consistency with other
            Connection implementations.
        client_cert (str, optional): Path to the file containing the private key and the certificate,
            or cert only if using client_key.
        client_key (str, optional): Path to the file containing the private key if using separate cert and key files
            (client_cert will contain only the cert).
        headers (dict, optional): Any custom HTTP headers to be added to requests.
        http_compress (bool, optional): Use gzip compression.
        opaque_id (str, optional): Send this value in the 'X-Opaque-Id' HTTP header for tracing
            all requests made by this transport.
        pool_maxsize (int, optional): Maximum connection pool size used by pool-manager
            for custom connection-pooling on current session.
        metrics (Metrics, optional): Instance of a subclass of the `opensearchpy.Metrics` class,
            used for collecting and reporting metrics related to the client's operations.

    Keyword Args:
        root_url (str, optional): Root URL for the Fusion API. Defaults to
            "https://fusion.jpmorgan.com/api/v1/".
        credentials (FusionCredentials or str, optional): Credentials for the Fusion API. Can be a `FusionCredentials`
            object or a path to a credentials file. Defaults to "config/client_credentials.json".
        catalog (str, optional): Catalog name. Defaults to "common".
        knowledge_base (str | list[str], optional): Knowledge base name. A dataset identifier. If multiple identifiers
            are provided, the connection will perform searches across all provided knowledge bases. Any other operations
            will not be supported, as a single knowledge base is required for those operations.
        index (str, optional): Index name. Defaults to `None`. Used to determine index when the _bulk operation
            is attempted. If not provided, it will be extracted from the request body.
    """

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        host: str = "localhost",
        port: int | None = None,
        http_auth: Any = None,
        use_ssl: bool = False,
        verify_certs: bool = True,
        ssl_show_warn: bool = True,
        ca_certs: Any = None,
        client_cert: Any = None,
        client_key: Any = None,
        headers: Any = None,
        http_compress: Any = None,
        opaque_id: Any = None,
        pool_maxsize: Any = None,
        metrics: Metrics = MetricsNone(),  # noqa: B008
        **kwargs: Any,
    ) -> None:
        self.metrics = metrics

        # Initialize Session so .headers works before calling super().__init__().
        fusion_root_url: str = kwargs.get("root_url", "https://fusion.jpmorgan.com/api/v1/")
        credentials: FusionCredentials | str | None = kwargs.get("credentials", "config/client_credentials.json")
        self.credentials = _load_credentials(credentials)
        self.catalog = kwargs.get("catalog", "common")
        self.knowledge_base = kwargs.get("knowledge_base")

        self.session = get_session(self.credentials, fusion_root_url)
        self.base_url = fusion_root_url
        for key in list(self.session.headers):
            self.session.headers.pop(key)

        # Mount https-adapter with custom connection-pool size. Default=10
        if pool_maxsize and isinstance(pool_maxsize, int):
            pool_adapter = requests.adapters.HTTPAdapter(pool_maxsize=pool_maxsize)
            self.session.mount("https://", pool_adapter)

        super().__init__(
            host=host,
            port=port,
            use_ssl=use_ssl,
            headers=headers,
            http_compress=http_compress,
            opaque_id=opaque_id,
            **kwargs,
        )

        self._configure_sync_connection(http_auth, verify_certs, ssl_show_warn, ca_certs, client_cert, client_key)
        self.url_prefix, self.multi_dataset_url_prefix = _resolve_url_prefix(self.catalog, self.knowledge_base)

        self.index_name: str | None = kwargs.get("index")

    def _configure_sync_connection(
        self,
        http_auth: Any,
        verify_certs: bool,
        ssl_show_warn: bool,
        ca_certs: Any,
        client_cert: Any,
        client_key: Any,
    ) -> None:
        if not self.http_compress:
            # Need to set this to 'None' otherwise Requests adds its own.
            self.session.headers["accept-encoding"] = None  # type: ignore

        http_auth = _normalize_sync_http_auth(http_auth)
        if http_auth is not None:
            self.session.auth = http_auth

        self.session.verify = verify_certs
        if not client_key:
            self.session.cert = client_cert
        elif client_cert:
            # cert is a tuple of (certfile, keyfile)
            self.session.cert = (client_cert, client_key)
        if ca_certs:
            if not verify_certs:
                raise ImproperlyConfigured("You cannot pass CA certificates when verify SSL is off.")
            self.session.verify = ca_certs
        elif verify_certs:
            ca_certs = self.default_ca_certs()
            if ca_certs:
                self.session.verify = ca_certs

        if not ssl_show_warn:
            requests.packages.urllib3.disable_warnings()  # type: ignore

        if self.use_ssl and not verify_certs and ssl_show_warn:
            warnings.warn(f"Connecting to {self.host} using SSL with verify_certs=False is insecure.", stacklevel=2)

    def _tidy_url(self, url: str) -> str:
        return url.replace("%2F%7B", "/").replace("%7D%2F", "/").replace("%2F", "/")

    @staticmethod
    def _remap_endpoints(url: str) -> str:
        return url.replace("_bulk", "embeddings").replace("_search", "search").replace("_doc", "embeddings")

    def _make_url_valid(self, url: str, body: bytes | None = None) -> str:
        if url == "/_bulk":
            index_name = self.index_name if self.index_name else _retrieve_index_name_from_bulk_body(body)
            url = self.base_url + self.url_prefix + index_name + url
        else:
            url = self.base_url + self.url_prefix + url.strip("/")

        url = self._remap_endpoints(url)

        return url

    def _send_prepared_request(
        self,
        method: str,
        url: str,
        prepared_request: requests.PreparedRequest,
        send_kwargs: dict[str, Any],
        orig_body: bytes | None,
        start: float,
    ) -> tuple[requests.Response, float, str]:
        try:
            self.metrics.request_start()
            response = self.session.send(prepared_request, **send_kwargs)
            duration = time.time() - start
            raw_data = response.content.decode("utf-8", "surrogatepass")
            return response, duration, raw_data
        except reraise_exceptions:
            raise
        except Exception as e:
            self.log_request_fail(
                method,
                url,
                prepared_request.path_url,
                orig_body,
                time.time() - start,
                exception=e,
            )
            if isinstance(e, requests.exceptions.SSLError):
                raise SSLError("N/A", str(e), e) from e
            if isinstance(e, requests.Timeout):
                raise ConnectionTimeout("TIMEOUT", str(e), e) from e
            raise OpenSearchConnectionError("N/A", str(e), e) from e
        finally:
            self.metrics.request_end()

    def perform_request(  # noqa: PLR0913, PLR0915
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,  # noqa: ARG002
        body: bytes | None = None,
        timeout: float | None = None,
        allow_redirects: bool | None = True,
        ignore: Collection[int] = (),
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        method = _normalize_method(method)

        url = self._tidy_url(url)
        url = self._make_url_valid(url, body)

        skipped_response = _get_short_circuit_response(self.base_url, self.knowledge_base, method, url, body)
        if skipped_response is not None:
            return skipped_response

        headers = dict(headers or {})
        body = _prepare_embeddings_request_body(url, method, body, self.knowledge_base)

        orig_body = body
        if self.http_compress and body:
            body = self._gzip_compress(body)
            headers["content-encoding"] = "gzip"

        start = time.time()
        request = requests.Request(method=method, headers=headers, url=url, data=body)
        prepared_request = self.session.prepare_request(request)
        settings = self.session.merge_environment_settings(prepared_request.url, {}, None, None, None)
        send_kwargs: Any = {
            "timeout": timeout or self.timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        response, duration, raw_data = self._send_prepared_request(
            method, url, prepared_request, send_kwargs, orig_body, start
        )

        # raise warnings if any from the 'Warnings' header.
        warnings_headers = (response.headers["warning"],) if "warning" in response.headers else ()
        self._raise_warnings(warnings_headers)

        raw_data_modified = cast(str, _modify_post_response_langchain(raw_data))

        # raise errors based on http status codes, let the client handle those if needed
        if not (HTTP_OK <= response.status_code < HTTP_MULTIPLE_CHOICES) and response.status_code not in ignore:
            self.log_request_fail(
                method,
                url,
                response.request.path_url,
                orig_body,
                duration,
                response.status_code,
                raw_data_modified,
            )
            self._raise_error(
                response.status_code,
                raw_data_modified,
                response.headers.get("Content-Type"),
            )

        self.log_request_success(
            method,
            url,
            response.request.path_url,
            orig_body,
            response.status_code,
            raw_data_modified,
            duration,
        )

        return response.status_code, response.headers, raw_data_modified

    @property
    def headers(self) -> Any:
        return self.session.headers

    @headers.setter
    def headers(self, value: Any) -> None:
        self.session.headers.update(value)

    def close(self) -> None:
        """
        Explicitly closes connections
        """
        self.session.close()


class FusionAsyncHttpConnection(AIOHttpConnection):  # type: ignore[misc]
    session: FusionAiohttpSession | None

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        host: str = "localhost",
        port: int | None = None,
        http_auth: Any = None,
        use_ssl: bool = False,
        verify_certs: Any = VERIFY_CERTS_DEFAULT,
        ssl_show_warn: Any = SSL_SHOW_WARN_DEFAULT,
        ca_certs: Any = None,
        client_cert: Any = None,
        client_key: Any = None,
        ssl_version: Any = None,
        ssl_assert_fingerprint: Any = None,
        maxsize: int | None = 10,
        headers: Mapping[str, str] | None = None,
        ssl_context: Any = None,
        http_compress: bool | None = None,
        opaque_id: str | None = None,
        loop: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Class responsible for maintaining asynchronous HTTP connection to the Fusion Embedding API using OpenSearch.
        This class is a customized version of the `AsyncHttpConnection` class from the `opensearchpy` library, tailored
        to work with an internal vector database.

        The `FusionAsyncHttpConnection` class provides the following enhancements:
        - Establishes and manages asynchronous HTTP connections to the Fusion Embedding API.
        - Integrates with the Fusion API for authentication and session management.
        - Provides methods for modifying and validating URLs specific to the Fusion Embedding API.


        Args:
            host (str, optional): Hostname of the server. Defaults to "localhost".
            port (int | None, optional): Port number of the server. Defaults to None.
            http_auth (Any, optional): HTTP auth information as either ':' separated string or a tuple.
            use_ssl (bool, optional): Use SSL for the connection if `True`.
            verify_certs (Any, optional): Whether to verify SSL certificates.
            ssl_show_warn (Any, optional): Show warning when verify certs is disabled.
            ca_certs (Any, optional): Path to CA bundle. See https://urllib3.readthedocs.io/en/latest/security.html#using-certifi-with-urllib3
                for instructions how to get default set
            client_cert (Any, optional): Path to the file containing the private key and the certificate,
            or cert only if using client_key.
            client_key (Any, optional): Path to the file containing the private key if using separate cert and key files
            (client_cert will contain only the cert).
            ssl_version (Any, optional): SSL version to use (e.g. ssl.PROTOCOL_TLSv1). Defaults to None.
            ssl_assert_fingerprint (Any, optional): Verify the supplied certificate fingerprint if not None.
            maxsize (int | None, optional): The number of connections which will be kept open to this
            host. See https://urllib3.readthedocs.io/en/1.4/pools.html#api for more
            information.
            headers (Mapping[str, str] | None, optional): Any custom HTTP headers to be added to requests. Defaults to
                None.
            ssl_context (Any, optional): SSL context to use for the connection. Defaults to None.
            http_compress (bool | None, optional): Use gzip compression. Defaults to None.
            opaque_id (str | None, optional): Send this value in the 'X-Opaque-Id' HTTP header for tracing
            all requests made by this transport.
            loop (Any, optional): asyncio Event Loop to use with aiohttp. This is set by default to the currently
                running loop.

        Keyword Args:
        root_url (str, optional): Root URL for the Fusion API. Defaults to
            "https://fusion.jpmorgan.com/api/v1/".
        credentials (FusionCredentials or str, optional): Credentials for the Fusion API. Can be a `FusionCredentials`
            object or a path to a credentials file. Defaults to "config/client_credentials.json".
        catalog (str, optional): Catalog name. Defaults to "common".
        knowledge_base (str, optional): Knowledge base name. A dataset identifier. If multiple identifiers
            are provided, the connection will perform searches across all provided knowledge bases. Any other operations
            will not be supported, as a single knowledge base is required for those operations.
        index (str, optional): Index name. Defaults to `None`. Used to determine index when the _bulk operation
            is attempted. If not provided, it will be extracted from the request body.
        """
        fusion_root_url: str = kwargs.get("root_url", "https://fusion.jpmorgan.com/api/v1/")
        credentials: FusionCredentials | str | None = kwargs.get("credentials", "config/client_credentials.json")
        self.credentials = _load_credentials(credentials)
        self.catalog = kwargs.get("catalog", "common")
        self.knowledge_base: str | list[str] | None = kwargs.get("knowledge_base")

        self.session = None
        self.base_url = fusion_root_url

        if kwargs.get("url_prefix"):
            kwargs.pop("url_prefix")

        self.url_prefix, self.multi_dataset_url_prefix = _resolve_url_prefix(self.catalog, self.knowledge_base)

        self.headers: dict[Any, Any] = {}

        self.index_name: str | None = kwargs.get("index")

        super().__init__(
            host=host,
            port=port,
            url_prefix=self.url_prefix,
            use_ssl=use_ssl,
            headers=headers,
            http_compress=http_compress,
            opaque_id=opaque_id,
            **kwargs,
        )

        http_auth = _normalize_async_http_auth(http_auth)

        _warn_if_ssl_context_overrides(
            ssl_context,
            verify_certs,
            ssl_show_warn,
            ca_certs,
            client_cert,
            client_key,
            ssl_version,
        )

        self.ssl_assert_fingerprint = ssl_assert_fingerprint
        ssl_context = self._build_async_ssl_context(
            ssl_context,
            verify_certs,
            ssl_show_warn,
            ca_certs,
            client_cert,
            client_key,
            ssl_version,
        )

        self.headers.setdefault("connection", "keep-alive")
        self.loop = loop
        self.session = None

        # Align with Sync Interface
        if "pool_maxsize" in kwargs:
            maxsize = kwargs.pop("pool_maxsize")

        # Parameters for creating an aiohttp.ClientSession later.
        self._limit = maxsize
        self._http_auth = http_auth
        self._ssl_context = ssl_context

    def _build_async_ssl_context(
        self,
        ssl_context: Any,
        verify_certs: Any,
        ssl_show_warn: Any,
        ca_certs: Any,
        client_cert: Any,
        client_key: Any,
        ssl_version: Any,
    ) -> Any:
        if not self.use_ssl or ssl_context is not None:
            return ssl_context

        ssl_context = (
            create_secure_ssl_context()
            if ssl_version is None
            else ssl.SSLContext(ssl_version)  # NOSONAR (S4423): preserve caller-supplied protocol
        )
        verify_certs, ssl_show_warn = _resolve_async_ssl_defaults(verify_certs, ssl_show_warn)
        self._configure_async_ssl_context(ssl_context, verify_certs, ssl_show_warn, ca_certs)
        self._load_async_client_cert_chain(ssl_context, client_cert, client_key)
        return ssl_context

    def _configure_async_ssl_context(
        self,
        ssl_context: ssl.SSLContext,
        verify_certs: bool,
        ssl_show_warn: bool,
        ca_certs: Any,
    ) -> None:
        if verify_certs:
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.check_hostname = True
        else:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        ca_certs = self.default_ca_certs() if ca_certs is None else ca_certs
        if verify_certs:
            if not ca_certs:
                raise ImproperlyConfigured(
                    "Root certificates are missing for certificate "
                    "validation. Either pass them in using the ca_certs parameter or "
                    "install certifi to use it automatically."
                )
            _load_verify_locations(ssl_context, ca_certs)
        elif ssl_show_warn:
            warnings.warn(f"Connecting to {self.host} using SSL with verify_certs=False is insecure.", stacklevel=2)

    @staticmethod
    def _load_async_client_cert_chain(
        ssl_context: ssl.SSLContext,
        client_cert: Any,
        client_key: Any,
    ) -> None:
        _validate_client_cert_path(client_cert, "client_cert")
        _validate_client_cert_path(client_key, "client_key")
        if client_cert and client_key:
            ssl_context.load_cert_chain(client_cert, client_key)
        elif client_cert:
            ssl_context.load_cert_chain(client_cert)

    def _prepare_async_request(
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None,
        body: bytes | None,
    ) -> tuple[str, str, bytes | None, bytes | None, str]:
        method = _normalize_method(method)
        url = self._tidy_url(url)
        url = self._make_url_valid(url, body)
        body = _prepare_embeddings_request_body(url, method, body, self.knowledge_base)
        orig_body = body
        query_string = urlencode(params) if params else ""
        if query_string:
            url = f"{url}?{query_string}"
        return method, url, body, orig_body, query_string

    def _build_async_request_headers(
        self,
        method: str,
        url: str,
        query_string: str,
        body: bytes | None,
        headers: Mapping[str, str] | None,
    ) -> tuple[dict[str, Any], bytes | None, aiohttp.BasicAuth | None]:
        req_headers = self.headers.copy()
        if headers:
            req_headers.update(headers)

        if self.http_compress and body:
            body = self._gzip_compress(body)
            req_headers["content-encoding"] = "gzip"

        auth = self._http_auth if isinstance(self._http_auth, aiohttp.BasicAuth) else None
        if callable(self._http_auth):
            req_headers = {
                **req_headers,
                **self._http_auth(method, url, query_string, body),
            }
        return req_headers, body, auth

    def _raise_async_request_exception(
        self,
        method: str,
        url: str,
        orig_body: bytes | None,
        start: float,
        exception: Exception,
    ) -> None:
        self.log_request_fail(
            method,
            str(url),
            url,
            orig_body,
            self.loop.time() - start,
            exception=exception,
        )
        if isinstance(exception, aiohttp_exceptions.ServerFingerprintMismatch):
            raise SSLError("N/A", str(exception), exception) from exception
        if isinstance(exception, (asyncio.TimeoutError, aiohttp_exceptions.ServerTimeoutError)):
            raise ConnectionTimeout("TIMEOUT", str(exception), exception) from exception
        raise ConnectionError("N/A", str(exception), exception) from exception

    def _handle_async_response(
        self,
        method: str,
        url: str,
        orig_body: bytes | None,
        response: Any,
        raw_data: str,
        duration: float,
        ignore: Collection[int],
    ) -> None:
        warning_headers = response.headers.getall("warning", ())
        self._raise_warnings(warning_headers)

        raw_data_modified = str(_modify_post_response_langchain(raw_data))
        if not (HTTP_OK <= response.status < HTTP_MULTIPLE_CHOICES) and response.status not in ignore:
            self.log_request_fail(
                method,
                str(url),
                url,
                orig_body,
                duration,
                status_code=response.status,
                response=raw_data_modified,
            )
            self._raise_error(response.status, raw_data_modified)

        self.log_request_success(method, str(url), url, orig_body, response.status, raw_data_modified, duration)

    def _tidy_url(self, url: str) -> str:
        return url.replace("%2F%7B", "/").replace("%7D%2F", "/").replace("%2F", "/")

    @staticmethod
    def _remap_endpoints(url: str) -> str:
        return url.replace("_bulk", "embeddings").replace("_search", "search").replace("_doc", "embeddings")

    def _make_url_valid(self, url: str, body: bytes | None = None) -> str:
        if url == "/_bulk":
            index_name = self.index_name if self.index_name else _retrieve_index_name_from_bulk_body(body)
            url = self.base_url.strip("/") + self.url_prefix + "/" + index_name + url
        else:
            url = self.base_url.strip("/") + self.url_prefix + url

        url = self._remap_endpoints(url)

        return url

    async def perform_request(  # noqa: PLR0912, PLR0915
        self,
        method: str,
        url: str,
        params: Mapping[str, Any] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        ignore: Collection[int] = (),
        headers: Mapping[str, str] | None = None,
    ) -> Any:
        if self.session is None:
            await self._create_aiohttp_session()
        assert self.session is not None

        method, url, body, orig_body, query_string = self._prepare_async_request(method, url, params, body)

        if _is_root_info_request(method, url, self.base_url):
            return _mock_info_response()
        skipped_response = _skip_unsupported_endpoint(url) or _skip_multi_kb_write(self.knowledge_base, method, body)
        if skipped_response is not None:
            return skipped_response

        request_timeout = aiohttp.ClientTimeout(total=timeout if timeout is not None else self.timeout)
        req_headers, body, auth = self._build_async_request_headers(method, url, query_string, body, headers)

        start = self.loop.time()
        timeout_obj: ClientTimeout | None = ClientTimeout(total=timeout) if isinstance(timeout, (int, float)) else None

        try:
            async with self.session.request(
                method,
                yarl.URL(url, encoded=True),
                data=body,
                auth=auth,
                headers=req_headers,
                timeout=timeout_obj or request_timeout,
            ) as response:
                raw_data = await response.text()
                duration = self.loop.time() - start

        # We want to reraise a cancellation or recursion error.
        except reraise_exceptions:
            raise
        except Exception as e:  # noqa: BLE001
            self._raise_async_request_exception(method, url, orig_body, start, e)

        self._handle_async_response(method, url, orig_body, response, raw_data, duration, ignore)
        return response.status, response.headers, raw_data

    async def close(self) -> Any:
        """
        Explicitly closes connection
        """
        if self.session:
            await self.session.close()
            self.session = None

    async def _create_aiohttp_session(self) -> Any:
        """Creates an aiohttp.ClientSession(). This is delayed until
        the first call to perform_request() so that AsyncTransport has
        a chance to set AIOHttpConnection.loop
        """
        if self.loop is None:
            self.loop = get_running_loop()
        self.session = await get_client(self.credentials)


def format_index_body(number_of_shards: int = 1, number_of_replicas: int = 1, dimension: int = 1536) -> dict[str, Any]:
    """Format index body for index creation in Embeddings API.

    Args:
        number_of_shards (int, optional): Number of primary shards to split the index into. This should be determined
            by the amount of data that will be stored in the index. Defaults to 1.
        number_of_replicas (int, optional): Number of replica shards to create for each primary shard. Defaults to 1.
        dimension (int, optional): Dimension of your index, determined by embedding model to be used for your index.
            Defaults to 1536.

    Returns:
        dict: Index body expected by embeddings API.
    """
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": number_of_shards,
                "number_of_replicas": number_of_replicas,
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "vector": {"type": "knn_vector", "dimension": dimension},
                "content": {"type": "text"},
                "chunk-id": {"type": "text"},
            }
        },
    }
    return index_body


class PromptTemplateManager:
    """Class to manage prompt templates for different packages and tasks."""

    def __init__(self) -> None:
        self.templates: dict[tuple[str, str], str] = {}

        self._load_default_templates()

    def _load_default_templates(self) -> None:
        """Load default prompt templates."""
        self.add_template(
            "langchain",
            "RAG",
            """Given the following information, answer the question.
        
        {context}

        Question: {question}""",
        )

        self.add_template(
            "haystack",
            "RAG",
            """
        Given the following information, answer the question.
        
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """,
        )

    def add_template(self, package: str, task: str, template: str) -> None:
        """Add a new template to the manager.

        Args:
            package (str): Package name.
            task (str): Task name.
            template (str): Template string.
        """
        self.templates[(package, task)] = template

    def get_template(self, package: str, task: str) -> str:
        """Get the template for the given package and task.

        Args:
            package (str): Package name.
            task (str): Task name.

        Returns:
            str: Template string.
        """
        return self.templates.get((package, task), "")

    def remove_template(self, package: str, task: str) -> None:
        """Remove the template for the given package and task.

        Args:
            package (str): Package name.
            task (str): Task name.
        """
        self.templates.pop((package, task), None)

    def list_tasks(self, package: str) -> list[str]:
        """List all tasks for the given package.

        Args:
            package (str): Package name.

        Returns:
            list[str]: List of tasks.
        """
        return [task for (pkg, task) in self.templates if pkg == package]

    def list_packages(self) -> list[str]:
        """List all packages.

        Returns:
            list[str]: List of packages.
        """
        return list({pkg for (pkg, task) in self.templates})
