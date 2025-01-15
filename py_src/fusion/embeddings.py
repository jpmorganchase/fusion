"""HTTP Connectivity to Fusion Embeddings API"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fusion._fusion import FusionCredentials
from fusion.utils import get_session

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from opensearchpy.compat import reraise_exceptions, string_types
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

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

HTTP_OK = 200
HTTP_MULTIPLE_CHOICES = 300

logger = logging.getLogger(__name__)


class FusionEmbeddingsConnection(Connection):  # type: ignore
    """
    Class responsible for maintaining HTTP connection to the Fusion Embedding API using OpenSearch. This class is a
    customized version of the `RequestsHttpConnection` class from the `opensearchpy` library, tailored to work with an
    internal vector database.

    The `FusionEmbeddingsConnection` class provides the following enhancements:
    - Establishes and manages HTTP connections to the Fusion Embedding API.
    - Integrates with the Fusion API for authentication and session management.
    - Provides methods for modifying and validating URLs specific to the Fusion Embedding API.

    :arg http_auth: optional http auth information as either ':' separated
        string or a tuple. Any value will be passed into requests as `auth`.
    :arg use_ssl: use ssl for the connection if `True`
    :arg verify_certs: whether to verify SSL certificates
    :arg ssl_show_warn: show warning when verify certs is disabled
    :arg ca_certs: optional path to CA bundle. Defaults to configured OpenSSL
        bundles from environment variables and then certifi before falling
        back to the standard requests bundle to improve consistency with
        other Connection implementations
    :arg client_cert: path to the file containing the private key and the
        certificate, or cert only if using client_key
    :arg client_key: path to the file containing the private key if using
        separate cert and key files (client_cert will contain only the cert)
    :arg headers: any custom http headers to be add to requests
    :arg http_compress: Use gzip compression
    :arg opaque_id: Send this value in the 'X-Opaque-Id' HTTP header
        For tracing all requests made by this transport.
    :arg pool_maxsize: Maximum connection pool size used by pool-manager
        For custom connection-pooling on current session
    :arg metrics: metrics is an instance of a subclass of the
        :class:`~opensearchpy.Metrics` class, used for collecting
        and reporting metrics related to the client's operations;
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
        if not REQUESTS_AVAILABLE:
            raise ImproperlyConfigured("Please install requests to use FusionEmbeddingsConnection.")

        # Initialize Session so .headers works before calling super().__init__().
        fusion_root_url: str = kwargs.get("root_url", "https://fusion.jpmorgan.com/api/v1/")
        credentials: FusionCredentials | str | None = kwargs.get("credentials", "config/client_credentials.json")
        if isinstance(credentials, FusionCredentials):
            self.credentials = credentials
        elif isinstance(credentials, str):
            self.credentials = FusionCredentials.from_file(Path(credentials))
        else:
            raise ValueError("credentials must be a path to a credentials file or FusionCredentials object")
        self.catalog = kwargs.get("catalog", "common")
        self.knowledge_base = kwargs.get("knowledge_base")

        self.session = get_session(self.credentials, fusion_root_url)
        self.base_url = fusion_root_url
        for key in list(self.session.headers):
            self.session.headers.pop(key)

        # Mount http-adapter with custom connection-pool size. Default=10
        if pool_maxsize and isinstance(pool_maxsize, int):
            pool_adapter = requests.adapters.HTTPAdapter(pool_maxsize=pool_maxsize)
            self.session.mount("http://", pool_adapter)
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

        if not self.http_compress:
            # Need to set this to 'None' otherwise Requests adds its own.
            self.session.headers["accept-encoding"] = None  # type: ignore

        if http_auth is not None:
            if isinstance(http_auth, (tuple, list)):
                http_auth = tuple(http_auth)
            elif isinstance(http_auth, string_types):
                if isinstance(http_auth, bytes):
                    http_auth = tuple(http_auth.decode("utf-8").split(":", 1))
                else:
                    http_auth = tuple(http_auth.split(":", 1))
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
        self.url_prefix = f"dataspaces/{self.catalog}/datasets/{self.knowledge_base}/indexes/"
        self.index_name: str | None = None

    def _tidy_url(self, url: str) -> str:
        return self.base_url[:-1] + url.replace("%2F%7B", "/").replace("%7D%2F", "/").replace("%2F", "/")

    @staticmethod
    def _remap_endpoints(url: str) -> str:
        return url.replace("_bulk", "embeddings").replace("_search", "search")

    @staticmethod
    def _modify_post_body_langchain(body: Any) -> Any:
        if body and "query" in body.decode("utf-8"):
            try:
                # Decode the bytes to a string
                json_str = body.decode("utf-8")

                # Parse the JSON string into a python dictionary
                data = json.loads(json_str)

                # Check if "query" and "knn" are in the data
                if "query" in data and "knn" in data["query"]:
                    # Extract the "knn" dictionary
                    knn_data = data["query"]["knn"]

                    # Create new structure
                    data["query"] = {"hybrid": {"queries": {"knn": knn_data}}}
                body = json.dumps(data, separators=(",", ":")).encode("utf-8")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.exception(f"An error occurred during modification of langchain POST body: {e}")
        return body

    @staticmethod
    def _modify_post_response_langchain(raw_data: Any) -> Any:
        if len(raw_data) > 0 and "hits" in json.loads(raw_data):
            try:
                data = json.loads(raw_data)
                for hit in data["hits"]:
                    # Change "source" to "_source" if it exists
                    if "source" in hit:
                        hit["_source"] = hit.pop("source")
                        hit["_id"] = hit.pop("id")
                        hit["_score"] = hit.pop("score")

                # Wrap the existing "hits" list in another dicitonary wit the key "hits"
                data["hits"] = {"hits": data["hits"]}

                # Serialize the modified dictionary back to a JSON string

                raw_data = json.dumps(data, separators=(",", ":"))

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.exception(f"An error occurred during modification of langchain POST response: {e}")

        return raw_data

    @staticmethod
    def _modify_post_haystack(body: Any, method: str) -> Any:
        if method.lower() == "post":
            body_str = body.decode("utf-8")
            try:
                json_strings = body_str.strip().split("\n")
                dict_list = [json.loads(json_string) for json_string in json_strings]
                for dct in dict_list:
                    if "embedding" in dct:
                        dct["vector"] = dct.pop("embedding")
                json_strings_mod = [json.dumps(d, separators=(",", ":")) for d in dict_list]
                joined_str = "\n".join(json_strings_mod)
                body = joined_str.encode("utf-8")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.exception(f"An error occurred during modification of haystack POST body: {e}")

            body_str = body.decode("utf-8")
            if "query" in body_str:
                json_dict = json.loads(body_str)
                if "bool" in json_dict["query"]:
                    json_dict["query"]["hybrid"] = {}
                    json_dict["query"]["hybrid"]["queries"] = json_dict["query"]["bool"].pop("must")
                    json_dict["query"].pop("bool")

                for i in json_dict["query"]["hybrid"]["queries"]:
                    if isinstance(i, dict) and "knn" in i and "embedding" in i["knn"]:
                        i["knn"]["vector"] = i["knn"].pop("embedding")

                body = json.dumps(json_dict, separators=(",", ":")).encode("utf-8")
        return body

    def _make_url_valid(self, url: str) -> str:
        if self.index_name is None:
            self.index_name = url.split("/")[-1]
            self.url_prefix = self.url_prefix + self.index_name

        if url.split("/")[-1] != self.index_name:
            url = self.base_url + self.url_prefix + "/" + url.split("/")[-1]
        else:
            url = self.base_url + self.url_prefix
        url = self._remap_endpoints(url)

        return url

    def perform_request(  # noqa: PLR0913
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
        if method.lower() == "put":
            method = "POST"

        url = self._tidy_url(url)
        url = self._make_url_valid(url)

        # _refresh endpoint not supported
        if "_refresh" in url:
            return 200, {}, ""

        headers = headers or {}

        orig_body = body
        if self.http_compress and body:
            body = self._gzip_compress(body)
            headers["content-encoding"] = "gzip"  # type: ignore

        body = FusionEmbeddingsConnection._modify_post_body_langchain(body)  # langchain specific
        body = FusionEmbeddingsConnection._modify_post_haystack(body, method)

        start = time.time()
        request = requests.Request(method=method, headers=headers, url=url, data=body)
        prepared_request = self.session.prepare_request(request)
        settings = self.session.merge_environment_settings(prepared_request.url, {}, None, None, None)
        send_kwargs: Any = {
            "timeout": timeout or self.timeout,
            "allow_redirects": allow_redirects,
        }
        send_kwargs.update(settings)
        try:
            self.metrics.request_start()
            response = self.session.send(prepared_request, **send_kwargs)
            duration = time.time() - start
            raw_data = response.content.decode("utf-8", "surrogatepass")
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

        # raise warnings if any from the 'Warnings' header.
        warnings_headers = (response.headers["warning"],) if "warning" in response.headers else ()
        self._raise_warnings(warnings_headers)

        raw_data = FusionEmbeddingsConnection._modify_post_response_langchain(raw_data)

        # raise errors based on http status codes, let the client handle those if needed
        if not (HTTP_OK <= response.status_code < HTTP_MULTIPLE_CHOICES) and response.status_code not in ignore:
            self.log_request_fail(
                method,
                url,
                response.request.path_url,
                orig_body,
                duration,
                response.status_code,
                raw_data,
            )
            self._raise_error(
                response.status_code,
                raw_data,
                response.headers.get("Content-Type"),
            )

        self.log_request_success(
            method,
            url,
            response.request.path_url,
            orig_body,
            response.status_code,
            raw_data,
            duration,
        )

        return response.status_code, response.headers, raw_data

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


def format_full_index_name(index_name: str, knowledge_base: str, catalog: str) -> str:
    """Generate index name for Embeddings API.

    Args:
        index_name (str): Name for index.
        knowledge_base (str): Knowledge base identifier.
        catalog (str): Catalog identifier.

    Returns:
        str: Full index name expected by embeddings API.
    """
    full_index_name = f"dataspaces/{catalog}/datasets/{knowledge_base}/indexes/{index_name}"
    return full_index_name


def format_index_body(number_of_shards: int = 2, dimension: int = 1536) -> dict[str, Any]:
    """Format index body for Embeddings API.

    Args:
        number_of_shards (int, optional): _description_. Defaults to 2.
        dimension (int, optional): _description_. Defaults to 1536.

    Returns:
        dict: Index body expected by embeddings API.
    """
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": number_of_shards,
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


def format_prompt_template(package: str, task: str = "RAG") -> str:
    if package.lower() == "langchain" and task == "RAG":
        template = """Given the following information, answer the question.
        
        {context}

        Question: {question}"""
    if package.lower() == "haystack" and task == "RAG":
        template = """
        Given the following information, answer the question.
        
        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """
    return template
