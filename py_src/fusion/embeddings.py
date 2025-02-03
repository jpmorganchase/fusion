"""HTTP Connectivity to Fusion Embeddings API"""

from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
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

from fusion._fusion import FusionCredentials
from fusion.utils import get_session

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

    Args:
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
        knowledge_base (str, optional): Knowledge base name. A dataset identifier.
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
    def _modify_post_response_langchain(raw_data: str | bytes | bytearray) -> str | bytes | bytearray:
        """Modify the response from langchain POST request to match the expected format.

        Args:
            raw_data (str | bytes | bytearray): Raw post response data.

        Returns:
            str | bytes | bytearray: Modified post repsonse data.
        """
        if len(raw_data) > 0:
            try:
                data = json.loads(raw_data)
                if "hits" in data:
                    for hit in data["hits"]:
                        # Change "source" to "_source" if it exists
                        if "source" in hit:
                            hit["_source"] = hit.pop("source")
                            hit["_id"] = hit.pop("id")
                            hit["_score"] = hit.pop("score")

                    # Wrap the existing "hits" list in another dicitonary wit the key "hits"
                    data["hits"] = {"hits": data["hits"]}

                    # Serialize the modified dictionary back to a JSON string

                    return json.dumps(data, separators=(",", ":"))

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.exception(f"An error occurred during modification of langchain POST response: {e}")

                return raw_data.decode("utf-8", errors="ignore") if isinstance(raw_data, bytes) else raw_data

        return raw_data

    @staticmethod
    def _modify_post_haystack(body: bytes | None, method: str) -> bytes | None:
        """Method to modify haystack POST body to match the embeddings API, which expects the embedding field to be
            named "vector".

        Args:
            body (bytes): Request body.
            method (str): Request method.

        Returns:
            bytes: Modified request body.
        """
        if method.lower() == "post":
            body_str = body.decode("utf-8") if body else ""

            if "query" in body_str:
                try:
                    query_dict = json.loads(body_str)
                    knn_list = query_dict.get("query", {}).get("bool", {}).get("must", {})
                    for knn in knn_list:
                        if "knn" in knn and "embedding" in knn["knn"]:
                            knn["knn"]["vector"] = knn["knn"].pop("embedding")
                    body = json.dumps(query_dict, separators=(",", ":")).encode("utf-8")
                except json.JSONDecodeError as e:
                    logger.exception(f"An error occurred during modification of haystack POST body: {e}")
            elif body_str != "":
                try:
                    json_strings = body_str.strip().split("\n")
                    dict_list = [json.loads(json_string) for json_string in json_strings]
                    for dct in dict_list:
                        if "embedding" in dct:
                            dct["vector"] = dct.pop("embedding")
                    json_strings_mod = [json.dumps(d, separators=(",", ":")) for d in dict_list]
                    joined_str = "\n".join(json_strings_mod)
                    body = joined_str.encode("utf-8")
                except json.JSONDecodeError as e:
                    logger.exception(f"An error occurred during modification of haystack POST body: {e}")
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

        body = FusionEmbeddingsConnection._modify_post_haystack(body, method)

        orig_body = body
        if self.http_compress and body:
            body = self._gzip_compress(body)
            headers["content-encoding"] = "gzip"  # type: ignore

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

        raw_data_modified = FusionEmbeddingsConnection._modify_post_response_langchain(raw_data)

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
