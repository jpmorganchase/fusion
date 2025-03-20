"""Test for embeddings module."""

from __future__ import annotations

import asyncio
import ssl
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
import requests
from multidict import CIMultiDict
from opensearchpy import ImproperlyConfigured, RequestError
from opensearchpy._async._extra_imports import aiohttp_exceptions
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import (
    ConnectionTimeout,
    SSLError,
    TransportError,
)

from fusion._fusion import FusionCredentials
from fusion.embeddings import (
    FusionAsyncHttpConnection,
    FusionEmbeddingsConnection,
    PromptTemplateManager,
    format_index_body,
)


def test_format_index_body() -> None:
    """Test for format_index_body function."""

    formatted_body = format_index_body()
    assert formatted_body == {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "vector": {"type": "knn_vector", "dimension": 1536},
                "content": {"type": "text"},
                "chunk-id": {"type": "text"},
            }
        },
    }


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test for FusionEmbeddingsConnection class."""

    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    connection = FusionEmbeddingsConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        ca_certs="path/to/ca_certs",
        client_cert="path/to/client_cert",
        client_key="path/to/client_key",
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.session.verify == "path/to/ca_certs"
    assert connection.session.cert == ("path/to/client_cert", "path/to/client_key")
    assert connection.http_compress is True
    assert connection.session.auth == ("user", "pass")
    assert connection.url_prefix == "dataspaces/common/datasets/knowledge_base/indexes/"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == mock_credentials
    assert connection.session == mock_session


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_multi_kb(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test for FusionEmbeddingsConnection class."""

    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    connection = FusionEmbeddingsConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        ca_certs="path/to/ca_certs",
        client_cert="path/to/client_cert",
        client_key="path/to/client_key",
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base=["knowledge_base", "knowledge_base2"],
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.session.verify == "path/to/ca_certs"
    assert connection.session.cert == ("path/to/client_cert", "path/to/client_key")
    assert connection.http_compress is True
    assert connection.session.auth == ("user", "pass")
    assert connection.url_prefix == "dataspaces/common/indexes/"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == mock_credentials
    assert connection.session == mock_session


def test_fusion_embeddings_connection_wrong_creds() -> None:
    """Test for FusionEmbeddingsConnection class."""

    with pytest.raises(
        ValueError, match="credentials must be a path to a credentials file or FusionCredentials object"
    ):
        FusionEmbeddingsConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            ca_certs="path/to/ca_certs",
            client_cert="path/to/client_cert",
            client_key="path/to/client_key",
            headers={"custom-header": "value"},
            http_compress=True,
            opaque_id="opaque-id",
            pool_maxsize=20,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials={"my_id": "12345", "my_password": "password"},
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.get_session")
def test_fusion_embeddings_connection_creds_obj(mock_get_session: MagicMock, credentials: FusionCredentials) -> None:
    """Test for FusionEmbeddingsConnection class."""

    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    connection = FusionEmbeddingsConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        ca_certs="path/to/ca_certs",
        client_cert="path/to/client_cert",
        client_key="path/to/client_key",
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials=credentials,
        catalog="common",
        knowledge_base="knowledge_base",
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.session.verify == "path/to/ca_certs"
    assert connection.session.cert == ("path/to/client_cert", "path/to/client_key")
    assert connection.http_compress is True
    assert connection.session.auth == ("user", "pass")
    assert connection.url_prefix == "dataspaces/common/datasets/knowledge_base/indexes/"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == credentials
    assert connection.session == mock_session


def test_prompt_template_manager_initialization() -> None:
    manager = PromptTemplateManager()
    assert isinstance(manager.templates, dict)
    assert len(manager.templates) > 0, "Default templates should be loaded"


def test_add_and_retrieve_template() -> None:
    manager = PromptTemplateManager()
    manager.add_template("test_package", "test_task", "Test template content")

    retrieved = manager.get_template("test_package", "test_task")
    assert retrieved == "Test template content"


def test_retrieve_nonexistent_template() -> None:
    manager = PromptTemplateManager()
    template = manager.get_template("nonexistent_package", "nonexistent_task")
    assert template == ""


def test_remove_templates_single() -> None:
    manager = PromptTemplateManager()
    # Add two templates for the same package
    manager.add_template("my_package", "task_one", "Template 1")
    manager.add_template("my_package", "task_two", "Template 2")

    manager.remove_template("my_package", "task_one")
    assert manager.get_template("my_package", "task_one") == ""


def test_list_tasks() -> None:
    manager = PromptTemplateManager()
    manager.add_template("pkg_one", "task_a", "Template A")
    manager.add_template("pkg_one", "task_b", "Template B")
    manager.add_template("pkg_two", "task_c", "Template C")

    tasks = manager.list_tasks(package="pkg_one")

    assert "task_a" in tasks
    assert "task_b" in tasks


def test_list_packages() -> None:
    manager = PromptTemplateManager()
    manager.add_template("pkg_one", "task_a", "Template A")
    manager.add_template("pkg_one", "task_b", "Template B")
    manager.add_template("pkg_two", "task_c", "Template C")

    packages = manager.list_packages()
    assert "pkg_one" in packages
    assert "pkg_two" in packages


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_clears_session_headers_on_init(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    # Create a session with a dummy header
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_session.headers = {"X-Test": "test-value"}

    conn = FusionEmbeddingsConnection(
        host="localhost", root_url="https://example.com/api", credentials="some_credentials_file.json"
    )

    assert not conn.session.headers.get("X-Test"), "Session headers should be cleared on init"


@pytest.mark.parametrize(
    ("input_auth", "expected"),
    [
        (("user", "pass"), ("user", "pass")),
        (["user", "pass"], ("user", "pass")),
        (b"user:pass", ("user", "pass")),
        ("user:pass", ("user", "pass")),
    ],
)
@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_http_auth(
    mock_from_file: MagicMock,  # noqa: ARG001
    mock_get_session: MagicMock,  # noqa: ARG001
    input_auth: tuple[str, str] | list[str] | Literal[b"user:pass", "user:pass"],
    expected: tuple[str, str],
) -> None:
    conn = FusionEmbeddingsConnection(
        host="localhost",
        http_auth=input_auth,
    )
    # Verify that the `session.auth` was set to a tuple
    assert conn.session.auth == expected


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_wrong_inputs(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
) -> None:
    """Test for FusionEmbeddingsConnection class."""

    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    with pytest.raises(ImproperlyConfigured, match="You cannot pass CA certificates when verify SSL is off."):
        FusionEmbeddingsConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=False,
            ca_certs="path/to/ca_certs",
            client_cert="path/to/client_cert",
            client_key="path/to/client_key",
            headers={"custom-header": "value"},
            http_compress=True,
            opaque_id="opaque-id",
            pool_maxsize=20,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_ssl_warning(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
) -> None:
    """Test for FusionEmbeddingsConnection class."""

    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    with pytest.warns(
        UserWarning, match="Connecting to https://localhost:9200 using SSL with verify_certs=False is insecure."
    ):
        FusionEmbeddingsConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=True,
            client_cert="path/to/client_cert",
            client_key="path/to/client_key",
            headers={"custom-header": "value"},
            http_compress=True,
            opaque_id="opaque-id",
            pool_maxsize=20,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@pytest.mark.parametrize(
    ("base_url", "raw_url", "expected_url"),
    [
        (
            "https://example.com/api/v1/",
            "%2F%7Bmyindex%7D%2F_test%2Fmock%2F",
            "/myindex/_test/mock/",
        )
    ],
)
@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_tidy_url(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
    base_url: Literal["https://example.com/api/v1/"],
    raw_url: Literal["/%2F%7Bmyindex%7D%2F_test%2Fmock%2F"],
    expected_url: Literal["https://example.com/api/v1/myindex/_test/mock/"],
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(
        host="localhost",
        root_url=base_url,
        credentials="dummy_credentials.json",
    )

    tidied = conn._tidy_url(raw_url)
    assert tidied == expected_url


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_embeddings_connection_remap_url(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(
        host="localhost",
        credentials="dummy_credentials.json",
    )
    url = "https://example.com/api/v1/myindex/_test/mock/_bulk/_search"

    remapped = conn._remap_endpoints(url)
    expected = "https://example.com/api/v1/myindex/_test/mock/embeddings/search"

    assert remapped == expected


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_make_valid_url(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(
        host="localhost",
        credentials="dummy_credentials.json",
        root_url="https://example.com/api/v1/",
        index_name="myindex",
        catalog="mycatalog",
        knowledge_base="mykb",
    )
    url = "/myindex"
    exp_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex"

    modified_url = conn._make_url_valid(url)
    assert modified_url == exp_url


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_make_valid_url_bulk(
    mock_from_file: MagicMock,
    mock_get_session: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(
        host="localhost",
        credentials="dummy_credentials.json",
        root_url="https://example.com/api/v1/",
        index_name="myindex",
        catalog="mycatalog",
        knowledge_base="mykb",
        index="myindex",
    )
    url = "/_bulk"
    exp_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex/embeddings"

    modified_url = conn._make_url_valid(url)
    assert modified_url == exp_url


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_success(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test a normal 200 success response."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_response = MagicMock(status_code=200, content=b"OK-content")
    mock_session.send.return_value = mock_response
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    result = conn.perform_request("GET", url="http://example.com/test")

    # Verify we got the response back
    assert result
    # Ensure the session 'send' was called
    mock_session.send.assert_called_once()


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_ignore_status(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test ignoring certain status codes."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_response = MagicMock(status_code=404, content=b"Not Found")
    mock_session.send.return_value = mock_response
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    result = conn.perform_request("GET", url="http://example.com/test", ignore=[404])
    status_code = 404
    assert result[0] == status_code


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_refresh_endpoint(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that the '_refresh' endpoint raises OpenSearchConnectionError."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    result = conn.perform_request("PUT", url="http://example.com/_refresh")
    status_code = 200
    assert result[0] == status_code


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_compression(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that request body is gzipped and Content-Encoding header is set."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_response = MagicMock(status_code=200, content=b"ok-content", headers={"content-encoding": "gzip"})
    mock_session.send.return_value = mock_response
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json", http_compress=True)

    body_data = b'{"test": "data"}'

    result = conn.perform_request("POST", url="http://example.com/test", body=body_data)

    assert result[1].get("content-encoding") == "gzip"


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_multi_kb_pass(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that request body is gzipped and Content-Encoding header is set."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_session.send.return_value = mock_response
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json", knowledge_base=["kb1", "kb2"])

    body_data = b'{"test": "data"}'

    result = conn.perform_request("POST", url="http://example.com/test", body=body_data)

    status_code = 200
    assert result[0] == status_code


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_reraise_exceptions(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_session = MagicMock()
    # Trigger a KeyboardInterrupt to simulate a reraise_exceptions scenario
    mock_session.send.side_effect = KeyboardInterrupt("Testing raise")
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")

    # Verify that the exception is indeed re-raised
    with pytest.raises(KeyboardInterrupt, match="Testing raise"):
        conn.perform_request("GET", url="http://example.com/test")


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_reraise_exceptions_recursion(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """
    Test that a RecursionError is re-raised, covering
    'except reraise_exceptions: raise'.
    """
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_session = MagicMock()
    # Trigger a RecursionError to match what's in reraise_exceptions
    mock_session.send.side_effect = RecursionError("Test recursion error")
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")

    with pytest.raises(RecursionError, match="Test recursion error"):
        conn.perform_request("GET", url="http://example.com/test")


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_ssl_error(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that an SSLError in requests is re-raised as SSLError from opensearchpy."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_session = MagicMock()
    mock_session.send.side_effect = requests.exceptions.SSLError("SSL error occurred")
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(SSLError) as exc_info:
        conn.perform_request("GET", url="http://example.com")
    assert "SSL error occurred" in str(exc_info.value)


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_timeout_error(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that a requests.Timeout is re-raised as ConnectionTimeout from opensearchpy."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_session = MagicMock()
    mock_session.send.side_effect = requests.Timeout("Request timed out")
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(ConnectionTimeout) as exc_info:
        conn.perform_request("GET", url="http://example.com")
    assert "Request timed out" in str(exc_info.value)


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_perform_request_other_exception(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that any other exception is re-raised as OpenSearchConnectionError."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_session = MagicMock()
    mock_session.send.side_effect = RuntimeError("Something went wrong")
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(OpenSearchConnectionError) as exc_info:
        conn.perform_request("GET", url="http://example.com")
    assert "Something went wrong" in str(exc_info.value)


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
@patch("fusion.embeddings.FusionEmbeddingsConnection.log_request_fail")
def test_perform_request_failure_status(
    mock_log_fail: MagicMock, mock_from_file: MagicMock, mock_get_session: MagicMock
) -> None:
    """
    Test that a 400 response outside the 200 range and not in ignore
    triggers a failure log and raises an error.
    """
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_response = MagicMock(status_code=400, content=b"Error content")
    mock_response.request.path_url = "/test"
    mock_response.headers = {"Content-Type": "application/json"}
    mock_session.send.return_value = mock_response
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")

    # If your _raise_error method raises an exception (e.g. NotFoundError), catch it
    with pytest.raises(RequestError):
        conn.perform_request("GET", url="http://example.com", ignore=[404])  # 400 not ignored

    # Ensure log_request_fail was called
    mock_log_fail.assert_called_once()


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_headers_property(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that setting the headers updates the session headers."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_session.headers = {"new-header": "value"}
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    # Verify initial

    # Update headers
    conn.headers = {"new-header": "value"}
    assert conn.headers.get("new-header") == "value"


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_close(mock_from_file: MagicMock, mock_get_session: MagicMock) -> None:
    """Test that close() calls session.close()."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    conn = FusionEmbeddingsConnection(host="localhost", credentials="dummy.json")
    conn.close()
    mock_session.close.assert_called_once()


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection(mock_from_file: MagicMock) -> None:
    """Test for FusionAsyncHttpConnection class."""

    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    connection = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.http_compress is True
    assert connection.url_prefix == "/dataspaces/common/datasets/knowledge_base/indexes"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == mock_credentials
    assert connection.session is None


def test_async_fusion_embeddings_connection_wrong_creds() -> None:
    """Test for FusionEmbeddingsConnection class."""

    with pytest.raises(
        ValueError, match="credentials must be a path to a credentials file or FusionCredentials object"
    ):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            headers={"custom-header": "value"},
            http_compress=True,
            opaque_id="opaque-id",
            pool_maxsize=20,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials={"username": "user", "password": "pass"},
            catalog="common",
            knowledge_base="knowledge_base",
        )


def test_async_fusion_embeddings_connection_creds_obj(credentials: FusionCredentials) -> None:
    """Test for FusionEmbeddingsConnection class."""
    connection = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials=credentials,
        catalog="common",
        knowledge_base="knowledge_base",
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.http_compress is True
    assert connection.url_prefix == "/dataspaces/common/datasets/knowledge_base/indexes"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == credentials


def test_async_fusion_embeddings_connection_url_prefix_kwarg(credentials: FusionCredentials) -> None:
    """Test for FusionEmbeddingsConnection class."""
    connection = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials=credentials,
        catalog="common",
        knowledge_base="knowledge_base",
        url_prefix="custom/url/prefix",
    )
    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.http_compress is True
    assert connection.url_prefix == "/dataspaces/common/datasets/knowledge_base/indexes"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == credentials


def test_async_fusion_embeddings_connection_multi_kb(credentials: FusionCredentials) -> None:
    """Test for FusionEmbeddingsConnection class."""
    connection = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials=credentials,
        catalog="common",
        knowledge_base=["knowledge_base", "knowledge_base2"],
        url_prefix="custom/url/prefix",
    )

    assert connection.host == "https://localhost:9200"
    assert connection.use_ssl is True
    assert connection.http_compress is True
    assert connection.url_prefix == "/dataspaces/common/indexes"
    assert connection.base_url == "https://fusion.jpmorgan.com/api/v1/"
    assert connection.credentials == credentials


@pytest.mark.parametrize(
    ("base_url", "raw_url", "expected_url"),
    [
        (
            "https://example.com/api/v1/",
            "%2F%7Bmyindex%7D%2F_test%2Fmock%2F",
            "/myindex/_test/mock/",
        )
    ],
)
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_async_fusion_embeddings_connection_tidy_url(
    mock_from_file: MagicMock,
    base_url: Literal["https://example.com/api/v1/"],
    raw_url: Literal["/%2F%7Bmyindex%7D%2F_test%2Fmock%2F"],
    expected_url: Literal["/myindex/_test/mock/"],
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    conn = FusionAsyncHttpConnection(
        host="localhost",
        root_url=base_url,
        credentials="dummy_credentials.json",
    )

    tidied = conn._tidy_url(raw_url)
    assert tidied == expected_url


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_async_fusion_embeddings_connection_remap_url(
    mock_from_file: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    conn = FusionAsyncHttpConnection(
        host="localhost",
        credentials="dummy_credentials.json",
    )
    url = "https://example.com/api/v1/myindex/_test/mock/_bulk/_search"

    remapped = conn._remap_endpoints(url)
    expected = "https://example.com/api/v1/myindex/_test/mock/embeddings/search"

    assert remapped == expected


@pytest.mark.parametrize(
    ("input_auth", "expected"),
    [
        (("user", "pass"), aiohttp.BasicAuth(login="user", password="pass")),
        (["user", "pass"], aiohttp.BasicAuth(login="user", password="pass")),
        ("user:pass", aiohttp.BasicAuth(login="user", password="pass")),
    ],
)
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_http_auth(
    mock_from_file: MagicMock,  # noqa: ARG001
    input_auth: tuple[str, str] | list[str] | Literal[b"user:pass", "user:pass"],
    expected: tuple[str, str],
) -> None:
    conn = FusionAsyncHttpConnection(
        host="localhost",
        http_auth=input_auth,
    )
    # Verify that the `session.auth` was set to a tuple
    assert conn._http_auth == expected


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_ssl_context_warning(mock_from_file: MagicMock) -> None:
    """Test that a warning is raised when using ssl_context with other SSL-related kwargs."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.warns(UserWarning, match="When using `ssl_context`, all other SSL related kwargs are ignored"):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            ca_certs="path/to/ca_certs",
            ssl_context=ssl.create_default_context(),
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_verify_certs_missing_ca_certs(mock_from_file: MagicMock) -> None:
    """Test that an error is raised when verify_certs is True and ca_certs is missing."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.raises(ImproperlyConfigured, match="Root certificates are missing for certificate validation"):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ca_certs=False,
            ssl_show_warn=False,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_verify_certs_invalid_ca_certs_path(mock_from_file: MagicMock) -> None:
    """Test that an error is raised when ca_certs is not a valid path."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.raises(ImproperlyConfigured, match="ca_certs parameter is not a path"):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            ca_certs="invalid/path/to/ca_certs",
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_ssl_show_warn(mock_from_file: MagicMock) -> None:
    """Test that a warning is raised when verify_certs is False and ssl_show_warn is True."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.warns(
        UserWarning, match="Connecting to https://localhost:9200 using SSL with verify_certs=False is insecure."
    ):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=False,
            ssl_show_warn=True,
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_ssl_context_creation(mock_from_file: MagicMock) -> None:
    """Test that an SSL context is created when use_ssl is True and ssl_context is None."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    connection = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    assert isinstance(connection._ssl_context, ssl.SSLContext)
    assert connection._ssl_context.verify_mode == ssl.CERT_REQUIRED
    assert connection._ssl_context.check_hostname is True


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_invalid_client_cert(mock_from_file: MagicMock) -> None:
    """Test that an error is raised when client_cert is not a valid path."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.raises(ImproperlyConfigured, match="client_cert is not a path to a file"):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            client_cert="invalid/path/to/client_cert",
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_fusion_async_http_connection_invalid_client_key(mock_from_file: MagicMock) -> None:
    """Test that an error is raised when client_key is not a valid path."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    with pytest.raises(ImproperlyConfigured, match="client_key is not a path to a file"):
        FusionAsyncHttpConnection(
            host="localhost",
            port=9200,
            http_auth=("user", "pass"),
            use_ssl=True,
            verify_certs=True,
            ssl_show_warn=False,
            client_key="invalid/path/to/client_key",
            root_url="https://fusion.jpmorgan.com/api/v1/",
            credentials="config/client_credentials.json",
            catalog="common",
            knowledge_base="knowledge_base",
        )


@patch("fusion.embeddings.FusionCredentials.from_file")
@patch("ssl.SSLContext.load_cert_chain")
@patch("pathlib.Path.is_file", return_value=True)
def test_fusion_async_http_connection_load_cert_chain(
    mock_is_file: MagicMock,  # noqa: ARG001
    mock_load_cert_chain: MagicMock,
    mock_from_file: MagicMock,
) -> None:
    """Test that client_cert and client_key are loaded correctly."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        client_cert="path/to/client_cert",
        client_key="path/to/client_key",
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    mock_load_cert_chain.assert_called_once_with("path/to/client_cert", "path/to/client_key")


@patch("fusion.embeddings.FusionCredentials.from_file")
@patch("ssl.SSLContext.load_cert_chain")
@patch("pathlib.Path.is_file", return_value=True)
def test_fusion_async_http_connection_load_cert_chain_no_key(
    mock_is_file: MagicMock,  # noqa: ARG001
    mock_load_cert_chain: MagicMock,
    mock_from_file: MagicMock,
) -> None:
    """Test that client_cert is loaded correctly when client_key is not provided."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        client_cert="path/to/client_cert",
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    mock_load_cert_chain.assert_called_once_with("path/to/client_cert")


@patch("fusion.embeddings.FusionCredentials.from_file")
@patch("ssl.SSLContext.load_verify_locations")
@patch("pathlib.Path.is_file", return_value=False)
@patch("pathlib.Path.is_dir", return_value=True)
def test_fusion_async_http_connection_ca_certs_is_dir(
    mock_is_dir: MagicMock,  # noqa: ARG001
    mock_is_file: MagicMock,  # noqa: ARG001
    mock_load_verify_locations: MagicMock,
    mock_from_file: MagicMock,
) -> None:
    """Test that ca_certs is loaded correctly when it is a directory."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
        ca_certs="path/to/ca_certs_dir",
        root_url="https://fusion.jpmorgan.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="common",
        knowledge_base="knowledge_base",
    )

    mock_load_verify_locations.assert_called_with(capath="path/to/ca_certs_dir")


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_async_make_valid_url(
    mock_from_file: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    conn = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://example.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="mycatalog",
        knowledge_base="mykb",
        index="myindex",
    )
    url = "/myindex"
    exp_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex"

    modified_url = conn._make_url_valid(url)
    assert modified_url == exp_url


@patch("fusion.embeddings.FusionCredentials.from_file")
def test_async_make_valid_url_bulk(
    mock_from_file: MagicMock,
) -> None:
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    conn = FusionAsyncHttpConnection(
        host="localhost",
        port=9200,
        http_auth=("user", "pass"),
        use_ssl=True,
        headers={"custom-header": "value"},
        http_compress=True,
        opaque_id="opaque-id",
        pool_maxsize=20,
        root_url="https://example.com/api/v1/",
        credentials="config/client_credentials.json",
        catalog="mycatalog",
        knowledge_base="mykb",
        index="myindex",
    )
    url = "/_bulk"
    exp_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex/embeddings"

    modified_url = conn._make_url_valid(url)
    assert modified_url == exp_url


class MockAsyncResponse:
    def __init__(
        self,
        text: Any = None,
        status: int = 200,
        headers: Any = CIMultiDict(),  # noqa: B008
    ) -> None:
        self._text = text
        self.status = status
        self.headers = headers

    async def text(self) -> Any:
        return self._text

    async def __aexit__(self, *args: Any) -> None:  # noqa: PYI036
        pass

    async def __aenter__(self) -> Any:
        return self


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_async_perform_request_success(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test a normal 200 success response."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("OK-content", 200)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    status, headers, raw_data = await conn.perform_request("GET", url="http://example.com/test")
    http_ok = 200
    assert status == http_ok
    assert raw_data == "OK-content"
    mock_request.assert_called_once()


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_async_perform_put_request_success_multi_kb(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test a normal 200 success response."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    body_data = b'{"test": "data"}'

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json", knowledge_base=["kb1", "kb2"])
    status, headers, raw_data = await conn.perform_request("put", url="http://example.com/test", body=body_data)
    http_ok = 200
    assert status == http_ok
    assert raw_data == ""


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_async_perform_request_refresh_endpoint(mock_from_file: MagicMock, mock_request: MagicMock) -> None:
    """Test that the '_refresh' endpoint raises OpenSearchConnectionError."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json", knowledge_base=["kb1", "kb2"])
    status, headers, raw_data = await conn.perform_request("PUT", url="http://example.com/_refresh")
    status_code = 200
    assert status == status_code


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_query_string_handling(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test query string handling."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    status, headers, raw_data = await conn.perform_request(
        "get", url="http://example.com/test", params={"key": "value"}
    )
    status_code = 200
    assert status == status_code
    assert raw_data == ""


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_req_headers_updated(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test req_headers being updated when headers is populated."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    status, headers, raw_data = await conn.perform_request(
        "get", url="http://example.com/test", headers={"Custom-Header": "value"}
    )
    status_code = 200
    assert status == status_code
    assert raw_data == ""


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_http_compress_and_body(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test if self.http_compress and body clause."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json", http_compress=True)
    body_data = b'{"test": "data"}'
    status, headers, raw_data = await conn.perform_request("post", url="http://example.com/test", body=body_data)
    status_code = 200
    assert status == status_code
    assert raw_data == ""


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_callable_http_auth(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test if callable(self._http_auth) clause."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("", status=200)

    def mock_auth(_: str, __: str, ___: str, ____: bytes) -> dict[str, str]:
        """Mock auth function."""
        return {"Authorization": "Bearer token"}

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json", http_auth=mock_auth)
    status, headers, raw_data = await conn.perform_request("get", url="http://example.com/test")
    status_code = 200
    assert status == status_code
    assert raw_data == ""


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_reraise_exceptions(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test except reraise_exceptions clause."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.side_effect = asyncio.CancelledError

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(asyncio.CancelledError):
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_log_request_fail(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test log_request_fail."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.side_effect = Exception("Test exception")

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(Exception):  # noqa: B017, PT011
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_raise_ssl_error(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test raising SSLError."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    # Define the arguments for ServerFingerprintMismatch
    expected = b"expected_fingerprint"
    got = b"got_fingerprint"
    host = "example.com"
    port = 443

    mock_request.side_effect = aiohttp_exceptions.ServerFingerprintMismatch(expected, got, host, port)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(SSLError):
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_raise_connection_timeout(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test raising ConnectionTimeout error."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.side_effect = asyncio.TimeoutError

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(ConnectionTimeout):
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_raise_connection_error(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test raising ConnectionError."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.side_effect = Exception("Test connection error")

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(ConnectionError):
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession.request")
@patch("fusion.embeddings.FusionCredentials.from_file")
async def test_http_status_code_handling(mock_from_file: MagicMock, mock_request: AsyncMock) -> None:
    """Test if not (HTTP_OK ...) clause."""
    mock_credentials = MagicMock(spec=FusionCredentials)
    mock_from_file.return_value = mock_credentials

    mock_request.return_value = MockAsyncResponse("Error", status=500)

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    with pytest.raises(TransportError):
        await conn.perform_request("get", url="http://example.com/test")


@pytest.mark.asyncio
@patch("fusion.authentication.FusionAiohttpSession")
async def test_close_method(mock_session_class: AsyncMock) -> None:
    """Test the close method."""
    mock_session = AsyncMock()
    mock_session_class.return_value = mock_session

    conn = FusionAsyncHttpConnection(host="localhost", credentials="dummy.json")
    conn.session = mock_session  # Set the mocked session

    await conn.close()

    # Assert that the session's close method was called
    mock_session.close.assert_called_once()

    # Assert that the session is set to None
    assert conn.session is None
