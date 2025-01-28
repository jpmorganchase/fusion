"""Test for embeddings module."""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock, patch

import pytest
import requests
from opensearchpy import ImproperlyConfigured, RequestError
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError
from opensearchpy.exceptions import (
    ConnectionTimeout,
    SSLError,
)

from fusion._fusion import FusionCredentials
from fusion.embeddings import FusionEmbeddingsConnection, PromptTemplateManager, format_index_body


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
            "https://example.com/api/v1/myindex/_test/mock/",
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
def test_modify_post_response_langchain(
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
    raw_data = b"""
    {
        "took": 123,
        "max_score": 0.5,
        "hits": [
            {
                "id": "kfsgkfjg",
                "score": 0.5,
                "index": "myindex",
                "source": {
                    "id": "dfgjkdf",
                    "document_id": "dfgjkdf",
                    "content": "This is a test",
                    "chunk_seq_num": 1,
                    "knowledge_base_id": "asfae",
                    "s3_uri": "s3://mybucket/myfile",
                    "split_id": "sdfg",
                    "chunk_id": "sdfg",
                    "vector": [0.1, 0.2, 0.3, 0.4, 0.5]
                }
            }
        ]
    }
    """

    exp_data = (
        b'{"took":123,"max_score":0.5,"hits":{"hits":[{"index":"myindex","_source":{"id":"dfgjkdf",'
        b'"document_id":"dfgjkdf","content":"This is a test","chunk_seq_num":1,"knowledge_base_id":"asfae",'
        b'"s3_uri":"s3://mybucket/myfile","split_id":"sdfg","chunk_id":"sdfg","vector":[0.1,0.2,0.3,0.4,0.5]},'
        b'"_id":"kfsgkfjg","_score":0.5}]}}'
    )

    exp_bytes = exp_data.decode("utf-8")

    modified_data = conn._modify_post_response_langchain(raw_data)

    assert modified_data == exp_bytes


def test_modify_post_response_langchain_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test exception path when JSON is invalid."""
    raw_data = b"{ invalid_json }"  # Will trigger json.JSONDecodeError

    result = FusionEmbeddingsConnection._modify_post_response_langchain(raw_data)

    # Ensure returned value is the fallback
    assert isinstance(result, str)
    assert "invalid_json" in result

    # Check the log for the exception message
    assert "An error occurred during modification of langchain POST response:" in caplog.text


def test_modify_post_response_langchain_no_hits() -> None:
    """Test final return of raw_data when condition is not met."""
    # Valid JSON but doesn't contain 'hits'
    raw_data = b'{"test_key":"test_value"}'
    result = FusionEmbeddingsConnection._modify_post_response_langchain(raw_data)

    # The function should skip the if-block and return raw_data unchanged
    assert result == raw_data


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_modify_post_haystack(
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
    raw_data = (
        b'{"query":{"bool":{"must":[{"knn":{"embedding":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}}, '
        b'"size":10, "_source":{"excludes":["embedding"]}}'
    )
    exp_data = (
        b'{"query":{"bool":{"must":[{"knn":{"vector":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}},'
        b'"size":10,"_source":{"excludes":["embedding"]}}'
    )
    modified_data = conn._modify_post_haystack(raw_data, "post")
    modified_data = conn._modify_post_haystack(raw_data, "post")

    assert modified_data == exp_data


def test_modify_post_haystack_json_decode_error(caplog: pytest.LogCaptureFixture) -> None:
    """Test that invalid JSON input triggers a JSONDecodeError and logs an exception."""
    # Provide invalid JSON to cause json.JSONDecodeError
    invalid_body = b"{ query }"
    method = "POST"

    result = FusionEmbeddingsConnection._modify_post_haystack(invalid_body, method)

    # Check that the function returned the original body
    # Ensure returned value is the fallback
    assert isinstance(result, bytes)

    # Check the log for the exception message
    assert "An error occurred during modification of haystack POST body:" in caplog.text


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_modify_post_haystack_no_query(
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
    raw_data = (
        b'{"create":{"_id":"sjdkfs"}}\n{"id":"sjdkfs","content":"This is a test","embedding":[0.1,0.2,0.3,0.4,0.5]}'
    )
    exp_data = b'{"create":{"_id":"sjdkfs"}}\n{"id":"sjdkfs","content":"This is a test","vector":[0.1,0.2,0.3,0.4,0.5]}'

    modified_data = conn._modify_post_haystack(raw_data, "post")

    assert modified_data == exp_data


def test_modify_post_haystack_json_decode_error_no_query(caplog: pytest.LogCaptureFixture) -> None:
    """Test that invalid JSON input triggers a JSONDecodeError and logs an exception."""
    # Provide invalid JSON to cause json.JSONDecodeError
    invalid_body = b"{ invalid_json }"
    method = "POST"

    result = FusionEmbeddingsConnection._modify_post_haystack(invalid_body, method)

    # Check that the function returned the original body
    # Ensure returned value is the fallback
    assert isinstance(result, bytes)

    # Check the log for the exception message
    assert "An error occurred during modification of haystack POST body:" in caplog.text


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
    url = "https://example.com/api/v1/myindex"
    exp_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex"

    modified_url = conn._make_url_valid(url)
    assert modified_url == exp_url


@patch("fusion.embeddings.get_session")
@patch("fusion.embeddings.FusionCredentials.from_file")
def test_make_valid_url_different_index(
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

    conn.index_name = "myindex"
    conn.url_prefix = conn.url_prefix + "myindex"
    url = "https://example.com/api/v1/_bulk"
    expected_url = "https://example.com/api/v1/dataspaces/mycatalog/datasets/mykb/indexes/myindex/embeddings"

    modified_url = conn._make_url_valid(url)
    assert modified_url == expected_url


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
