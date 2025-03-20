"""Test for embeddings_utils.py module."""

import pytest

from fusion.embeddings_utils import (
    _modify_post_haystack,
    _modify_post_response_langchain,
    _retrieve_index_name_from_bulk_body,
)


def test_retrieve_index_name_from_bulk_body() -> None:
    """Test for _retrieve_index_name_from_bulk_body function."""

    bulk_body = b'{"index":{"_id":"23134werw", "_index":"test-index"}}\n{"vector":[],"content":"content","metadata":{}}'
    index_name = _retrieve_index_name_from_bulk_body(bulk_body)
    assert index_name == "test-index"


def test_retrieve_index_name_from_bulk_body_second_obj() -> None:
    """Test for _retrieve_index_name_from_bulk_body function."""

    bulk_body = b'{"vector":[],"content":"content","metadata":{}}\n{"index":{"_id":"23134werw", "_index":"test-index"}}'
    index_name = _retrieve_index_name_from_bulk_body(bulk_body)
    assert index_name == "test-index"


def test_retrieve_index_name_from_bulk_body_no_index() -> None:
    """Test for _retrieve_index_name_from_bulk_body function."""

    bulk_body_no_index = b'{"index":{"_id":"23134werw"}}\n{"vector":[],"content":"content","metadata":{}}'
    with pytest.raises(ValueError, match="Index name not found in bulk body"):
        _retrieve_index_name_from_bulk_body(bulk_body_no_index)


def test_retrieve_index_name_from_bulk_body_empty() -> None:
    """Test for _retrieve_index_name_from_bulk_body function."""
    empty_bulk_body = b""
    with pytest.raises(ValueError, match="Index name not found in bulk body"):
        _retrieve_index_name_from_bulk_body(empty_bulk_body)


def test_modify_post_haystack() -> None:
    raw_data = (
        b'{"query":{"bool":{"must":[{"knn":{"embedding":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}}, '
        b'"size":10, "_source":{"excludes":["embedding"]}}'
    )
    exp_data = (
        b'{"query":{"bool":{"must":[{"knn":{"vector":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}},'
        b'"size":10,"_source":{"excludes":["embedding"]}}'
    )
    modified_data = _modify_post_haystack(knowledge_base="mykb", body=raw_data, method="post")

    assert modified_data == exp_data


def test_modify_post_haystack_multi_kb() -> None:
    knowledge_base = ["mykb", "mykb2"]
    raw_data = (
        b'{"query":{"bool":{"must":[{"knn":{"embedding":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}}, '
        b'"size":10, "_source":{"excludes":["embedding"]}}'
    )
    exp_data = (
        b'{"query":{"hybrid":{"queries":[{"knn":{"vector":{"vector":[0.1,0.2,0.3,0.4,0.5],"k":5}}}]}},'
        b'"size":10,"_source":{"excludes":["embedding"]},"datasets":["mykb","mykb2"]}'
    )
    modified_data = _modify_post_haystack(knowledge_base=knowledge_base, body=raw_data, method="post")
    modified_data = _modify_post_haystack(knowledge_base=knowledge_base, body=raw_data, method="post")

    assert modified_data == exp_data


def test_modify_post_haystack_json_decode_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that invalid JSON input triggers a JSONDecodeError and logs an exception."""
    # Provide invalid JSON to cause json.JSONDecodeError
    invalid_body = b"{ query }"
    method = "POST"

    result = _modify_post_haystack(knowledge_base="mykb", body=invalid_body, method=method)

    # Check that the function returned the original body
    # Ensure returned value is the fallback
    assert isinstance(result, bytes)

    # Check the log for the exception message
    assert "An error occurred during modification of haystack POST body:" in caplog.text


def test_modify_post_haystack_no_query() -> None:
    raw_data = (
        b'{"create":{"_id":"sjdkfs"}}\n{"id":"sjdkfs","content":"This is a test","embedding":[0.1,0.2,0.3,0.4,0.5]}'
    )
    exp_data = b'{"create":{"_id":"sjdkfs"}}\n{"id":"sjdkfs","content":"This is a test","vector":[0.1,0.2,0.3,0.4,0.5]}'

    modified_data = _modify_post_haystack(knowledge_base="mykb", body=raw_data, method="post")

    assert modified_data == exp_data


def test_modify_post_haystack_json_decode_error_no_query(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that invalid JSON input triggers a JSONDecodeError and logs an exception."""
    # Provide invalid JSON to cause json.JSONDecodeError
    invalid_body = b"{ invalid_json }"
    method = "POST"

    result = _modify_post_haystack(knowledge_base="mykb", body=invalid_body, method=method)

    # Check that the function returned the original body
    # Ensure returned value is the fallback
    assert isinstance(result, bytes)

    # Check the log for the exception message
    assert "An error occurred during modification of haystack POST body:" in caplog.text


def test_modify_post_response_langchain() -> None:
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

    modified_data = _modify_post_response_langchain(raw_data)

    assert modified_data == exp_bytes


def test_modify_post_response_langchain_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test exception path when JSON is invalid."""
    raw_data = b"{ invalid_json }"  # Will trigger json.JSONDecodeError

    result = _modify_post_response_langchain(raw_data)

    # Ensure returned value is the fallback
    assert isinstance(result, str)
    assert "invalid_json" in result

    # Check the log for the exception message
    assert "An error occurred during modification of langchain POST response:" in caplog.text


def test_modify_post_response_langchain_no_hits() -> None:
    """Test final return of raw_data when condition is not met."""
    # Valid JSON but doesn't contain 'hits'
    raw_data = b'{"test_key":"test_value"}'
    result = _modify_post_response_langchain(raw_data)

    # The function should skip the if-block and return raw_data unchanged
    assert result == raw_data
