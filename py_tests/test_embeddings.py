"""Test for embeddings module."""

from fusion.embeddings import format_index_body


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

