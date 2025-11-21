"""Utility functions for working with embeddings in Fusion."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import requests

logger = logging.getLogger(__name__)


def _format_full_index_response(response: requests.Response) -> pd.DataFrame:
    """Format get index response.

    Args:
        response (requests.Response): Response object.

    Returns:
        pd.DataFrame: Dataframe containing the response formatted with a column for each index.
    """
    df_resp = pd.json_normalize(response.json())
    df2 = df_resp.transpose()
    df2.index = df2.index.map(str)
    df2.columns = pd.Index(df2.loc["settings.index.provided_name"])
    df2 = df2.rename(columns=lambda x: x.split("index-")[-1])
    df2.columns.names = ["index_name"]
    df2.loc["settings.index.creation_date"] = pd.to_datetime(df2.loc["settings.index.creation_date"], unit="ms")
    multi_index = [index.split(".", 1) for index in df2.index]
    df2.index = pd.MultiIndex.from_tuples(multi_index)
    return df2


def _format_summary_index_response(response: requests.Response) -> pd.DataFrame:
    """Format summary of get index response.

    Args:
        response (requests.Response): Response object

    Returns:
        pd.DataFrame: Dataframe containing the response formatted with a column for each index.
    """
    index_list = response.json()
    idices = []
    for index in index_list:
        name = index.get("settings").get("index").get("provided_name").split("index-")[-1]
        props = index.get("mappings").get("properties")
        for prop, info in props.items():
            if info.get("type") == "knn_vector":
                vector_field_name = prop
                vector_dimension = info.get("dimension")
        idices.append(
            {
                "index_name": name,
                "vector_field_name": vector_field_name,
                "vector_dimension": vector_dimension,
            }
        )
    summary_df = pd.json_normalize(idices)
    summary_df = summary_df.set_index("index_name")

    return summary_df.transpose()


def _retrieve_index_name_from_bulk_body(body: bytes | None) -> str:
    body_str = body.decode("utf-8") if body else ""

    json_objects = body_str.split("\n")

    for json_str in json_objects:
        if json_str.strip():
            json_obj = json.loads(json_str)

            if "index" in json_obj and "_index" in json_obj["index"]:
                index_name = str(json_obj["index"]["_index"])
                return index_name

    raise ValueError("Index name not found in bulk body")


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


def _modify_post_haystack(knowledge_base: str | list[str] | None, body: bytes | None, method: str) -> bytes | None:
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
                if isinstance(knowledge_base, list) and knn_list != {}:
                    query_dict["query"] = {"hybrid": {"queries": knn_list}}
                    query_dict["datasets"] = knowledge_base
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


def extract_index_name_from_url(url: str) -> str | None:
    """Extract index name from url.


    Args:
        url (str): The URL to extract index name from.

    Returns:
        str | None: The extracted index name, or None if not found
    """
    index_name_match = re.search(r"/indexes/([^/]+)", url)
    if index_name_match:
        return index_name_match.group(1)
    return None


def is_index_creation_request(url: str, method: str, body: bytes | None) -> bool:
    """Check if request is index creation request.

    Args:
        url (str): The request URL
        method (str): http method
        body (bytes | None): request body

    Returns:
        bool: True if request is index creation request
    """

    if not (body and method.lower() == "post" and "indexes/" in url):
        return False

    index_pos = url.find("indexes/")

    if index_pos == -1:
        return False

    path_after_indexes = url[index_pos + len("indexes/") :]

    path_segments = path_after_indexes.strip("/").split("/")

    return len(path_segments) == 1 and path_segments[0] not in ["embeddings", "search", ""]


def transform_index_creation_body(body: bytes) -> bytes:
    """Transform index creation body by filtering out unsupported parameters.

    This function performs minimal transformation - it only removes the 'method'
    parameter from vector fields as the Fusion Embeddings Service does not support
    this parameter currently. Everything else is preserved as the user specified.

    Args:
        body (bytes): _description_

    Returns:
        bytes: _description_
    """
    try:
        body_dict = json.loads(body.decode("utf-8"))

        if "mappings" in body_dict and "properties" in body_dict["mappings"]:
            properties = body_dict["mappings"]["properties"]

            for field_name, field_config in properties.items():
                if isinstance(field_config, dict):
                    field_type = field_config.get("type", "")

                    if (
                        any(vector_type in field_type.lower() for vector_type in ["vector", "knn"])
                        and "method" in field_config
                    ):
                        logger.debug(f"Removing unsupported 'method' parameter from vector field '{field_name}'")
                        field_config.pop("method")

        logger.debug("Filtered unsupported parameters from index creation body")
        logger.debug("Original: %s", json.dumps(json.loads(body.decode("utf-8")), indent=2))
        logger.debug("Filtered:  %s", json.dumps(body_dict, indent=2))

        return json.dumps(body_dict).encode("utf-8")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Failed to filter index creation body: %s", e)
        return body
