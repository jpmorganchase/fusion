"""Utility functions for working with embeddings in Fusion."""

from __future__ import annotations

import json
import logging
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
