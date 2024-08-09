# from fsspec.implementations.http import sync
import json
from pathlib import Path
from typing import Any, Optional
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import fsspec
import pytest
from aiohttp import ClientResponse

from fusion._fusion import FusionCredentials
from fusion.fusion_filesystem import FusionHTTPFileSystem


@pytest.fixture()
def http_fs_instance(credentials_examples: Path) -> FusionHTTPFileSystem:
    """Fixture to create a new instance for each test."""
    creds = FusionCredentials.from_file(credentials_examples)
    return FusionHTTPFileSystem(credentials=creds)


def test_filesystem(
    example_creds_dict: dict[str, Any], example_creds_dict_https_pxy: dict[str, Any], tmp_path: Path
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)
    assert FusionHTTPFileSystem(creds)

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict_https_pxy, f)
    creds = FusionCredentials.from_file(credentials_file)
    assert FusionHTTPFileSystem(creds)

    kwargs = {"client_kwargs": {"credentials": creds}}

    assert FusionHTTPFileSystem(None, **kwargs)

    kwargs = {"client_kwargs": {"credentials": 3.14}}  # type: ignore
    with pytest.raises(ValueError, match="Credentials not provided"):
        FusionHTTPFileSystem(None, **kwargs)


@pytest.mark.asyncio()
async def test_not_found_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    # Create a mock response object
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 404
    response.text = mock.AsyncMock(return_value="404 NotFound")

    # Use a context manager to catch the FileNotFoundError
    with pytest.raises(FileNotFoundError):
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")


@pytest.mark.asyncio()
async def test_other_error_status(credentials: FusionCredentials) -> None:
    # Create a mock response object
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 500  # Some error status
    response.text = mock.AsyncMock(return_value="Internal server error")

    # Instance of your class
    instance = FusionHTTPFileSystem(credentials)

    # Patching the internal method to just throw an Exception for testing
    with mock.patch.object(instance, "_raise_not_found_for_status", side_effect=Exception("Custom error")):
        with pytest.raises(Exception, match="Custom error"):
            await instance._async_raise_not_found_for_status(response, "http://example.com")

        response.text.assert_awaited_once()
        assert response.reason == "Internal server error", "The reason should be updated to the response text"


@pytest.mark.asyncio()
async def test_successful_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    # Create a mock response object with a successful status code
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 200  # HTTP 200 OK
    response.text = mock.AsyncMock(return_value="Success response body")

    # Attempt to call the method with the mock response
    # Since this is the successful path, we are primarily checking that nothing adverse happens
    try:
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")
    except FileNotFoundError as e:
        pytest.fail(f"No exception should be raised for a successful response, but got: {e}")


@pytest.mark.asyncio()
async def test_decorate_url_with_http_async(http_fs_instance: FusionHTTPFileSystem) -> None:
    url = "resource/path"
    exp_res = f"{http_fs_instance.client_kwargs['root_url']}catalogs/{url}"
    result = await http_fs_instance._decorate_url_a(url)
    assert result == exp_res


@pytest.mark.asyncio()
async def test_isdir_true(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = AsyncMock(return_value="decorated_path_dir")  # type: ignore
    http_fs_instance._info = AsyncMock(return_value={"type": "directory"})
    result = await http_fs_instance._isdir("path_dir")
    assert result


@pytest.mark.asyncio()
async def test_isdir_false(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = AsyncMock(return_value="decorated_path_file")  # type: ignore
    http_fs_instance._info = AsyncMock(return_value={"type": "file"})
    result = await http_fs_instance._isdir("path_file")
    assert not result


@pytest.mark.parametrize(
    ("overwrite", "exists", "expected_result"),
    [
        (True, True, 0),  # Overwrite enabled, file exists
        (True, False, 0),  # Overwrite enabled, file does not exist
    ],
)
@pytest.mark.asyncio()
@patch("aiohttp.ClientSession")  # type: ignore
async def test_stream_file(MockClientSession, overwrite: bool, exists: bool, expected_result: int) -> None:
    url = "http://example.com/data"
    output_file = AsyncMock(spec=fsspec.spec.AbstractBufferedFile)
    start, end = 0, 10
    results: list[tuple[bool, str, Optional[str]]] = [(False, "", "")] * 1  # single element list
    idx = 0
    fs = AsyncMock(spec=fsspec.AbstractFileSystem)
    fs.exists.return_value = exists

    # Create a mock response object with the necessary context manager methods
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.content.iter_chunked = AsyncMock(return_value=iter([b"0123456789"]))
    # Mock the __enter__ method to return the mock response itself
    mock_response.__aenter__.return_value = mock_response
    # Mock the __exit__ method to do nothing
    mock_response.__aexit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response
    MockClientSession.return_value.__aenter__.return_value = mock_session

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem()

    # Run the async function
    result = await http_fs_instance.stream_single_file(url, output_file)

    # Assertions to verify the behavior
    assert result[0] == expected_result
    if not overwrite and exists:
        fs.exists.assert_called_once_with(output_file)
        assert results[idx] == (True, output_file, None)
    else:
        output_file.seek.assert_called_once_with(start)
        output_file.write.assert_called_once_with(b"0123456789")
        assert results[idx] == (True, output_file, None)
