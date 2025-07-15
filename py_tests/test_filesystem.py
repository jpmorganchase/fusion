import copy
import io
import json
import logging
from pathlib import Path
from typing import Any, Literal
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import fsspec
import pytest
from aiohttp import ClientResponse

from fusion._fusion import FusionCredentials
from fusion.exceptions import APIResponseError
from fusion.fusion_filesystem import FusionHTTPFileSystem


@pytest.fixture
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


@pytest.mark.asyncio
async def test_not_found_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    # Create a mock response object
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 404
    response.text = mock.AsyncMock(return_value="404 NotFound")

    # Use a context manager to catch the FileNotFoundError
    with pytest.raises(
        APIResponseError,
        match="APIResponseError: Status 404, Error when accessing http://example.com, Error: .*http://example.com",
    ):
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_successful_status(http_fs_instance: FusionHTTPFileSystem) -> None:
    # Create a mock response object with a successful status code
    response = mock.MagicMock(spec=ClientResponse)
    response.status = 200  # HTTP 200 OK
    response.text = mock.AsyncMock(return_value="Success response body")

    # Attempt to call the method with the mock response
    # Since this is the successful path, we are primarily checking that nothing adverse happens
    try:
        await http_fs_instance._async_raise_not_found_for_status(response, "http://example.com")
    except APIResponseError as e:
        pytest.fail(f"No exception should be raised for a successful response, but got: {e}")


@pytest.mark.asyncio
async def test_decorate_url_with_http_async(http_fs_instance: FusionHTTPFileSystem) -> None:
    url = "resource/path"
    exp_res = f"{http_fs_instance.client_kwargs['root_url']}catalogs/{url}"
    result = await http_fs_instance._decorate_url_a(url)
    assert result == exp_res


@pytest.mark.asyncio
async def test_isdir_true(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = AsyncMock(return_value="decorated_path_dir")  # type: ignore
    http_fs_instance._info = AsyncMock(return_value={"type": "directory"})  # type: ignore
    result = await http_fs_instance._isdir("path_dir")
    assert result


@pytest.mark.asyncio
async def test_isdir_false(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._decorate_url = AsyncMock(return_value="decorated_path_file")  # type: ignore
    http_fs_instance._info = AsyncMock(return_value={"type": "file"})  # type: ignore
    result = await http_fs_instance._isdir("path_file")
    assert not result


@pytest.mark.asyncio
async def test_check_sess_open(http_fs_instance: FusionHTTPFileSystem) -> None:
    new_fs_session_closed = not http_fs_instance._check_session_open()
    assert new_fs_session_closed

    # Corresponds to running .set_session()
    session_mock = MagicMock()
    session_mock.closed = False
    http_fs_instance._session = session_mock
    fs_session_open = http_fs_instance._check_session_open()
    assert fs_session_open

    # Corresponds to situation where session was closed
    session_mock2 = MagicMock()
    session_mock2.closed = True
    http_fs_instance._session = session_mock2
    fs_session_closed = not http_fs_instance._check_session_open()
    assert fs_session_closed


@pytest.mark.asyncio
async def test_async_startup(http_fs_instance: FusionHTTPFileSystem) -> None:
    http_fs_instance._session = None
    with (
        patch("fusion.fusion_filesystem.FusionHTTPFileSystem.set_session") as SetSessionMock,
        pytest.raises(RuntimeError) as re,
    ):
        await http_fs_instance._async_startup()
    SetSessionMock.assert_called_once()
    assert re.match("FusionFS session closed before operation")

    # Mock an open session
    MockClient = MagicMock()
    MockClient.closed = False
    http_fs_instance._session = MockClient
    with patch("fusion.fusion_filesystem.FusionHTTPFileSystem.set_session") as SetSessionMock2:
        await http_fs_instance._async_startup()
    SetSessionMock2.assert_called_once()


@pytest.mark.asyncio
async def test_exists_methods(http_fs_instance: FusionHTTPFileSystem) -> None:
    with patch("fusion.fusion_filesystem.HTTPFileSystem.exists") as MockExists:
        MockExists.return_value = True
        exists_out = http_fs_instance.exists("dummy_path")
        MockExists.assert_called_once()
    assert exists_out

    with (
        patch("fusion.fusion_filesystem.HTTPFileSystem._exists") as Mock_Exists,
        patch("fusion.fusion_filesystem.FusionHTTPFileSystem._async_startup") as MockStartup,
    ):
        Mock_Exists.return_value = True
        _exists_out = await http_fs_instance._exists("dummy_path")
        Mock_Exists.assert_awaited_once()
        MockStartup.assert_awaited_once()
    assert _exists_out


@patch("requests.Session")
def test_stream_single_file(mock_session_class: MagicMock, example_creds_dict: dict[str, Any], tmp_path: Path) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=fsspec.spec.AbstractBufferedFile)
    output_file.path = "./output_file_path/file.txt"
    output_file.name = "file.txt"

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create a mock response object with the necessary context manager methods
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b"0123456789", b""])
    # Mock the __enter__ method to return the mock response itself
    mock_response.__enter__.return_value = mock_response
    # Mock the __exit__ method to do nothing
    mock_response.__exit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.sync_session = mock_session

    # Run the function
    results = http_fs_instance.stream_single_file(url, output_file)

    # Assertions to verify the behavior
    output_file.write.assert_any_call(b"0123456789")
    assert results == (True, output_file.path, None)


@patch("requests.Session")
def test_stream_single_file_exception(
    mock_session_class: MagicMock, example_creds_dict: dict[str, Any], tmp_path: Path
) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=fsspec.spec.AbstractBufferedFile)
    output_file.path = "./output_file_path/file.txt"
    output_file.name = "file.txt"

    # Create a mock response object with the necessary context manager methods
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(side_effect=Exception("Test exception"))
    # Mock the __enter__ method to return the mock response itself
    mock_response.__enter__.return_value = mock_response
    # Mock the __exit__ method to do nothing
    mock_response.__exit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_session_class.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.sync_session = mock_session

    # Run the function and catch the exception
    results = http_fs_instance.stream_single_file(url, output_file)

    # Assertions to verify the behavior
    output_file.close.assert_called_once()
    assert results == (False, output_file.path, "Test exception")


@pytest.mark.asyncio
@patch("fsspec.asyn._run_coros_in_chunks", new_callable=AsyncMock)
async def test_download_single_file_async(
    mock_run_coros_in_chunks: mock.AsyncMock, example_creds_dict: dict[str, Any], tmp_path: Path
) -> None:
    # Define the mock return value
    mock_run_coros_in_chunks.return_value = [True, True, True]

    url = "http://example.com/data"
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    file_size = 20
    chunk_size = 10
    n_threads = 3

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:  # noqa: ASYNC101, ASYNC230
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.set_session = AsyncMock(return_value=AsyncMock())

    # Mock the _fetch_range methodA
    http_fs_instance._fetch_range = AsyncMock()  # type: ignore

    # Run the async function
    result = await http_fs_instance._download_single_file_async(url, output_file, file_size, chunk_size, n_threads)

    # Assertions to verify the behavior
    assert result == (True, output_file.path, None)  # type: ignore
    output_file.close.assert_called_once()

    # Simulate an exception in the mock return value
    mock_run_coros_in_chunks.return_value = [Exception("Test exception")]
    result = await http_fs_instance._download_single_file_async(url, output_file, file_size, chunk_size, n_threads)

    # Assertions to verify the behavior on exception
    assert result == (False, output_file.path, "Test exception")  # type: ignore
    output_file.close.assert_called()


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_fetch_range_exception(
    mock_client_session: mock.AsyncMock, example_creds_dict: dict[str, Any], tmp_path: Path
) -> None:
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    output_file.seek = MagicMock()
    output_file.write = MagicMock()

    # Create a mock response object with the necessary context manager methods
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.read = AsyncMock(side_effect=Exception("Test exception"))
    mock_response.status = 500
    # Mock the __aenter__ method to return the mock response itself
    mock_response.__aenter__.return_value = mock_response
    # Mock the __aexit__ method to do nothing
    mock_response.__aexit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_client_session.return_value.__aenter__.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:  # noqa: ASYNC101, ASYNC230
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.kwargs = {}  # Add any necessary kwargs here

    # Assertions to verify the behavior
    output_file.seek.assert_not_called()
    output_file.write.assert_not_called()


@pytest.mark.asyncio
@patch("aiohttp.ClientSession")
async def test_fetch_range_success(
    mock_client_session: mock.AsyncMock, example_creds_dict: dict[str, Any], tmp_path: Path
) -> None:
    url = "http://example.com/data"
    output_file = MagicMock(spec=io.IOBase)
    output_file.path = "./output_file_path/file.txt"
    output_file.seek = MagicMock()
    output_file.write = MagicMock()
    start = 0
    end = 10

    # Create a mock response object with the necessary context manager methods
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.read = AsyncMock(return_value=b"some data")
    mock_response.status = 200
    # Mock the __aenter__ method to return the mock response itself
    mock_response.__aenter__.return_value = mock_response
    # Mock the __aexit__ method to do nothing
    mock_response.__aexit__.return_value = None

    # Set up the mock session to return the mock response
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response
    mock_client_session.return_value.__aenter__.return_value = mock_session

    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:  # noqa: ASYNC101, ASYNC230
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Create an instance of FusionHTTPFileSystem
    http_fs_instance = FusionHTTPFileSystem(credentials=creds)
    http_fs_instance.kwargs = {}  # Add any necessary kwargs here

    # Run the async function and ensure it completes successfully
    await http_fs_instance._fetch_range(mock_session, url, start, end, output_file)  # noqa: W0212

    # Assertions to verify the behavior
    output_file.seek.assert_called_once_with(0)
    output_file.write.assert_called_once_with(b"some data")
    mock_response.raise_for_status.assert_not_called()
    mock_session.get.assert_called_once_with(url + f"?downloadRange=bytes={start}-{end - 1}", **http_fs_instance.kwargs)


@pytest.mark.parametrize(
    ("n_threads", "is_local_fs", "expected_method"),
    [
        (10, False, "stream_single_file"),
        (10, True, "_download_single_file_async"),
    ],
)
@patch("fusion.utils.get_default_fs")
@patch("fsspec.asyn.sync")
@patch.object(FusionHTTPFileSystem, "stream_single_file", new_callable=AsyncMock)
@patch.object(FusionHTTPFileSystem, "_download_single_file_async", new_callable=AsyncMock)
def test_get(  # noqa: PLR0913
    mock_download_single_file_async: mock.AsyncMock,
    mock_stream_single_file: mock.AsyncMock,
    mock_sync: mock.AsyncMock,
    mock_get_default_fs: MagicMock,
    n_threads: int,
    is_local_fs: bool,
    expected_method: Literal["stream_single_file", "_download_single_file_async"],
    example_creds_dict: dict[str, Any],
    tmp_path: Path,
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Arrange
    fs = FusionHTTPFileSystem(credentials=creds)
    rpath = "http://example.com/data"
    chunk_size = 5 * 2**20
    kwargs = {"n_threads": n_threads, "is_local_fs": is_local_fs, "headers": {"Content-Length": "100"}}

    mock_file = AsyncMock(spec=fsspec.spec.AbstractBufferedFile)
    mock_default_fs = MagicMock()
    mock_default_fs.open.return_value = mock_file
    mock_get_default_fs.return_value = mock_default_fs
    mock_sync.side_effect = lambda _, func, *args, **kwargs: func(*args, **kwargs)

    # Act
    _ = fs.get(rpath, mock_file, chunk_size, **kwargs)

    # Assert
    if expected_method == "stream_single_file":
        mock_stream_single_file.assert_called_once_with(str(rpath), mock_file, block_size=chunk_size)
        mock_download_single_file_async.assert_not_called()
    else:
        mock_download_single_file_async.assert_called_once_with(
            str(rpath) + "/operationType/download", mock_file, 100, chunk_size, n_threads
        )
        mock_stream_single_file.assert_not_called()

    mock_get_default_fs.return_value.open.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overwrite", "preserve_original_name", "expected_lpath"),
    [
        (True, False, "local_file.txt"),
        (False, False, "local_file.txt"),
        (True, True, "original_file.txt"),
        (False, True, "original_file.txt"),
    ],
)
@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
@patch("aiohttp.ClientSession")
def test_download(  # noqa: PLR0913
    mock_client_session: mock.AsyncMock,
    mock_fs_class: mock.AsyncMock,
    mock_set_session: mock.AsyncMock,
    mock_get: mock.AsyncMock,
    overwrite: bool,
    preserve_original_name: bool,
    expected_lpath: Literal["local_file.txt", "original_file.txt"],
    example_creds_dict: dict[str, Any],
    tmp_path: Path,
) -> None:
    credentials_file = tmp_path / "client_credentials.json"
    with Path(credentials_file).open("w") as f:
        json.dump(example_creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Arrange
    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    rpath = "http://example.com/data"
    lpath = "local_file.txt"
    chunk_size = 5 * 2**20

    mock_session = mock_client_session
    mock_set_session.return_value = mock_session
    mock_response = AsyncMock()
    mock_response.raise_for_status = AsyncMock()
    mock_response.headers = {"Content-Length": "100", "x-jpmc-file-name": "original_file.txt"}
    mock_response.__aenter__.return_value = mock_response
    mock_session.head.return_value = mock_response

    # Act
    result = fs.download(lfs, rpath, lpath, chunk_size, overwrite, preserve_original_name)

    # Assert
    if overwrite:
        assert result == ("mocked_return", "mocked_lpath", "mocked_extra")
        mock_get.assert_called_once_with(
            str(rpath),
            lfs.open(expected_lpath, "wb"),
            chunk_size=chunk_size,
            headers={"Content-Length": "100", "x-jpmc-file-name": "original_file.txt"},
            is_local_fs=False,
        )
    elif preserve_original_name:
        assert result == (True, Path(expected_lpath), None)
    else:
        assert result == (True, lpath, None)

@patch.object(FusionHTTPFileSystem, "get", return_value=("mocked_return", "mocked_lpath", "mocked_extra"))
@patch.object(FusionHTTPFileSystem, "set_session", new_callable=AsyncMock)
@patch("fsspec.AbstractFileSystem", autospec=True)
def test_download_mkdir_logs_exception(
    mock_fs_class: MagicMock,
    mock_set_session: AsyncMock, # noqa: ARG001
    mock_get: MagicMock, # noqa: ARG001
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Setup dummy credentials
    creds_dict: dict[str, Any] = {
        "auth_url": "https://authe.mysite.com/as/token.oauth2",
        "client_id": "test",
        "client_secret": "secret",
        "proxies": {},
        "scope": "test_scope",
    }
    credentials_file = tmp_path / "creds.json"
    with credentials_file.open("w") as f:
        json.dump(creds_dict, f)
    creds = FusionCredentials.from_file(credentials_file)

    # Arrange
    fs = FusionHTTPFileSystem(credentials=creds)
    lfs = mock_fs_class.return_value
    rpath = "http://example.com/skip"
    lpath = tmp_path / "output/file.txt"

    # Simulate parent dir missing and mkdir failing
    lfs.exists.return_value = False
    lfs.mkdir.side_effect = Exception("directory exists")

    caplog.set_level(logging.INFO)

    # Act
    result = fs.download(
        lfs=lfs,
        rpath=rpath,
        lpath=lpath,
        chunk_size=5 * 2**20,
        overwrite=True,
        preserve_original_name=False,
    )

    # Assert expected call result
    assert result == ("mocked_return", "mocked_lpath", "mocked_extra")

    # Confirm log message and exception info were recorded
    assert any("exists already" in record.getMessage() for record in caplog.records)


PAGINATED = 2


@pytest.mark.asyncio
@patch("fusion.fusion_filesystem._merge_responses", return_value={"merged": True})
async def test__changes_single_page(mock_merge: MagicMock) -> None:
    with patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None):
        fs = FusionHTTPFileSystem()
        with (
            patch.object(fs, "_decorate_url", MagicMock(return_value="decorated_url")),
            patch.object(fs, "set_session", AsyncMock()),
            patch.object(fs, "_raise_not_found_for_status", MagicMock()),
        ):
            fs.kwargs = {}
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={"foo": "bar"})
            mock_response.headers = {}
            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            fs.set_session.return_value = mock_session
            result = await fs._changes("input_url")
            assert result == {"merged": True}
            mock_merge.assert_called_once_with([{"foo": "bar"}])
            fs._decorate_url.assert_called_once_with("input_url")  # type: ignore[attr-defined]
            fs._raise_not_found_for_status.assert_called_once_with(mock_response, "decorated_url")  # type: ignore[attr-defined]


@pytest.mark.asyncio
@patch("fusion.fusion_filesystem._merge_responses", return_value={"merged": True})
async def test__changes_multiple_pages(mock_merge: MagicMock) -> None:
    with patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None):
        fs = FusionHTTPFileSystem()
        url = "input_url"
        decorated_url = "decorated_url"
        resp1 = {"page": 1}
        resp2 = {"page": 2}
        with (
            patch.object(fs, "_decorate_url", MagicMock(return_value=decorated_url)),
            patch.object(fs, "set_session", AsyncMock()),
            patch.object(fs, "_raise_not_found_for_status", MagicMock()),
        ):
            fs.kwargs = {}
            mock_response1 = AsyncMock()
            mock_response1.json = AsyncMock(return_value=resp1)
            mock_response1.headers = {"x-jpmc-next-token": "token2"}
            mock_response2 = AsyncMock()
            mock_response2.json = AsyncMock(return_value=resp2)
            mock_response2.headers = {}
            mock_session = MagicMock()
            mock_session.get.side_effect = [
                MagicMock(__aenter__=AsyncMock(return_value=mock_response1), __aexit__=AsyncMock()),
                MagicMock(__aenter__=AsyncMock(return_value=mock_response2), __aexit__=AsyncMock()),
            ]
            fs.set_session.return_value = mock_session

            result = await fs._changes(url)
            assert result == {"merged": True}
            mock_merge.assert_called_once_with([resp1, resp2])
            assert fs._raise_not_found_for_status.call_count == PAGINATED  # type: ignore[attr-defined]


@pytest.mark.asyncio
@patch("fusion.fusion_filesystem.logger")
@patch("fusion.fusion_filesystem._merge_responses", return_value={"merged": True})
async def test__changes_json_exception(mock_merge: MagicMock, mock_logger: MagicMock) -> None:
    with patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None):
        fs = FusionHTTPFileSystem()
        url = "input_url"
        decorated_url = "decorated_url"
        with (
            patch.object(fs, "_decorate_url", MagicMock(return_value=decorated_url)),
            patch.object(fs, "set_session", AsyncMock()),
            patch.object(fs, "_raise_not_found_for_status", MagicMock()),
        ):
            fs.kwargs = {}
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(side_effect=Exception("bad json"))
            mock_response.headers = {}
            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            fs.set_session.return_value = mock_session
            result = await fs._changes(url)
            assert result == {"merged": True}
            mock_merge.assert_called_once_with([{}])
            mock_logger.exception.assert_called()
            fs._raise_not_found_for_status.assert_called_once_with(mock_response, decorated_url)  # type: ignore[attr-defined]


@pytest.mark.asyncio
@patch("fusion.fusion_filesystem.logger")
async def test__changes_outer_exception(mock_logger: MagicMock) -> None:
    with patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None):
        fs = FusionHTTPFileSystem()
        url = "input_url"
        decorated_url = "decorated_url"
        with (
            patch.object(fs, "_decorate_url", MagicMock(return_value=decorated_url)),
            patch.object(fs, "set_session", AsyncMock()),
            patch.object(fs, "_raise_not_found_for_status", MagicMock(side_effect=RuntimeError("fail!"))),
        ):
            fs.kwargs = {}
            mock_response = AsyncMock()
            mock_response.json = AsyncMock(return_value={})
            mock_response.headers = {}
            mock_session = MagicMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            fs.set_session.return_value = mock_session
            with pytest.raises(RuntimeError, match="fail!"):
                await fs._changes(url)
            mock_logger.log.assert_called()


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_single_page_with_new_pagination_logic() -> None:
    """Test ls method with single page (no pagination)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_token", MagicMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        # For detail=False, super().ls() returns a list of strings
        string_resources = [
            "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
        ]

        # For detail=True, super().ls() returns a list of dictionaries
        dict_resources = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            },
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
            },
        ]

        # Test detail=False, keep_protocol=False (default)
        mock_super_ls.return_value = string_resources
        result = fs.ls("some_url", detail=False)
        assert result == ["foo", "bar"]
        mock_super_ls.assert_called_once_with("some_url", detail=False)

        # Test detail=True, keep_protocol=False
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(dict_resources)
        result = fs.ls("some_url", detail=True)
        assert result == [
            {"name": "foo", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo"},
            {"name": "bar", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar"},
        ]

        # Test keep_protocol=True
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(dict_resources)
        result = fs.ls("some_url", detail=True, keep_protocol=True)
        assert result == dict_resources

        # Test detail=False, keep_protocol=True
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(string_resources)
        result = fs.ls("some_url", detail=False, keep_protocol=True)
        assert result == string_resources


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_multiple_pages_with_new_pagination_logic() -> None:
    """Test ls method with multiple pages (pagination enabled)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_token", MagicMock(side_effect=["token2", None])),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        dict_resources1 = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            }
        ]
        dict_resources2: list[dict[str, str]] = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
            }
        ]

        # Test detail=False pagination
        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_ls.side_effect = [string_resources1, string_resources2]
            result = fs.ls("some_url", detail=False)
            assert result == ["foo", "bar"]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test detail=True pagination
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_ls.side_effect = [copy.deepcopy(dict_resources1), copy.deepcopy(dict_resources2)]
            result = fs.ls("some_url", detail=True)
            assert result == [
                {"name": "foo", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo"},
                {"name": "bar", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar"},
            ]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test pagination with keep_protocol=True
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_ls.side_effect = [copy.deepcopy(dict_resources1), copy.deepcopy(dict_resources2)]
            result = fs.ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2
            assert mock_super_ls.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_pagination_with_existing_headers() -> None:
    """Test ls method pagination when kwargs already contains headers."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_ls.side_effect = [string_resources1, string_resources2]

            # Test with existing headers
            initial_headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
            result = fs.ls("some_url", detail=False, headers=initial_headers)
            assert result == ["foo", "bar"]

            # Verify headers were properly handled
            calls = mock_super_ls.call_args_list
            first_call_headers = calls[0][1]["headers"]
            second_call_headers = calls[1][1]["headers"]

            # First call should have original headers
            assert first_call_headers["Authorization"] == "Bearer token"
            assert first_call_headers["Custom-Header"] == "value"

            # Second call should have original headers plus pagination token
            assert second_call_headers["Authorization"] == "Bearer token"
            assert second_call_headers["Custom-Header"] == "value"
            assert second_call_headers["x-jpmc-next-token"] == "token2"

            # Original headers dict should not be modified
            assert "x-jpmc-next-token" not in initial_headers


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_pagination_three_pages() -> None:
    """Test ls method with three pages of results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]
        string_resources3 = ["https://fusion.jpmorgan.com/api/v1/catalogs/baz"]

        dict_resources1 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo", "identifier": "foo"}]
        dict_resources2 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar", "identifier": "bar"}]
        dict_resources3 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/baz", "identifier": "baz"}]

        # Test detail=False with three pages
        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_ls.side_effect = [string_resources1, string_resources2, string_resources3]
            result = fs.ls("some_url", detail=False)
            assert result == ["foo", "bar", "baz"]
            assert mock_super_ls.call_count == CALL_COUNT

            # Verify pagination tokens in headers
            calls = mock_super_ls.call_args_list
            assert "x-jpmc-next-token" not in calls[0][1].get("headers", {})
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"
            assert calls[2][1]["headers"]["x-jpmc-next-token"] == "token3"

        # Test detail=True with three pages
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_ls.side_effect = [
                copy.deepcopy(dict_resources1),
                copy.deepcopy(dict_resources2),
                copy.deepcopy(dict_resources3),
            ]
            result = fs.ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2 + dict_resources3
            assert mock_super_ls.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_pagination_empty_results() -> None:
    """Test ls method pagination when one of the pages returns empty results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2: list[str] = []
        string_resources3 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        dict_resources1 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo", "identifier": "foo"}]
        dict_resources2: list[dict[str, str]] = []
        dict_resources3 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar", "identifier": "bar"}]

        # Test detail=False with empty page
        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_ls.side_effect = [string_resources1, string_resources2, string_resources3]
            result = fs.ls("some_url", detail=False)
            assert result == ["foo", "bar"]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test detail=True with empty page
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_ls.side_effect = [dict_resources1, dict_resources2, dict_resources3]
            result = fs.ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2 + dict_resources3
            assert mock_super_ls.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_ls_kwargs_not_modified_during_pagination() -> None:
    """Test that original kwargs are not modified during pagination."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_ls.side_effect = [string_resources1, string_resources2]

            original_kwargs = {"custom_param": "value", "headers": {"Auth": "token"}}
            kwargs_copy = copy.deepcopy(original_kwargs)

            result = fs.ls("some_url", detail=False, **original_kwargs)
            assert result == ["foo", "bar"]

            # Verify original kwargs are unchanged
            assert original_kwargs == kwargs_copy
            assert "x-jpmc-next-token" not in original_kwargs.get("headers", {})

            # Verify keep_protocol was properly removed from kwargs copy
            calls = mock_super_ls.call_args_list
            for call in calls:
                assert "keep_protocol" not in call[1]


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_single_page_with_new_pagination_logic() -> None:
    """Test async _ls method with single page (no pagination)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_async_token", AsyncMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        # For detail=False, super()._ls() returns a list of strings
        string_resources = [
            "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
        ]

        # For detail=True, super()._ls() returns a list of dictionaries
        dict_resources = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            },
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
            },
        ]

        # Test detail=False, keep_protocol=False (default)
        mock_super_ls.return_value = string_resources
        result = await fs._ls("some_url", detail=False)
        assert result == ["foo", "bar"]
        mock_super_ls.assert_called_once_with("some_url", False)

        # Test detail=True, keep_protocol=False
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(dict_resources)
        result = await fs._ls("some_url", detail=True)
        assert result == [
            {"name": "foo", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo"},
            {"name": "bar", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar"},
        ]

        # Test keep_protocol=True
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(dict_resources)
        result = await fs._ls("some_url", detail=True, keep_protocol=True)
        assert result == dict_resources

        # Test detail=False, keep_protocol=True
        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(string_resources)
        result = await fs._ls("some_url", detail=False, keep_protocol=True)
        assert result == string_resources


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_multiple_pages_with_new_pagination_logic() -> None:
    """Test async _ls method with multiple pages (pagination enabled)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2: list[str] = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        dict_resources1 = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo",
            }
        ]
        dict_resources2: list[dict[str, str]] = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar",
            }
        ]

        # Test detail=False pagination
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_ls.side_effect = [string_resources1, string_resources2]
            result = await fs._ls("some_url", detail=False)
            assert result == ["foo", "bar"]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test detail=True pagination
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_ls.side_effect = [copy.deepcopy(dict_resources1), copy.deepcopy(dict_resources2)]
            result = await fs._ls("some_url", detail=True)
            assert result == [
                {"name": "foo", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/foo"},
                {"name": "bar", "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/bar"},
            ]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test pagination with keep_protocol=True
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_ls.side_effect = [copy.deepcopy(dict_resources1), copy.deepcopy(dict_resources2)]
            result = await fs._ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2
            assert mock_super_ls.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_pagination_with_existing_headers() -> None:
    """Test async _ls method pagination when kwargs already contains headers."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_ls.side_effect = [string_resources1, string_resources2]

            # Test with existing headers
            initial_headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
            result = await fs._ls("some_url", detail=False, headers=initial_headers)
            assert result == ["foo", "bar"]

            # Verify headers were properly handled
            calls = mock_super_ls.call_args_list
            first_call_headers = calls[0][1]["headers"]
            second_call_headers = calls[1][1]["headers"]

            # First call should have original headers
            assert first_call_headers["Authorization"] == "Bearer token"
            assert first_call_headers["Custom-Header"] == "value"

            # Second call should have original headers plus pagination token
            assert second_call_headers["Authorization"] == "Bearer token"
            assert second_call_headers["Custom-Header"] == "value"
            assert second_call_headers["x-jpmc-next-token"] == "token2"

            # Original headers dict should not be modified
            assert "x-jpmc-next-token" not in initial_headers


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_pagination_three_pages() -> None:
    """Test async _ls method with three pages of results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]
        string_resources3 = ["https://fusion.jpmorgan.com/api/v1/catalogs/baz"]

        dict_resources1 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo", "identifier": "foo"}]
        dict_resources2 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar", "identifier": "bar"}]
        dict_resources3 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/baz", "identifier": "baz"}]

        # Test detail=False with three pages
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_ls.side_effect = [string_resources1, string_resources2, string_resources3]
            result = await fs._ls("some_url", detail=False)
            assert result == ["foo", "bar", "baz"]
            assert mock_super_ls.call_count == CALL_COUNT

            # Verify pagination tokens in headers
            calls = mock_super_ls.call_args_list
            assert "x-jpmc-next-token" not in calls[0][1].get("headers", {})
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"
            assert calls[2][1]["headers"]["x-jpmc-next-token"] == "token3"

        # Test detail=True with three pages
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_ls.side_effect = [
                copy.deepcopy(dict_resources1),
                copy.deepcopy(dict_resources2),
                copy.deepcopy(dict_resources3),
            ]
            result = await fs._ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2 + dict_resources3
            assert mock_super_ls.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_pagination_empty_results() -> None:
    """Test async _ls method pagination when one of the pages returns empty results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2: list[str] = []
        string_resources3 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        dict_resources1 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/foo", "identifier": "foo"}]
        dict_resources2: list[dict[str, str]] = []
        dict_resources3 = [{"name": "https://fusion.jpmorgan.com/api/v1/catalogs/bar", "identifier": "bar"}]

        # Test detail=False with empty page
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_ls.side_effect = [string_resources1, string_resources2, string_resources3]
            result = await fs._ls("some_url", detail=False)
            assert result == ["foo", "bar"]
            assert mock_super_ls.call_count == CALL_COUNT

        # Test detail=True with empty page
        mock_super_ls.reset_mock()
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_ls.side_effect = [dict_resources1, dict_resources2, dict_resources3]
            result = await fs._ls("some_url", detail=True, keep_protocol=True)
            assert result == dict_resources1 + dict_resources2 + dict_resources3
            assert mock_super_ls.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_kwargs_not_modified_during_pagination() -> None:
    """Test that original kwargs are not modified during async _ls pagination."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources1 = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        string_resources2 = ["https://fusion.jpmorgan.com/api/v1/catalogs/bar"]

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_ls.side_effect = [string_resources1, string_resources2]

            original_kwargs = {"custom_param": "value", "headers": {"Auth": "token"}}
            kwargs_copy = copy.deepcopy(original_kwargs)

            result = await fs._ls("some_url", detail=False, **original_kwargs)
            assert result == ["foo", "bar"]

            # Verify original kwargs are unchanged
            assert original_kwargs == kwargs_copy
            assert "x-jpmc-next-token" not in original_kwargs.get("headers", {})

            # Verify keep_protocol was properly removed from kwargs copy
            calls = mock_super_ls.call_args_list
            for call in calls:
                assert "keep_protocol" not in call[1]


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_startup_and_url_decoration() -> None:
    """Test that async _ls properly calls startup and url decoration methods."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()) as mock_startup,
        patch.object(fs, "_decorate_url_a", AsyncMock(return_value="decorated_url")) as mock_decorate,
        patch.object(fs, "_get_next_async_token", AsyncMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        string_resources = ["https://fusion.jpmorgan.com/api/v1/catalogs/foo"]
        mock_super_ls.return_value = string_resources

        result = await fs._ls("some_url", detail=False)

        # Verify startup was called
        mock_startup.assert_called_once()

        # Verify URL decoration was called with the original URL
        mock_decorate.assert_called_once_with("some_url")

        # Verify super()._ls was called with decorated URL
        mock_super_ls.assert_called_once_with("decorated_url", False)

        assert result == ["foo"]


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_ls_protocol_stripping_variations() -> None:
    """Test async _ls method with various protocol stripping scenarios."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url_a", AsyncMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_async_token", AsyncMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._ls") as mock_super_ls,
    ):
        # Test with complex URLs
        complex_resources = [
            {
                "name": "https://fusion.jpmorgan.com/api/v1/catalogs/test/dataset/sample",
                "identifier": "https://fusion.jpmorgan.com/api/v1/catalogs/test/dataset/sample",
                "type": "directory",
            }
        ]

        mock_super_ls.return_value = copy.deepcopy(complex_resources)
        result = await fs._ls("some_url", detail=True, keep_protocol=False)
        assert result[0]["name"] == "test/dataset/sample"
        assert result[0]["identifier"] == "https://fusion.jpmorgan.com/api/v1/catalogs/test/dataset/sample"

        # Test with URLs that don't contain the expected prefix
        unusual_resources = [
            {
                "name": "https://example.com/different/path",
                "identifier": "https://example.com/different/path",
                "type": "file",
            }
        ]

        mock_super_ls.reset_mock()
        mock_super_ls.return_value = copy.deepcopy(unusual_resources)
        result = await fs._ls("some_url", detail=True, keep_protocol=False)
        # Should return the original name if split doesn't find the expected pattern
        assert result[0]["name"] == "https://example.com/different/path"


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_single_page_no_pagination() -> None:
    """Test cat method with single page (no pagination)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_token", MagicMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        # Test with bytes content
        test_content = b"test file content"
        mock_super_cat.return_value = test_content

        result = fs.cat("some_url")
        assert result == test_content
        mock_super_cat.assert_called_once_with("some_url", start=None, end=None)

        # Test with string content (should be returned as-is)
        mock_super_cat.reset_mock()
        test_string = "test string content"
        mock_super_cat.return_value = test_string

        result = fs.cat("some_url")
        assert result == test_string
        mock_super_cat.assert_called_once_with("some_url", start=None, end=None)


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_multiple_pages_with_pagination() -> None:
    """Test cat method with multiple pages (pagination enabled)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"first page content"
        content2 = b"second page content"

        # Mock pagination: first call returns token, second call returns None
        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            result = fs.cat("some_url")
            expected = content1 + content2
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT

            # Check that headers were properly set for pagination
            calls = mock_super_cat.call_args_list
            assert calls[0][1].get("headers", {}) == {}  # First call has no pagination header
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"  # Second call has token


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_with_existing_headers() -> None:
    """Test cat method pagination when kwargs already contains headers."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            # Test with existing headers
            initial_headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
            result = fs.cat("some_url", headers=initial_headers)

            expected = content1 + content2
            assert result == expected

            # Verify headers were properly handled
            calls = mock_super_cat.call_args_list
            first_call_headers = calls[0][1].get("headers", {})
            second_call_headers = calls[1][1]["headers"]

            # First call should have original headers
            assert first_call_headers["Authorization"] == "Bearer token"
            assert first_call_headers["Custom-Header"] == "value"

            # Second call should have original headers plus pagination token
            assert second_call_headers["Authorization"] == "Bearer token"
            assert second_call_headers["Custom-Header"] == "value"
            assert second_call_headers["x-jpmc-next-token"] == "token2"

            # Original headers dict should not be modified
            assert "x-jpmc-next-token" not in initial_headers


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_three_pages() -> None:
    """Test cat method with three pages of results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"
        content3 = b"page3"

        # Mock three pages: token2 -> token3 -> None
        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_cat.side_effect = [content1, content2, content3]

            result = fs.cat("some_url")
            expected = content1 + content2 + content3
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT

            # Verify pagination tokens in headers
            calls = mock_super_cat.call_args_list
            assert "x-jpmc-next-token" not in calls[0][1].get("headers", {})
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"
            assert calls[2][1]["headers"]["x-jpmc-next-token"] == "token3"


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_with_start_end_parameters() -> None:
    """Test cat method pagination with start and end parameters."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_START = 10
    CALL_END = 100

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"partial1"
        content2 = b"partial2"

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            result = fs.cat("some_url", start=10, end=100)
            expected = content1 + content2
            assert result == expected

            # Verify start and end parameters are passed through
            calls = mock_super_cat.call_args_list
            for call in calls:
                assert call[1]["start"] == CALL_START
                assert call[1]["end"] == CALL_END


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_mixed_content_types() -> None:
    """Test cat method pagination with mixed content types."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        # First page is bytes, second page is string (should be ignored in concatenation)
        content1 = b"bytes content"
        content2 = "string content"  # This should not be concatenated

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            result = fs.cat("some_url")
            # Only bytes content should be concatenated
            assert result == content1
            assert mock_super_cat.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_first_page_non_bytes() -> None:
    """Test cat method pagination when first page is not bytes."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        # First page is string, second page is bytes
        content1 = "string content"
        content2 = b"bytes content"

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            result = fs.cat("some_url")
            # Should start with empty bytes and add only bytes content
            assert result == content2
            assert mock_super_cat.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_empty_pages() -> None:
    """Test cat method pagination when some pages return empty content."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b""  # Empty page
        content3 = b"page3"

        with patch.object(fs, "_get_next_token", side_effect=["token2", "token3", None]):
            mock_super_cat.side_effect = [content1, content2, content3]

            result = fs.cat("some_url")
            expected = content1 + content2 + content3
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_kwargs_not_modified_during_pagination() -> None:
    """Test that original kwargs are not modified during cat pagination."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            original_kwargs = {"custom_param": "value", "headers": {"Auth": "token"}}
            kwargs_copy = copy.deepcopy(original_kwargs)

            result = fs.cat("some_url", custom_param="value", headers={"Auth": "token"})
            expected = content1 + content2
            assert result == expected

            # Verify original kwargs are unchanged
            assert original_kwargs == kwargs_copy
            assert "x-jpmc-next-token" not in original_kwargs.get("headers", {})


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_url_decoration() -> None:
    """Test that cat properly decorates URLs."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated_url")) as mock_decorate,
        patch.object(fs, "_get_next_token", MagicMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        test_content = b"test content"
        mock_super_cat.return_value = test_content

        result = fs.cat("original_url")

        # Verify URL decoration was called
        mock_decorate.assert_called_once_with("original_url")

        # Verify super().cat was called with decorated URL
        mock_super_cat.assert_called_once_with("decorated_url", start=None, end=None)

        assert result == test_content


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_cat_pagination_with_additional_kwargs() -> None:
    """Test cat method pagination preserves additional kwargs."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_START = 0
    CALL_END = 100

    with (
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem.cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_token", side_effect=["token2", None]):
            mock_super_cat.side_effect = [content1, content2]

            # Test with additional kwargs
            result = fs.cat("some_url", start=0, end=100, custom_param="value")
            expected = content1 + content2
            assert result == expected

            # Verify all kwargs are preserved
            calls = mock_super_cat.call_args_list
            for call in calls:
                assert call[1]["start"] == CALL_START
                assert call[1]["end"] == CALL_END
                assert call[1]["custom_param"] == "value"

            # Second call should have pagination token in headers
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_single_page_no_pagination() -> None:
    """Test async _cat method with single page (no pagination)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch.object(fs, "_get_next_async_token", AsyncMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        # Test with bytes content
        test_content = b"test file content"
        mock_super_cat.return_value = test_content

        result = await fs._cat("some_url")
        assert result == test_content
        mock_super_cat.assert_called_once_with("some_url", start=None, end=None)

        # Test with string content (should be returned as-is)
        mock_super_cat.reset_mock()
        test_string = "test string content"
        mock_super_cat.return_value = test_string

        result = await fs._cat("some_url")
        assert result == test_string
        mock_super_cat.assert_called_once_with("some_url", start=None, end=None)


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_multiple_pages_with_pagination() -> None:
    """Test async _cat method with multiple pages (pagination enabled)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"first page content"
        content2 = b"second page content"

        # Mock pagination: first call returns token, second call returns None
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            result = await fs._cat("some_url")
            expected = content1 + content2
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT

            # Check that headers were properly set for pagination
            calls = mock_super_cat.call_args_list
            assert calls[0][1].get("headers", {}) == {}  # First call has no pagination header
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"  # Second call has token


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_with_existing_headers() -> None:
    """Test async _cat method pagination when kwargs already contains headers."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            # Test with existing headers
            initial_headers = {"Authorization": "Bearer token", "Custom-Header": "value"}
            result = await fs._cat("some_url", headers=initial_headers)

            expected = content1 + content2
            assert result == expected

            # Verify headers were properly handled
            calls = mock_super_cat.call_args_list
            first_call_headers = calls[0][1].get("headers", {})
            second_call_headers = calls[1][1]["headers"]

            # First call should have original headers
            assert first_call_headers["Authorization"] == "Bearer token"
            assert first_call_headers["Custom-Header"] == "value"

            # Second call should have original headers plus pagination token
            assert second_call_headers["Authorization"] == "Bearer token"
            assert second_call_headers["Custom-Header"] == "value"
            assert second_call_headers["x-jpmc-next-token"] == "token2"

            # Original headers dict should not be modified
            assert "x-jpmc-next-token" not in initial_headers


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_three_pages() -> None:
    """Test async _cat method with three pages of results."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"
        content3 = b"page3"

        # Mock three pages: token2 -> token3 -> None
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_cat.side_effect = [content1, content2, content3]

            result = await fs._cat("some_url")
            expected = content1 + content2 + content3
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT

            # Verify pagination tokens in headers
            calls = mock_super_cat.call_args_list
            assert "x-jpmc-next-token" not in calls[0][1].get("headers", {})
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"
            assert calls[2][1]["headers"]["x-jpmc-next-token"] == "token3"


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_with_start_end_parameters() -> None:
    """Test async _cat method pagination with start and end parameters."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_START = 10
    CALL_END = 100

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"partial1"
        content2 = b"partial2"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            result = await fs._cat("some_url", start=10, end=100)
            expected = content1 + content2
            assert result == expected

            # Verify start and end parameters are passed through
            calls = mock_super_cat.call_args_list
            for call in calls:
                assert call[1]["start"] == CALL_START
                assert call[1]["end"] == CALL_END


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_mixed_content_types() -> None:
    """Test async _cat method pagination with mixed content types."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        # First page is bytes, second page is string (should be ignored in concatenation)
        content1 = b"bytes content"
        content2 = "string content"  # This should not be concatenated

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            result = await fs._cat("some_url")
            # Only bytes content should be concatenated
            assert result == content1
            assert mock_super_cat.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_first_page_non_bytes() -> None:
    """Test async _cat method pagination when first page is not bytes."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        # First page is string, second page is bytes
        content1 = "string content"
        content2 = b"bytes content"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            result = await fs._cat("some_url")
            # Should start with empty bytes and add only bytes content
            assert result == content2
            assert mock_super_cat.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_empty_pages() -> None:
    """Test async _cat method pagination when some pages return empty content."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 3

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b""  # Empty page
        content3 = b"page3"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", "token3", None])):
            mock_super_cat.side_effect = [content1, content2, content3]

            result = await fs._cat("some_url")
            expected = content1 + content2 + content3
            assert result == expected
            assert mock_super_cat.call_count == CALL_COUNT


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_kwargs_not_modified_during_pagination() -> None:
    """Test that original kwargs are not modified during async _cat pagination."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            original_kwargs: dict[str, Any] = {"custom_param": "value", "headers": {"Auth": "token"}}
            kwargs_copy = copy.deepcopy(original_kwargs)

            result = await fs._cat("some_url", **original_kwargs)
            expected = content1 + content2
            assert result == expected

            # Verify original kwargs are unchanged
            assert original_kwargs == kwargs_copy
            assert "x-jpmc-next-token" not in original_kwargs.get("headers", {})


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_startup_and_url_decoration() -> None:
    """Test that async _cat properly calls startup and url decoration methods."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_async_startup", AsyncMock()) as mock_startup,
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated_url")) as mock_decorate,
        patch.object(fs, "_get_next_async_token", AsyncMock(return_value=None)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        test_content = b"test content"
        mock_super_cat.return_value = test_content

        result = await fs._cat("original_url")

        # Verify startup was called
        mock_startup.assert_called_once()

        # Verify URL decoration was called with the original URL
        mock_decorate.assert_called_once_with("original_url")

        # Verify super()._cat was called with decorated URL
        mock_super_cat.assert_called_once_with("decorated_url", start=None, end=None)

        assert result == test_content


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_with_additional_kwargs() -> None:
    """Test async _cat method pagination preserves additional kwargs."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_START = 0
    CALL_END = 100

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"
        content2 = b"page2"

        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, content2]

            # Test with additional kwargs
            result = await fs._cat("some_url", start=0, end=100, custom_param="value")
            expected = content1 + content2
            assert result == expected

            # Verify all kwargs are preserved
            calls = mock_super_cat.call_args_list
            for call in calls:
                assert call[1]["start"] == CALL_START
                assert call[1]["end"] == CALL_END
                assert call[1]["custom_param"] == "value"

            # Second call should have pagination token in headers
            assert calls[1][1]["headers"]["x-jpmc-next-token"] == "token2"


@pytest.mark.asyncio
@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
async def test_async_cat_pagination_error_handling() -> None:
    """Test async _cat method error handling during pagination."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}
    CALL_COUNT = 2

    with (
        patch.object(fs, "_async_startup", AsyncMock()),
        patch.object(fs, "_decorate_url", MagicMock(side_effect=lambda x: x)),
        patch("fusion.fusion_filesystem.HTTPFileSystem._cat") as mock_super_cat,
    ):
        content1 = b"page1"

        # Mock an error on the second page
        with patch.object(fs, "_get_next_async_token", AsyncMock(side_effect=["token2", None])):
            mock_super_cat.side_effect = [content1, Exception("Network error")]

            with pytest.raises(Exception, match="Network error"):
                await fs._cat("some_url")

            # Verify first call succeeded before error
            assert mock_super_cat.call_count == CALL_COUNT


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_info_file_type() -> None:
    """Test info method when path points to a file."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated_path")) as mock_decorate,
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        # Mock file response
        file_response = [{"name": "test_file.csv", "type": "file", "size": 1024, "modified": "2023-01-01T00:00:00Z"}]
        mock_super_ls.return_value = file_response

        result = fs.info("some/path/file.csv")

        # Verify URL decoration was called
        mock_decorate.assert_called_once_with("some/path/file.csv")

        # Verify super().ls was called with keep_protocol=True
        mock_super_ls.assert_called_once_with("decorated_path", detail=True, keep_protocol=True)

        # For file type, should return the ls result directly
        assert result == file_response

        # Should not call super().info since it's a file
        assert mock_super_ls.call_count == 1


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_info_distributions_path() -> None:
    """Test info method for distributions path that should return first element."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated/path/distributions/dist_item")),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
        patch("fusion.fusion_filesystem.HTTPFileSystem.info") as mock_super_info,
    ):
        # Mock directory response from ls (not a file)
        ls_response = [{"name": "dist_item", "type": "directory"}]
        mock_super_ls.return_value = ls_response

        # Mock info response as a list (simulating distributions behavior)
        info_response = [
            {"name": "distribution_item", "type": "file", "size": 2048, "modified": "2023-01-01T00:00:00Z"}
        ]
        mock_super_info.return_value = info_response

        result = fs.info("some/path/distributions/dist_item")

        # For distributions path, should return first element of the response
        assert result == info_response[0]


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_info_non_dataset_directory() -> None:
    """Test info method for directory that is not a dataset (no changes)."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated/path/catalogs/regular_dir")),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
        patch("fusion.fusion_filesystem.HTTPFileSystem.info") as mock_super_info,
        patch("fusion.fusion_filesystem.sync") as mock_sync,
    ):
        # Mock directory response from ls (not a file)
        ls_response = [{"name": "regular_dir", "type": "directory"}]
        mock_super_ls.return_value = ls_response

        info_response = {"name": "regular_dir", "type": "directory", "size": None}
        mock_super_info.return_value = info_response

        result = fs.info("some/path/catalogs/regular_dir")

        # Should not call sync for changes since it's not a dataset
        mock_sync.assert_not_called()

        # Should return info response without changes
        assert result == info_response


@patch.object(FusionHTTPFileSystem, "__init__", lambda *_, **__: None)
def test_info_error_handling() -> None:
    """Test info method error handling."""
    fs = FusionHTTPFileSystem()
    fs.client_kwargs = {"root_url": "https://fusion.jpmorgan.com/api/v1/"}
    fs.kwargs = {}

    with (
        patch.object(fs, "_decorate_url", MagicMock(return_value="decorated_path")),
        patch("fusion.fusion_filesystem.HTTPFileSystem.ls") as mock_super_ls,
    ):
        # Mock ls to raise an exception
        mock_super_ls.side_effect = FileNotFoundError("Path not found")

        with pytest.raises(FileNotFoundError, match="Path not found"):
            fs.info("nonexistent/path")
