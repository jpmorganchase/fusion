import io
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

import fsspec
import pytest
from aiohttp import ClientResponse

from fusion._fusion import FusionCredentials
from fusion.exceptions import APIResponseError
from fusion.fusion_filesystem import FusionFile, FusionHTTPFileSystem


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
    mock_set_session: AsyncMock,  # noqa: ARG001
    mock_get: MagicMock,  # noqa: ARG001
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


@pytest.mark.asyncio
async def test__async_fetch_range_with_headers_success() -> None:
    # Arrange
    mock_session = mock.AsyncMock()
    mock_fs = mock.Mock()
    mock_fs.encode_url.return_value = "http://test-url"
    mock_response = mock.AsyncMock()
    mock_response.__aenter__.return_value = mock_response
    mock_response.raise_for_status.return_value = None
    mock_response.read.return_value = b"test-bytes"
    mock_response.headers = {"Content-Length": "10", "Range": "bytes=0-9"}
    mock_session.get.return_value = mock_response

    fusion_file = FusionFile(
        url="http://test-url", session=mock_session, fs=mock_fs, kwargs={"headers": {"Authorization": "Bearer token"}}
    )

    # Act
    out, headers = await fusion_file._async_fetch_range_with_headers(0, 10)

    # Assert
    assert out == b"test-bytes"
    assert headers == {"Content-Length": "10", "Range": "bytes=0-9"}
    mock_session.get.assert_awaited_once()
    mock_fs.encode_url.assert_called_once_with("http://test-url")
    assert mock_response.raise_for_status.called


class DummyResponse:
    def __init__(self, content: bytes, headers: dict[str, Any]) -> None:
        self.content = content
        self.headers = headers
        self.status_code = 206

    def raise_for_status(self) -> None:
        pass

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        pass


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response

    def get(self, _url: str, _headers: Optional[dict[str, Any]] = None, **_kwargs: Any) -> DummyResponse:
        return self.response


class DummyFS:
    def encode_url(self, url: str) -> str:
        return url


@pytest.fixture
def fusion_file() -> FusionFile:
    file = FusionFile.__new__(FusionFile)
    file.url = "http://test-url/file"
    file.fs = DummyFS()
    file.kwargs = {"headers": {"Authorization": "Bearer token"}}
    return file


def test_fetch_range_with_headers_success(fusion_file: FusionFile) -> None:
    content = b"test-bytes"
    headers: dict[str, Any] = {"Content-Length": "10", "Range": "bytes=0-9"}
    headers = {"Content-Length": "10", "Range": "bytes=0-9"}
    dummy_response = DummyResponse(content, headers)
    fusion_file.session = DummySession(dummy_response)

    out, out_headers = fusion_file._fetch_range_with_headers(0, 10)
    assert out == content
    assert out_headers == headers


def test_fetch_range_with_headers_sets_range_header(fusion_file: FusionFile) -> None:
    content = b"abc"
    headers: dict[str, Any] = {"Content-Length": "3"}
    headers = {"Content-Length": "3"}
    dummy_response = DummyResponse(content, headers)
    fusion_file.session = DummySession(dummy_response)

    with mock.patch.object(fusion_file.session, "get", wraps=fusion_file.session.get) as mock_get:
        fusion_file._fetch_range_with_headers(5, 8)
        args, kwargs = mock_get.call_args
        assert kwargs["headers"]["Range"] == "bytes=5-7"


def test_fetch_range_with_headers_raises_for_status_called(fusion_file: FusionFile) -> None:
    content = b"xyz"
    headers: dict[str, Any] = {}
    headers = {}
    dummy_response = DummyResponse(content, headers)
    with mock.patch.object(dummy_response, "raise_for_status", wraps=dummy_response.raise_for_status) as mock_raise:
        fusion_file.session = DummySession(dummy_response)
        fusion_file._fetch_range_with_headers(0, 3)
        mock_raise.assert_called_once()


@pytest.fixture
def fusion_http_fs() -> FusionHTTPFileSystem:
    fs = FusionHTTPFileSystem.__new__(FusionHTTPFileSystem)
    fs.sync_session = mock.Mock()
    object.__setattr__(fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    fs._merge_all_data = FusionHTTPFileSystem._merge_all_data  # type: ignore
    object.__setattr__(fs, "_session", mock.Mock())
    object.__setattr__(fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fs, "_info", mock.AsyncMock())
    return fs


def test_cat_single_page(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    with mock.patch.object(fusion_http_fs, "info", mock.Mock(return_value={"size": 10})):
        fusion_file_mock = mock.Mock(spec=FusionFile)
        fusion_file_mock._fetch_range_with_headers = mock.Mock()
        fusion_file_mock._fetch_range_with_headers.return_value = (
            b'{"resources": ["item1", "item2"]}',
            {"x-jpmc-next-token": None},
        )
        monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

        result = fusion_http_fs.cat("test-url")
        assert b"item1" in result
        assert b"item2" in result
        fusion_file_mock._fetch_range_with_headers.assert_called_once()


def test_cat_multiple_pages(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    with mock.patch.object(fusion_http_fs, "info", mock.Mock(return_value={"size": 10})):
        fusion_file_mock = mock.Mock(spec=FusionFile)
        fusion_file_mock._fetch_range_with_headers = mock.Mock()
        fusion_file_mock._fetch_range_with_headers.side_effect = [
            (b'{"resources": ["item1"]}', {"x-jpmc-next-token": "token123"}),
            (b'{"resources": ["item2"]}', {"x-jpmc-next-token": None}),
        ]
        monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

        result = fusion_http_fs.cat("test-url")
        assert b"item1" in result
        assert b"item2" in result
        CALLS_EXPECTED = 2
        assert fusion_file_mock._fetch_range_with_headers.call_count == CALLS_EXPECTED


def test_cat_empty(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    with mock.patch.object(fusion_http_fs, "info", mock.Mock(return_value={"size": 10})):
        fusion_file_mock = mock.Mock(spec=FusionFile)
        fusion_file_mock._fetch_range_with_headers = mock.Mock()
        fusion_file_mock._fetch_range_with_headers.return_value = (b'{"resources": []}', {"x-jpmc-next-token": None})
        monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

        result = fusion_http_fs.cat("test-url")
        assert b"resources" in result
        assert b"item" not in result


def test_cat_handles_list_info(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    with mock.patch.object(fusion_http_fs, "info", mock.Mock(return_value=[{"size": 20}])):
        fusion_file_mock = mock.Mock(spec=FusionFile)
        fusion_file_mock._fetch_range_with_headers = mock.Mock()
        fusion_file_mock._fetch_range_with_headers.return_value = (
            b'{"resources": ["itemA"]}',
            {"x-jpmc-next-token": None},
        )
        monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

        result = fusion_http_fs.cat("test-url")
        assert b"itemA" in result


@pytest.mark.asyncio
async def test__cat_single_page(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    object.__setattr__(fusion_http_fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fusion_http_fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    object.__setattr__(fusion_http_fs, "_info", mock.AsyncMock(return_value={"size": 10}))

    fusion_file_mock = mock.Mock(spec=FusionFile)
    fusion_file_mock._async_fetch_range_with_headers = mock.AsyncMock(
        return_value=(b'{"resources": ["item1", "item2"]}', {"x-jpmc-next-token": None})
    )
    monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

    result = await fusion_http_fs._cat("test-url")
    assert b"item1" in result
    assert b"item2" in result
    fusion_file_mock._async_fetch_range_with_headers.assert_awaited_once()


@pytest.mark.asyncio
async def test__cat_multiple_pages(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    object.__setattr__(fusion_http_fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fusion_http_fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    object.__setattr__(fusion_http_fs, "_info", mock.AsyncMock(return_value={"size": 10}))

    fusion_file_mock = mock.Mock(spec=FusionFile)
    fusion_file_mock._async_fetch_range_with_headers = mock.AsyncMock(
        side_effect=[
            (b'{"resources": ["item1"]}', {"x-jpmc-next-token": "token123"}),
            (b'{"resources": ["item2"]}', {"x-jpmc-next-token": None}),
        ]
    )
    monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

    result = await fusion_http_fs._cat("test-url")
    assert b"item1" in result
    assert b"item2" in result
    CALLS_EXPECTED = 2
    assert fusion_file_mock._async_fetch_range_with_headers.await_count == CALLS_EXPECTED


@pytest.mark.asyncio
async def test__cat_empty(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    object.__setattr__(fusion_http_fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fusion_http_fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    object.__setattr__(fusion_http_fs, "_info", mock.AsyncMock(return_value={"size": 10}))

    fusion_file_mock = mock.Mock(spec=FusionFile)
    fusion_file_mock._async_fetch_range_with_headers = mock.AsyncMock(
        return_value=(b'{"resources": []}', {"x-jpmc-next-token": None})
    )
    monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

    result = await fusion_http_fs._cat("test-url")
    assert b"resources" in result
    assert b"item" not in result


@pytest.mark.asyncio
async def test__cat_handles_list_info(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    object.__setattr__(fusion_http_fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fusion_http_fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    object.__setattr__(fusion_http_fs, "_info", mock.AsyncMock(return_value=[{"size": 20}]))

    fusion_file_mock = mock.Mock(spec=FusionFile)
    fusion_file_mock._async_fetch_range_with_headers = mock.AsyncMock(
        return_value=(b'{"resources": ["itemA"]}', {"x-jpmc-next-token": None})
    )
    monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

    result = await fusion_http_fs._cat("test-url")
    assert b"itemA" in result


@pytest.mark.asyncio
async def test__cat_handles_no_size(monkeypatch: pytest.MonkeyPatch, fusion_http_fs: FusionHTTPFileSystem) -> None:
    object.__setattr__(fusion_http_fs, "_async_startup", mock.AsyncMock())
    object.__setattr__(fusion_http_fs, "_decorate_url", mock.Mock(side_effect=lambda x: x))
    object.__setattr__(fusion_http_fs, "_info", mock.AsyncMock(return_value={}))

    fusion_file_mock = mock.Mock(spec=FusionFile)
    fusion_file_mock._async_fetch_range_with_headers = mock.AsyncMock(
        return_value=(b'{"resources": ["itemX"]}', {"x-jpmc-next-token": None})
    )
    monkeypatch.setattr("fusion.fusion_filesystem.FusionFile", lambda *_args, **_kwargs: fusion_file_mock)

    result = await fusion_http_fs._cat("test-url")
    assert b"itemX" in result
