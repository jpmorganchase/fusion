from __future__ import annotations

from fusion.exceptions import APIResponseError, CredentialError


class DummyError(Exception):
    def __init__(self, message: str, *, code: int) -> None:
        super().__init__(message)
        self.code = code


class TestExceptionWrappers:
    def test_api_response_error_includes_message_and_copies_attributes(self) -> None:
        status_code = 503
        original = DummyError("boom", code=status_code)

        err = APIResponseError(original, message="request failed", status_code=status_code)

        assert str(err) == "APIResponseError: Status 503, request failed, Error: boom"
        assert err.original_exception is original
        assert err.status_code == status_code
        assert getattr(err, "code", None) == status_code

    def test_credential_error_without_prefix_message_uses_default_format(self) -> None:
        status_code = 400
        original = DummyError("bad creds", code=status_code)

        err = CredentialError(original, status_code=status_code)

        assert str(err) == "APIResponseError: Status 400, Error: bad creds"
        assert err.original_exception is original
        assert err.status_code == status_code
        assert getattr(err, "code", None) == status_code
