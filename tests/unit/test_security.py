import pytest
from fastapi import HTTPException

from app.core.security import verify_api_key


def test_verify_api_key_accepts_valid_key():
    assert verify_api_key(api_key="devkey123") == "devkey123"


def test_verify_api_key_rejects_missing_key():
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key=None)
    assert exc_info.value.status_code == 401


def test_verify_api_key_rejects_bad_key():
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 401
