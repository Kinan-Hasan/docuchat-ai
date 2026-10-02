"""API key authentication. Simple by design (header -> allow-list), but wired
the way a real service would be, so it's a one-line swap to OAuth2/JWT later."""
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str | None = Depends(_api_key_header)) -> str:
    settings = get_settings()
    if not api_key or api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key. Pass it in the X-API-Key header.",
        )
    return api_key
