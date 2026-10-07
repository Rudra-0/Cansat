"""Access control for operator-only endpoints."""

import hmac
import os
from typing import Optional

from fastapi import Header, HTTPException, Query, Request

LOCAL_CLIENT_HOSTS = {"127.0.0.1", "::1", "localhost"}


def require_operator(
    request: Request,
    x_api_key: Optional[str] = Header(None),
    api_key: Optional[str] = Query(None, description="Alternative to the X-API-Key header"),
) -> None:
    """Guard for destructive / hardware-control endpoints.

    With CANSAT_API_KEY set, callers must present it (X-API-Key header or
    ?api_key=). Without it, only clients on this machine are allowed, so a
    laptop ground station works out of the box while a deployed server is
    locked down until a key is configured.
    """
    expected = os.environ.get("CANSAT_API_KEY")
    if expected:
        supplied = x_api_key or api_key or ""
        if hmac.compare_digest(supplied.encode(), expected.encode()):
            return
        raise HTTPException(status_code=401, detail="Valid X-API-Key required",
                            headers={"WWW-Authenticate": "ApiKey"})
    client_host = request.client.host if request.client else ""
    if client_host in LOCAL_CLIENT_HOSTS:
        return
    raise HTTPException(status_code=403,
                        detail="Operator endpoints are restricted to localhost unless CANSAT_API_KEY is set")
