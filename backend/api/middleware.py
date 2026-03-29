"""HTTP request-tracing middleware for the Lersha Credit Scoring API.

Assigns a unique ``X-Request-ID`` to every incoming HTTP request.  If the
client supplies the header, it is echoed back; otherwise a new UUID v4 is
generated.  The ID is stored on ``request.state.request_id`` so that route
handlers and background tasks can reference it for log correlation.
"""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns/forwards an X-Request-ID header per request.

    Order of precedence:
      1. If the inbound request contains ``X-Request-ID``, that value is used.
      2. Otherwise a new ``uuid.uuid4()`` string is generated.

    The identifier is attached to ``request.state.request_id`` for use within
    the request handler, and echoed in the outbound ``X-Request-ID`` response
    header so clients can correlate responses with their own trace IDs.
    """

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        """Process the request and inject the trace ID into state and response.

        Args:
            request: Incoming Starlette/FastAPI request.
            call_next: Next middleware or route handler in the chain.

        Returns:
            Response with ``X-Request-ID`` header set.
        """
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
