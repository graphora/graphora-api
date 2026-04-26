"""FastAPI authentication dependencies backed by Clerk JWTs."""

from functools import lru_cache
from typing import Optional
import logging
import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from graphora_server.auth.models import AuthContext
from graphora_server.config import settings

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)

# Track if we've already logged auth warnings to avoid spam
_auth_warnings_logged = False


class AuthConfigError(RuntimeError):
    """Raised when the authentication environment is misconfigured."""


def _is_production() -> bool:
    """Check if we're running in production mode."""
    env = os.getenv("ENVIRONMENT", "").lower()
    return env in ("production", "prod")


@lru_cache(maxsize=1)
def _get_jwk_client() -> PyJWKClient:
    """Create a cached PyJWKClient instance."""
    jwks_url = settings.CLERK_JWKS_URL
    if not jwks_url:
        raise AuthConfigError("CLERK_JWKS_URL is not configured")
    return PyJWKClient(jwks_url)


def _decode_token(token: str) -> dict:
    """Decode and validate a Clerk-issued JWT."""
    global _auth_warnings_logged

    jwk_client = _get_jwk_client()
    signing_key = jwk_client.get_signing_key_from_jwt(token)

    algorithms = ["RS256", "ES256"]

    options = {"verify_signature": True}
    issuer: Optional[str] = settings.CLERK_ISSUER or None
    audience: Optional[str] = settings.CLERK_AUDIENCE or None

    # Check for missing issuer/audience configuration
    is_production = _is_production()

    if issuer is None:
        if is_production:
            raise AuthConfigError(
                "CLERK_ISSUER must be configured in production environment. "
                "JWT issuer validation is required for security."
            )
        if not _auth_warnings_logged:
            logger.warning(
                "CLERK_ISSUER is not configured. JWT issuer validation is disabled. "
                "This is a security risk and should not be used in production."
            )
        options["verify_iss"] = False

    if audience is None:
        # Audience is optional - Clerk's default session tokens don't include 'aud'
        # Only warn once, don't block in production since issuer validation is sufficient
        if not _auth_warnings_logged:
            logger.warning(
                "CLERK_AUDIENCE is not configured. JWT audience validation is disabled. "
                "Consider setting CLERK_AUDIENCE for additional security if using custom JWT templates."
            )
        options["verify_aud"] = False

    # Only log warnings once to avoid log spam
    if not _auth_warnings_logged and (issuer is None or audience is None):
        _auth_warnings_logged = True

    decoded = jwt.decode(
        token,
        signing_key.key,
        algorithms=algorithms,
        audience=audience,
        issuer=issuer,
        options=options,
    )
    return decoded


def _get_bypass_auth_context() -> AuthContext:
    """Return a mock AuthContext for local development bypass mode."""
    return AuthContext(
        user_id=settings.AUTH_BYPASS_USER_ID,
        session_id="local-dev-session",
        token="bypass-token",
        claims={
            "sub": settings.AUTH_BYPASS_USER_ID,
            "email": settings.AUTH_BYPASS_EMAIL,
            "iss": "local-dev",
            "aud": "graphora-local",
        },
    )


def get_current_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> AuthContext:
    """Validate the bearer token and return the authenticated context.

    If AUTH_BYPASS_ENABLED is set, returns a mock context for local development.
    """
    # Auth bypass for local development
    if settings.AUTH_BYPASS_ENABLED:
        if _is_production():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="AUTH_BYPASS_ENABLED cannot be used in production",
            )
        return _get_bypass_auth_context()

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    token = credentials.credentials
    try:
        claims = _decode_token(token)
    except AuthConfigError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        ) from exc
    except jwt.InvalidAudienceError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid audience",
        ) from exc
    except jwt.InvalidIssuerError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid issuer",
        ) from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        ) from exc

    user_id = claims.get("sub") or claims.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    session_id = claims.get("sid") or claims.get("session_id")

    return AuthContext(
        user_id=user_id, session_id=session_id, token=token, claims=claims
    )


def get_current_user_id(auth: AuthContext = Depends(get_current_auth)) -> str:
    """Convenience dependency that exposes only the authenticated user's ID."""
    return auth.user_id
