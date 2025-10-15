"""FastAPI authentication dependencies backed by Clerk JWTs."""

from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from app.auth.models import AuthContext
from app.config import settings

bearer_scheme = HTTPBearer(auto_error=False)


class AuthConfigError(RuntimeError):
    """Raised when the authentication environment is misconfigured."""


@lru_cache(maxsize=1)
def _get_jwk_client() -> PyJWKClient:
    """Create a cached PyJWKClient instance."""
    jwks_url = settings.CLERK_JWKS_URL
    if not jwks_url:
        raise AuthConfigError("CLERK_JWKS_URL is not configured")
    return PyJWKClient(jwks_url)


def _decode_token(token: str) -> dict:
    """Decode and validate a Clerk-issued JWT."""
    jwk_client = _get_jwk_client()
    signing_key = jwk_client.get_signing_key_from_jwt(token)

    algorithms = ["RS256", "ES256"]

    options = {"verify_signature": True}
    issuer: Optional[str] = settings.CLERK_ISSUER or None
    audience: Optional[str] = settings.CLERK_AUDIENCE or None

    if issuer is None:
        options["verify_iss"] = False
    if audience is None:
        options["verify_aud"] = False

    decoded = jwt.decode(
        token,
        signing_key.key,
        algorithms=algorithms,
        audience=audience,
        issuer=issuer,
        options=options,
    )
    return decoded


def get_current_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> AuthContext:
    """Validate the bearer token and return the authenticated context."""
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
