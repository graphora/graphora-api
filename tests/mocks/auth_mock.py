"""Mock authentication dependencies for London School TDD unit tests.

These mocks allow testing of authenticated endpoints without
requiring real JWT tokens or Clerk integration.
"""

from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class MockAuthContext:
    """Mock authentication context.

    Matches the structure of the real AuthContext model.
    """

    user_id: str
    session_id: Optional[str] = None
    token: str = "mock-token"
    claims: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.claims:
            self.claims = {
                "sub": self.user_id,
                "email": f"{self.user_id}@example.com",
            }


def create_mock_auth_dependency(
    user_id: str = "test-user-123",
    session_id: Optional[str] = "test-session-456",
    token: str = "mock-jwt-token",
    claims: Dict[str, Any] = None,
    raise_exception: Optional[Exception] = None,
) -> Callable:
    """Create a mock authentication dependency for FastAPI testing.

    Args:
        user_id: User ID to return in auth context.
        session_id: Optional session ID.
        token: Mock JWT token string.
        claims: Additional JWT claims.
        raise_exception: Optional exception to raise (for testing auth failures).

    Returns:
        A dependency function that returns MockAuthContext or raises.

    Example:
        ```python
        from fastapi.testclient import TestClient

        app.dependency_overrides[get_current_auth] = create_mock_auth_dependency(
            user_id="user-123"
        )

        client = TestClient(app)
        response = client.get("/protected/endpoint")
        ```
    """

    async def mock_auth_dependency():
        if raise_exception:
            raise raise_exception

        return MockAuthContext(
            user_id=user_id,
            session_id=session_id,
            token=token,
            claims=claims or {"sub": user_id},
        )

    return mock_auth_dependency


def create_mock_user_id_dependency(
    user_id: str = "test-user-123",
    raise_exception: Optional[Exception] = None,
) -> Callable:
    """Create a mock user ID dependency for FastAPI testing.

    Args:
        user_id: User ID to return.
        raise_exception: Optional exception to raise.

    Returns:
        A dependency function that returns the user ID string.

    Example:
        ```python
        app.dependency_overrides[get_current_user_id] = create_mock_user_id_dependency(
            user_id="user-456"
        )
        ```
    """

    async def mock_user_id_dependency():
        if raise_exception:
            raise raise_exception
        return user_id

    return mock_user_id_dependency


class MockJWTDecoder:
    """Mock JWT decoder for testing authentication logic.

    Allows configuring what claims are returned for different tokens
    and simulating various JWT validation errors.
    """

    def __init__(self):
        self._token_claims: Dict[str, Dict[str, Any]] = {}
        self._raise_for_token: Dict[str, Exception] = {}
        self._default_claims: Dict[str, Any] = {
            "sub": "default-user",
            "iss": "https://test.clerk.dev",
            "aud": "test-audience",
            "exp": 9999999999,  # Far future
            "iat": 1000000000,
        }

    def configure_token(self, token: str, claims: Dict[str, Any]):
        """Configure claims to return for a specific token."""
        self._token_claims[token] = claims

    def configure_error(self, token: str, exception: Exception):
        """Configure an exception to raise for a specific token."""
        self._raise_for_token[token] = exception

    def decode(self, token: str, **kwargs) -> Dict[str, Any]:
        """Decode a token (mock implementation).

        Args:
            token: The JWT token string.
            **kwargs: Additional decode options (ignored in mock).

        Returns:
            Dict of claims.

        Raises:
            Configured exception if token is set to raise.
        """
        if token in self._raise_for_token:
            raise self._raise_for_token[token]

        if token in self._token_claims:
            return self._token_claims[token]

        # Return default claims with token-derived user ID
        return {**self._default_claims, "sub": f"user-from-{token[:8]}"}


class MockJWKClient:
    """Mock JWKS client for testing."""

    def __init__(self, signing_key: Any = None):
        self._signing_key = signing_key or MockSigningKey()

    def get_signing_key_from_jwt(self, token: str) -> "MockSigningKey":
        """Get signing key for token verification."""
        return self._signing_key


class MockSigningKey:
    """Mock signing key for JWT verification."""

    def __init__(self, key: str = "mock-public-key"):
        self.key = key
