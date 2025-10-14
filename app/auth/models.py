from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AuthContext(BaseModel):
    """Represents the authenticated user and raw token claims."""

    user_id: str = Field(..., description="Clerk user identifier")
    session_id: Optional[str] = Field(None, description="Active Clerk session identifier")
    token: str = Field(..., description="Original bearer token")
    claims: Dict[str, Any] = Field(default_factory=dict, description="Decoded JWT claims")

    @property
    def email(self) -> Optional[str]:
        """Convenience accessor for the user's primary email address if it exists in the claims."""
        return self.claims.get("email") or self.claims.get("primary_email_address_id")
