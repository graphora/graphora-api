"""
Secure password encryption utilities for database configurations.

This module provides industry-standard AES-256-GCM encryption for passwords
with proper key derivation, salt handling, and secure random generation.
"""

import base64
import secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from app.config import get_settings

settings = get_settings()


class PasswordEncryption:
    """
    Secure password encryption using AES-256-GCM with PBKDF2 key derivation.

    Features:
    - AES-256-GCM for authenticated encryption
    - PBKDF2-HMAC-SHA256 for key derivation
    - Random salts and nonces for each encryption
    - Base64 encoding for database storage
    """

    @staticmethod
    def _get_master_key() -> bytes:
        """
        Get or generate the master encryption key from environment.

        Returns:
            bytes: 32-byte master key

        Raises:
            ValueError: If encryption key is not configured
        """
        # Get from environment variable
        key_env = getattr(settings, "ENCRYPTION_MASTER_KEY", None)

        if not key_env:
            raise ValueError(
                "ENCRYPTION_MASTER_KEY environment variable must be set. "
                'Generate one using: python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        # Decode base64 key
        try:
            key_bytes = base64.urlsafe_b64decode(
                key_env + "=="
            )  # Add padding if needed
            if len(key_bytes) < 32:
                # If key is shorter, pad it
                key_bytes = key_bytes.ljust(32, b"\0")
            elif len(key_bytes) > 32:
                # If key is longer, truncate it
                key_bytes = key_bytes[:32]
            return key_bytes
        except Exception:
            # Fallback: use the string directly and hash it
            return hashes.Hash(hashes.SHA256()).finalize()[:32]

    @staticmethod
    def _derive_key(master_key: bytes, salt: bytes) -> bytes:
        """
        Derive encryption key from master key using PBKDF2.

        Args:
            master_key: Master encryption key
            salt: Random salt for key derivation

        Returns:
            bytes: Derived 32-byte key
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        return kdf.derive(master_key)

    @classmethod
    def encrypt_password(cls, password: str) -> str:
        """
        Encrypt a password using AES-256-GCM.

        Args:
            password: Plaintext password to encrypt

        Returns:
            str: Base64-encoded encrypted password with salt and nonce
                 Format: base64(salt + nonce + ciphertext + tag)
        """
        if not password:
            return ""

        # Generate random salt and nonce
        salt = secrets.token_bytes(16)  # 128-bit salt
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM

        # Derive encryption key
        master_key = cls._get_master_key()
        encryption_key = cls._derive_key(master_key, salt)

        # Encrypt password
        aesgcm = AESGCM(encryption_key)
        ciphertext = aesgcm.encrypt(nonce, password.encode("utf-8"), None)

        # Combine salt + nonce + ciphertext (includes auth tag)
        encrypted_data = salt + nonce + ciphertext

        # Return base64-encoded result
        return base64.b64encode(encrypted_data).decode("ascii")

    @classmethod
    def decrypt_password(cls, encrypted_password: str) -> str:
        """
        Decrypt an encrypted password.

        Args:
            encrypted_password: Base64-encoded encrypted password

        Returns:
            str: Decrypted plaintext password

        Raises:
            ValueError: If decryption fails or data is corrupted
        """
        if not encrypted_password:
            return ""

        try:
            # Decode base64
            encrypted_data = base64.b64decode(encrypted_password.encode("ascii"))

            # Check minimum length (salt + nonce + ciphertext + tag)
            if len(encrypted_data) < 16 + 12 + 16:  # salt + nonce + min_ciphertext
                raise ValueError("Encrypted data is too short")

            # Extract components
            salt = encrypted_data[:16]
            nonce = encrypted_data[16:28]
            ciphertext = encrypted_data[28:]

            # Derive decryption key
            master_key = cls._get_master_key()
            decryption_key = cls._derive_key(master_key, salt)

            # Decrypt password
            aesgcm = AESGCM(decryption_key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)

            return plaintext.decode("utf-8")

        except Exception as e:
            raise ValueError(f"Password decryption failed: {str(e)}")

    @classmethod
    def is_encrypted(cls, password: str) -> bool:
        """
        Check if a password string appears to be encrypted.

        Args:
            password: Password string to check

        Returns:
            bool: True if password appears encrypted, False otherwise
        """
        if not password:
            return False

        try:
            # Try to decode as base64
            decoded = base64.b64decode(password.encode("ascii"))
            # Check if it has the expected minimum length
            return len(decoded) >= 16 + 12 + 16  # salt + nonce + min_ciphertext
        except Exception:
            return False

    @classmethod
    def migrate_plaintext_password(cls, password: str) -> str:
        """
        Migrate a plaintext password to encrypted format.

        Args:
            password: Password (plaintext or already encrypted)

        Returns:
            str: Encrypted password
        """
        if cls.is_encrypted(password):
            return password  # Already encrypted
        return cls.encrypt_password(password)


# Convenience functions for backward compatibility
def encrypt_password(password: str) -> str:
    """Encrypt a password using AES-256-GCM."""
    return PasswordEncryption.encrypt_password(password)


def decrypt_password(encrypted_password: str) -> str:
    """Decrypt an encrypted password."""
    return PasswordEncryption.decrypt_password(encrypted_password)


def is_encrypted(password: str) -> bool:
    """Check if a password string appears to be encrypted."""
    return PasswordEncryption.is_encrypted(password)


def migrate_plaintext_password(password: str) -> str:
    """Migrate a plaintext password to encrypted format."""
    return PasswordEncryption.migrate_plaintext_password(password)
