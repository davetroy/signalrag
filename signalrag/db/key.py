"""Extract the SQLCipher database key from macOS Keychain."""

import hashlib
import json
import subprocess

from Crypto.Cipher import AES

from ..config import (
    SIGNAL_CONFIG_PATH,
    KEYCHAIN_SERVICE,
    KEYCHAIN_ACCOUNT,
    PBKDF2_SALT,
    PBKDF2_ITERATIONS,
    AES_KEY_LENGTH,
    AES_IV,
    ENCRYPTED_KEY_PREFIX,
)

_cached_key: str | None = None


def extract_signal_key() -> str:
    """Extract the SQLCipher database key from macOS Keychain + config.json.

    The key is encrypted using Electron's safeStorage API (Chromium os_crypt v10):
      1. Keychain password -> PBKDF2-HMAC-SHA1 -> AES-128 key
      2. AES-128-CBC decrypt of encryptedKey from config.json
      3. Result is 64-char hex string used as SQLCipher raw key

    Returns the 64-character hex key string.
    """
    global _cached_key
    if _cached_key is not None:
        return _cached_key

    # Step 1: Get the safe storage password from macOS Keychain
    password = subprocess.check_output([
        "security", "find-generic-password",
        "-s", KEYCHAIN_SERVICE, "-a", KEYCHAIN_ACCOUNT, "-w"
    ]).strip()

    # Step 2: Read encryptedKey from Signal config
    with open(SIGNAL_CONFIG_PATH) as f:
        config = json.load(f)

    if "encryptedKey" in config:
        encrypted = bytes.fromhex(config["encryptedKey"])
        if not encrypted.startswith(ENCRYPTED_KEY_PREFIX):
            raise ValueError(
                f"Expected {ENCRYPTED_KEY_PREFIX!r} prefix, got {encrypted[:3]!r}"
            )

        # Step 3: Derive AES key via PBKDF2
        aes_key = hashlib.pbkdf2_hmac(
            "sha1", password, PBKDF2_SALT, PBKDF2_ITERATIONS, dklen=AES_KEY_LENGTH
        )

        # Step 4: Decrypt (AES-128-CBC, strip v10 prefix)
        cipher = AES.new(aes_key, AES.MODE_CBC, iv=AES_IV)
        decrypted = cipher.decrypt(encrypted[len(ENCRYPTED_KEY_PREFIX):])

        # Remove PKCS7 padding
        pad_len = decrypted[-1]
        db_key = decrypted[:-pad_len].decode("ascii")

    elif "key" in config:
        # Legacy format: plaintext key
        db_key = config["key"]

    else:
        raise ValueError(
            "No 'encryptedKey' or 'key' found in Signal config.json"
        )

    if len(db_key) != 64:
        raise ValueError(f"Expected 64-char hex key, got {len(db_key)} chars")
    int(db_key, 16)  # validate hex

    _cached_key = db_key
    return db_key
