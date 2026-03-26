"""Configuration and paths for SignalRAG."""

import os
from pathlib import Path

# Signal Desktop paths (macOS)
SIGNAL_DIR = Path(os.path.expanduser("~/Library/Application Support/Signal"))
SIGNAL_DB_PATH = SIGNAL_DIR / "sql" / "db.sqlite"
SIGNAL_CONFIG_PATH = SIGNAL_DIR / "config.json"

# SignalRAG data directory
SIGNALRAG_DIR = Path(os.path.expanduser("~/.signalrag"))
VECTORSTORE_DIR = SIGNALRAG_DIR / "vectorstore"
STATE_FILE = SIGNALRAG_DIR / "state.json"

# SQLCipher 4 parameters (must match Signal's settings)
SQLCIPHER_PRAGMAS = {
    "cipher_page_size": 4096,
    "kdf_iter": 64000,
    "cipher_hmac_algorithm": "HMAC_SHA512",
    "cipher_kdf_algorithm": "PBKDF2_HMAC_SHA512",
}

# Keychain parameters (macOS Chromium os_crypt v10)
KEYCHAIN_SERVICE = "Signal Safe Storage"
KEYCHAIN_ACCOUNT = "Signal Key"
PBKDF2_SALT = b"saltysalt"
PBKDF2_ITERATIONS = 1003
AES_KEY_LENGTH = 16
AES_IV = b" " * 16
ENCRYPTED_KEY_PREFIX = b"v10"
