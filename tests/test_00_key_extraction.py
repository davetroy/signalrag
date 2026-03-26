"""Phase 0 Test: Extract Signal Desktop SQLCipher key from macOS Keychain.

Requires a real Signal Desktop installation on macOS.
"""

import pytest

pytestmark = pytest.mark.requires_signal


def test_key_extraction():
    from signalrag.db.key import extract_signal_key

    key = extract_signal_key()
    assert len(key) == 64, f"Expected 64-char hex key, got {len(key)} chars"
    int(key, 16)  # validate hex
