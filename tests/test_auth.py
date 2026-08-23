"""test_auth.py - unit tests for password hashing and JWT round-tripping."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth import create_access_token, decode_token, hash_password, verify_password


def test_hash_and_verify_password():
    hashed = hash_password("correct-horse-battery-staple")
    assert hashed != "correct-horse-battery-staple"
    assert verify_password("correct-horse-battery-staple", hashed)
    assert not verify_password("wrong-password", hashed)


def test_jwt_round_trip():
    token = create_access_token("user-123")
    subject = decode_token(token)
    assert subject == "user-123"


def test_jwt_rejects_garbage_token():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        decode_token("not-a-real-token")
