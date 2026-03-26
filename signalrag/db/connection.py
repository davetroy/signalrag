"""SQLCipher connection management for Signal Desktop database."""

import shutil
import tempfile
from pathlib import Path

import sqlcipher3

from ..config import SIGNAL_DB_PATH, SQLCIPHER_PRAGMAS
from .key import extract_signal_key


class SignalDB:
    """Context manager for read-only access to the Signal Desktop database.

    By default, copies the database to a temp directory to avoid lock
    contention with Signal Desktop. Pass copy=False to open in-place
    (read-only mode).
    """

    def __init__(self, *, db_path: Path | None = None, copy: bool = True):
        self._source_path = db_path or SIGNAL_DB_PATH
        self._copy = copy
        self._conn = None
        self._temp_dir = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()

    def connect(self):
        db_key = extract_signal_key()

        if self._copy:
            self._temp_dir = tempfile.mkdtemp(prefix="signalrag_")
            work_path = Path(self._temp_dir) / "db.sqlite"
            shutil.copy2(self._source_path, work_path)
            # Also copy WAL/SHM if they exist for consistency
            for suffix in ("-wal", "-shm"):
                wal = self._source_path.parent / (self._source_path.name + suffix)
                if wal.exists():
                    shutil.copy2(wal, Path(self._temp_dir) / (work_path.name + suffix))
            db_path = str(work_path)
        else:
            db_path = str(self._source_path)

        self._conn = sqlcipher3.connect(db_path)
        c = self._conn.cursor()
        c.execute(f"PRAGMA key = \"x'{db_key}'\"")
        for pragma, value in SQLCIPHER_PRAGMAS.items():
            c.execute(f"PRAGMA {pragma} = {value}")

        # Verify the connection works
        c.execute("SELECT count(*) FROM sqlite_master")
        if c.fetchone()[0] == 0:
            raise RuntimeError("Failed to decrypt Signal database")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._temp_dir:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    @property
    def conn(self):
        if self._conn is None:
            raise RuntimeError("Database not connected. Use 'with SignalDB() as db:'")
        return self._conn

    def cursor(self):
        return self.conn.cursor()

    def execute(self, sql: str, params=None):
        c = self.cursor()
        if params:
            c.execute(sql, params)
        else:
            c.execute(sql)
        return c

    def fetchall(self, sql: str, params=None) -> list:
        return self.execute(sql, params).fetchall()

    def fetchone(self, sql: str, params=None):
        return self.execute(sql, params).fetchone()
