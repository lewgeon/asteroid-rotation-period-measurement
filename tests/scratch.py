"""inversion 测试的私有临时目录。

``tempfile.TemporaryDirectory`` / ``mkdtemp`` 在 Windows 上建出的目录带
``0o700`` 对应的 ACL，受限令牌的进程写不进去（``PermissionError``）。测试只需要
一个跑完就删的私有目录，因此改在项目内、被 git 忽略的 ``tmp/`` 下按确定名字
创建——与 ``echo/tests/scratch.py`` 的做法一致。
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
INVERSION_DIR = TESTS_DIR.parent


@contextmanager
def scratch_directory(name: str):
    """Yield a writable, project-local scratch directory for one test."""

    path = INVERSION_DIR / "tmp" / f"{name}_{os.getpid()}"
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
