"""Read-only mappings of uncompressed NPY members, without extracted copies."""
from pathlib import Path
import math
import os
import struct
import zipfile

import numpy as np


def source_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime=stat.st_mtime_ns)


def _check_source(path, expected, handle=None):
    current = source_identity(path)
    if current != expected:
        raise ValueError('读取期间回波发生变化')
    if handle is not None:
        stat = os.fstat(handle.fileno())
        if (stat.st_size, stat.st_mtime_ns) != (expected['size'], expected['mtime']):
            raise ValueError('读取期间回波发生变化')


def try_map_npz_member(path, key, *, expected_identity=None):
    """Return a read-only memmap, or None for compressed/unsupported members.

    ``key`` is the NumPy archive key (without ``.npy``). Object arrays and
    malformed/truncated members raise rather than being mapped. The source is
    checked before and after constructing the mapping; long-lived consumers
    must also check its identity when finishing their read, as MappedArchive
    does. Like mmap of a standalone NPY, this does not scan the payload or
    verify its ZIP CRC; structural checks cannot detect arbitrary bit rot.
    Existing extracted caches are neither read nor removed here.
    """
    path = Path(path)
    expected = source_identity(path) if expected_identity is None else expected_identity
    _check_source(path, expected)
    with path.open('rb') as handle:
        _check_source(path, expected, handle)
        with zipfile.ZipFile(handle) as archive:
            name = key + '.npy'
            matches = [entry for entry in archive.infolist() if entry.filename == name]
            if len(matches) != 1:
                if not matches:
                    raise KeyError(key)
                raise ValueError('NPZ contains duplicate array members')
            member = matches[0]
            if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
                return None
            # The local extra field can differ from the central directory
            # (np.savez uses a ZIP64 local field even for small arrays).
            handle.seek(member.header_offset)
            header = handle.read(30)
            if len(header) != 30:
                raise ValueError('Truncated ZIP local header')
            fields = struct.unpack('<4s5H3I2H', header)
            if fields[0] != b'PK\x03\x04' or fields[3] != zipfile.ZIP_STORED:
                raise ValueError('Inconsistent ZIP local header')
            if fields[2] != member.flag_bits:
                raise ValueError('Inconsistent ZIP member flags')
            start = member.header_offset + 30 + fields[-2] + fields[-1]
            end = start + member.file_size
            if (member.compress_size != member.file_size or start < 0
                    or end > archive.start_dir or end > expected['size']):
                raise ValueError('Invalid ZIP member extent')
            # ZipFile.open also checks the local filename and overlapping
            # entries. Read only the NPY header, never the full array payload.
            with archive.open(member) as source:
                version = np.lib.format.read_magic(source)
                if version not in ((1, 0), (2, 0), (3, 0)):
                    return None
                shape, fortran, dtype = np.lib.format._read_array_header(source, version)
                payload_offset = source.tell()
            if dtype.hasobject:
                raise ValueError('Object arrays cannot be memory mapped')
            size = math.prod(shape) * dtype.itemsize
            if payload_offset + size != member.file_size:
                raise ValueError('NPY payload size does not match ZIP member extent')
            if size == 0:
                return None
            result = np.memmap(handle, dtype=dtype, mode='r',
                               offset=start + payload_offset, shape=shape,
                               order='F' if fortran else 'C')
            try:
                _check_source(path, expected, handle)
            except Exception:
                result._mmap.close()
                raise
            return result
