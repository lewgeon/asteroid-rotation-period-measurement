"""Small independent fixtures for direct NPZ mappings and extraction fallback."""
import io
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from inversion.src.npz_mapping import source_identity, try_map_npz_member
from inversion.src.processing_store import MappedArchive


class NpzMappingTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[2] / 'tmp'
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.root = Path(self.temp.name)
        self.path = self.root / 'echo.npz'

    def tearDown(self):
        self.temp.cleanup()

    def write_member(self, array, *, version=None, extra=b'', zip64=False, compression=zipfile.ZIP_STORED):
        data = io.BytesIO()
        np.lib.format.write_array(data, array, version=version, allow_pickle=True)
        with zipfile.ZipFile(self.path, 'w', compression=compression) as archive:
            info = zipfile.ZipInfo('iq.npy')
            info.extra = extra
            info.compress_type = compression
            with archive.open(info, 'w', force_zip64=zip64) as output:
                output.write(data.getvalue())

    def test_c_f_and_dtype_matrix_reads_exact_values_without_copy(self):
        base = (np.arange(35).reshape(5, 7) * 3 + 2).astype(np.float64)
        arrays = [base + 1j * base[::-1], np.asfortranarray(base + 1j * base[::-1]),
                  base.astype('>f8'), base > 40, np.array(17),
                  np.array([(7, 2.5)], dtype=[('编号', '<i4'), ('x', '<f8')])]
        self.assertFalse(arrays[0].flags.f_contiguous)
        self.assertTrue(arrays[1].flags.f_contiguous)
        self.assertGreater(np.max(np.abs(arrays[0] - arrays[0].T.reshape(5, 7))), 0)
        for index, expected in enumerate(arrays):
            with self.subTest(index=index):
                self.write_member(expected)
                with np.load(self.path, allow_pickle=False) as reference:
                    golden = reference['iq']
                mapped = try_map_npz_member(self.path, 'iq')
                self.assertIsInstance(mapped, np.memmap)
                self.assertEqual(mapped.dtype, golden.dtype)
                self.assertEqual(mapped.shape, golden.shape)
                np.testing.assert_array_equal(mapped, golden)
                self.assertFalse(mapped.flags.writeable)
                self.assertEqual(Path(mapped.filename), self.path)
                if index < 2:
                    self.assertEqual(mapped.flags.f_contiguous, index == 1)
                mapped._mmap.close()
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), ['echo.npz'])

    def test_numpy_zip64_local_extra_and_custom_extra(self):
        expected = np.arange(33).reshape(3, 11)
        self.write_member(expected, extra=struct.pack('<HH4s', 0xCAFE, 4, b'test'), zip64=True)
        # Some Python versions keep the small local 32-bit sizes even with
        # force_zip64. Exercise the actual ZIP64 sentinel layout explicitly.
        with self.path.open('r+b') as raw:
            raw.seek(4)
            raw.write(struct.pack('<H', 45))
            raw.seek(18)
            raw.write(b'\xff' * 8)
        with zipfile.ZipFile(self.path) as archive:
            member = archive.getinfo('iq.npy')
            with self.path.open('rb') as raw:
                raw.seek(member.header_offset)
                fields = struct.unpack('<4s5H3I2H', raw.read(30))
                self.assertGreaterEqual(fields[1], 45)
                self.assertGreater(fields[-1], len(member.extra))
                raw.read(fields[-2])
                self.assertIn(struct.pack('<HH', 1, 16), raw.read(fields[-1]))
        mapped = try_map_npz_member(self.path, 'iq')
        np.testing.assert_array_equal(mapped, expected)
        mapped._mmap.close()

    def test_versions_and_empty_fallback(self):
        for version in [(1, 0), (2, 0), (3, 0)]:
            self.write_member(np.arange(12), version=version)
            mapped = try_map_npz_member(self.path, 'iq')
            np.testing.assert_array_equal(mapped, np.arange(12))
            mapped._mmap.close()
        self.write_member(np.empty((0, 4)))
        self.assertIsNone(try_map_npz_member(self.path, 'iq'))

    def test_prefixed_zip_data_descriptor_and_array_outlives_zip_handle(self):
        class NonSeekable(io.BytesIO):
            def seekable(self):
                return False
            def seek(self, *args):
                raise io.UnsupportedOperation('not seekable')
        expected = np.arange(48).reshape(8, 6)
        payload = io.BytesIO()
        np.save(payload, expected)
        output = NonSeekable()
        output.write(b'archive prefix')
        with zipfile.ZipFile(output, 'w') as archive:
            archive.writestr('iq.npy', payload.getvalue())
        self.path.write_bytes(output.getvalue())
        with zipfile.ZipFile(self.path) as archive:
            self.assertTrue(archive.getinfo('iq.npy').flag_bits & 8)
        mapped = try_map_npz_member(self.path, 'iq')
        np.testing.assert_array_equal(mapped[2:7:2], expected[2:7:2])
        with self.assertRaises(ValueError):
            mapped[0, 0] = 99
        mapped._mmap.close()

    def test_duplicate_member_and_cross_member_extent_rejected(self):
        payload = io.BytesIO()
        np.save(payload, np.arange(2000))
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('iq.npy', payload.getvalue())
            archive.writestr('valid.npy', payload.getvalue())
        raw = bytearray(self.path.read_bytes())
        central = raw.index(b'PK\x01\x02')
        # Inflate both central sizes so iq would include the next ZIP entry.
        for position in (central + 20, central + 24):
            struct.pack_into('<I', raw, position, len(payload.getvalue()) + 100)
        self.path.write_bytes(raw)
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            try_map_npz_member(self.path, 'iq')
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('iq.npy', payload.getvalue())
            with self.assertWarns(UserWarning):
                archive.writestr('iq.npy', payload.getvalue())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            try_map_npz_member(self.path, 'iq')

    def test_mapped_archive_no_extracted_arrays_and_existing_copy_retained(self):
        expected = np.arange(40).reshape(5, 8)
        np.savez(self.path, iq=expected, clean_iq=expected + 1, valid=expected > 9, meta=np.array('metadata'))
        folder = self.root / 'cache'
        folder.mkdir()
        stale = folder / 'iq.npy'
        stale.write_bytes(b'old cache remains untouched')
        with MappedArchive(self.path, folder) as archive:
            arrays = [archive[key] for key in ('iq', 'clean_iq', 'valid')]
            self.assertTrue(all(isinstance(value, np.memmap) for value in arrays))
            np.testing.assert_array_equal(arrays[0], expected)
            self.assertEqual(archive['meta'].item(), 'metadata')
        for value in arrays:
            value._mmap.close()
        self.assertEqual(stale.read_bytes(), b'old cache remains untouched')
        self.assertFalse((folder / 'clean_iq.npy').exists())
        self.assertFalse((folder / 'valid.npy').exists())
        self.assertFalse((folder / 'iq.json').exists())

    def test_compressed_and_empty_archive_fallback(self):
        for expected in [np.arange(300).reshape(30, 10), np.empty((0, 2))]:
            np.savez_compressed(self.path, iq=expected)
            self.assertIsNone(try_map_npz_member(self.path, 'iq'))
            with MappedArchive(self.path, self.root / 'cache') as archive:
                mapped = archive['iq']
                np.testing.assert_array_equal(mapped, expected)
                self.assertTrue((self.root / 'cache' / 'iq.npy').exists())
                mapped._mmap.close()

    def test_object_and_truncated_npy_rejected(self):
        self.write_member(np.array([{'unsafe': True}], dtype=object))
        with self.assertRaises(ValueError):
            try_map_npz_member(self.path, 'iq')
        data = io.BytesIO()
        np.save(data, np.arange(100))
        with zipfile.ZipFile(self.path, 'w') as archive:
            archive.writestr('iq.npy', data.getvalue()[:-8])
        with self.assertRaisesRegex(ValueError, 'payload size'):
            try_map_npz_member(self.path, 'iq')

    def test_direct_mapping_does_not_scan_payload_crc(self):
        # Direct mmap intentionally has the same payload-integrity scope as
        # mmap of NPY; it must not turn GUI row reads into full-member scans.
        expected = np.arange(20000, dtype=np.int64)
        np.savez(self.path, iq=expected)
        with zipfile.ZipFile(self.path) as archive:
            info = archive.getinfo('iq.npy')
            with self.path.open('r+b') as raw:
                raw.seek(info.header_offset)
                fields = struct.unpack('<4s5H3I2H', raw.read(30))
                last = info.header_offset + 30 + fields[-2] + fields[-1] + info.file_size - 1
                raw.seek(last)
                original = raw.read(1)
                raw.seek(last)
                raw.write(bytes([original[0] ^ 1]))
        with np.load(self.path, allow_pickle=False) as archive:
            with self.assertRaises(zipfile.BadZipFile):
                archive['iq']
        mapped = try_map_npz_member(self.path, 'iq')
        np.testing.assert_array_equal(mapped[:10], expected[:10])
        self.assertNotEqual(mapped[-1], expected[-1])
        mapped._mmap.close()

    def test_changed_source_rejected_before_and_during_mapping(self):
        np.savez(self.path, iq=np.arange(100))
        before = source_identity(self.path)
        os.utime(self.path, ns=(self.path.stat().st_atime_ns, before['mtime'] + 1_000_000))
        with self.assertRaisesRegex(ValueError, '变化'):
            try_map_npz_member(self.path, 'iq', expected_identity=before)
        real_memmap = np.memmap
        def changing_memmap(*args, **kwargs):
            mapped = real_memmap(*args, **kwargs)
            stat = self.path.stat()
            os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            return mapped
        with patch('inversion.src.npz_mapping.np.memmap', side_effect=changing_memmap):
            with self.assertRaisesRegex(ValueError, '变化'):
                try_map_npz_member(self.path, 'iq')

    def test_source_changed_in_archive_context_rejected(self):
        np.savez(self.path, iq=np.arange(10))
        with self.assertRaisesRegex(ValueError, '变化'):
            with MappedArchive(self.path, self.root / 'cache') as archive:
                mapped = archive['iq']
                mapped._mmap.close()
                stat = self.path.stat()
                os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    def test_corrupt_local_header_and_member_extent_rejected(self):
        np.savez(self.path, iq=np.arange(100))
        with self.path.open('r+b') as handle:
            handle.write(b'NOPE')
        with self.assertRaises(ValueError):
            try_map_npz_member(self.path, 'iq')
        np.savez(self.path, iq=np.arange(100))
        with self.path.open('r+b') as handle:
            handle.seek(28)
            handle.write(struct.pack('<H', 65535))
        with self.assertRaises(ValueError):
            try_map_npz_member(self.path, 'iq')


if __name__ == '__main__':
    unittest.main()
