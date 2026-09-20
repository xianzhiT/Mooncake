"""Opt-in TCP integration checks against a dedicated Mooncake master.

Set MOONCAKE_WEIGHT_TEST_MASTER=host:port and make the freshly built
mooncake.store extension importable. No GPU or inference engine is required.
"""

import json
import os
import struct
import tempfile
import unittest
import uuid
from pathlib import Path

from mooncake.weight_store import (
    DEFAULT_FILE_CHUNK_SIZE,
    WeightCacheClient,
    model_file_chunk_key,
)


@unittest.skipUnless(os.getenv("MOONCAKE_WEIGHT_TEST_MASTER"), "requires a test master")
class TestNativeWeightCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mooncake.store import MooncakeDistributedStore

        cls.store = MooncakeDistributedStore()
        real_client = os.getenv("MOONCAKE_WEIGHT_TEST_REAL_CLIENT")
        cls.through_dummy = bool(real_client)
        if real_client:
            result = cls.store.setup_dummy(32 * 1024**2, 16 * 1024**2, real_client)
        else:
            result = cls.store.setup(
                "127.0.0.1:0",
                "P2PHANDSHAKE",
                256 * 1024**2,
                128 * 1024**2,
                "tcp",
                "",
                os.environ["MOONCAKE_WEIGHT_TEST_MASTER"],
            )
        if result != 0:
            raise RuntimeError(f"store setup failed: {result}")
        cls.client = WeightCacheClient(cls.store, progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def setUp(self):
        self.checkpoint = "weight-test-" + uuid.uuid4().hex
        self.addCleanup(self.client.delete_model, self.checkpoint)

    def test_query_errors_are_not_empty_results(self):
        self.assertEqual(self.store.query_keys_by_regex("^" + self.checkpoint), [])
        with self.assertRaises(RuntimeError):
            self.store.query_keys_by_regex("[invalid")

    def test_uninitialized_query_raises(self):
        from mooncake.store import MooncakeDistributedStore

        raw = MooncakeDistributedStore()
        with self.assertRaises(RuntimeError):
            raw.query_keys_by_regex(".*")

    def test_closed_query_raises(self):
        from mooncake.store import MooncakeDistributedStore

        raw = MooncakeDistributedStore()
        raw.close()
        with self.assertRaises(RuntimeError):
            raw.query_keys_by_regex(".*")

    def test_single_shard_roundtrip_and_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "config.json").write_text("{}")
            (source / "tokenizer.json").write_text("{}")
            # Keep forwarding tests within Docker's default /dev/shm size.
            # Direct-client tests exercise the real 64 MiB chunk boundary.
            data_size = (
                128 * 1024 if self.through_dummy else DEFAULT_FILE_CHUNK_SIZE + 1024
            )
            header = json.dumps(
                {
                    "w": {
                        "dtype": "U8",
                        "shape": [data_size],
                        "data_offsets": [0, data_size],
                    }
                }
            ).encode()
            shard = source / "model.safetensors"
            with shard.open("wb") as handle:
                handle.write(struct.pack("<Q", len(header)))
                handle.write(header)
                handle.write(b"x" * data_size)
            self.assertTrue(self.client.import_model(self.checkpoint, source).complete)
            self.assertTrue(self.client.verify_model(self.checkpoint).complete)
            self.assertEqual(
                self.client.read_file(self.checkpoint, shard.name), shard.read_bytes()
            )
            first = model_file_chunk_key(self.checkpoint, shard.name, 0)
            self.assertEqual(self.store.remove(first, True), 0)
            status = self.client.inspect_model(self.checkpoint)
            self.assertFalse(status.complete)
            self.assertIn(first, status.missing_keys)


if __name__ == "__main__":
    unittest.main()
