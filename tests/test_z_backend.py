import unittest


@unittest.skip("CPU/CUDA 回波一致性测试已迁移到 echo/tests")
class BackendTests(unittest.TestCase):
    def test_migrated_to_echo_module(self):
        pass
