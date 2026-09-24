import unittest


@unittest.skip("网格相关测试已迁移到 echo/tests；inversion 只读取 echo.npz")
class MeshTests(unittest.TestCase):
    def test_migrated_to_echo_module(self):
        pass
