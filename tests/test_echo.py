import unittest


@unittest.skip("回波仿真测试已迁移到 echo/tests；inversion 只读取 echo.npz")
class EchoTests(unittest.TestCase):
    def test_migrated_to_echo_module(self):
        pass
