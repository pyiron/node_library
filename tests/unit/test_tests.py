import unittest
import node_library


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = node_library.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
