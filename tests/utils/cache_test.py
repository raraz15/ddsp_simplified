import unittest

from utils.cache import Cache


class MyTestCase(unittest.TestCase):
    def test_cache(self):
        cache = Cache.get_instance()


if __name__ == '__main__':
    unittest.main()
