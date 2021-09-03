import unittest

import numpy as np

from utils.cache import Cache


class MyTestCase(unittest.TestCase):
    def test_cache(self):
        cache = Cache.get_instance()

        # this one doesn't exist yet
        self.assertFalse(cache.has_numpy_array('xxx'))

        array_to_cache = np.array([1, 2, 3])

        cache.put_numpy_array('xxx', array_to_cache)
        self.assertTrue(cache.has_numpy_array('xxx'))

        restored_array = cache.get_numpy_array('xxx')

        self.assertTrue(np.array_equal(array_to_cache, restored_array))

        cache.delete_numpy_array('xxx')
        self.assertFalse(cache.has_numpy_array('xxx'))


if __name__ == '__main__':
    unittest.main()
