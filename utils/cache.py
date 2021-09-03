import os
import pathlib
from typing import Optional, Type

import numpy as np
from pathlib import Path


class Cache:
    """
    A Simple filesystem cache implementation.
    """
    _instance = None  # type: Optional[Cache]

    @classmethod
    def get_instance(cls):  # type: (Type[Cache]) -> Cache
        if cls._instance is None:
            cls._instance = Cache()
        return cls._instance

    def __init__(self):
        self._cache_dir_path: str = str(pathlib.Path(__file__).parent.parent.resolve()) + '/cache'

    def put_numpy_array(self, key: str, data: np.ndarray) -> None:

        if not isinstance(data, np.ndarray):
            raise Exception('expected np.ndarray')

        data_dir_path = self._cache_dir_path + "/data"

        if not Path(data_dir_path).exists():
            os.mkdir(data_dir_path)

        np.save(self._generate_file_name_of_numpy_array(key),
                data)

    def has_numpy_array(self, key: str) -> bool:
        filename = self._generate_file_name_of_numpy_array(
            key
        )
        return Path(filename).exists()

    def get_numpy_array(self, key: str) -> np.ndarray:
        if not self.has_numpy_array(key):
            raise Exception(f'numpy array with the key {key} not fund in this cache')

        return np.load(self._generate_file_name_of_numpy_array(key))

    def delete_numpy_array(self, key: str):
        if not self.has_numpy_array(key):
            raise Exception(f'numpy array with the key {key} not fund in this cache')

        os.remove(self._generate_file_name_of_numpy_array(key))

    def _generate_file_name_of_numpy_array(self, external_key: str) -> str:
        return self._cache_dir_path + '/data/numpy_array_' + external_key + '.npy'
