import numpy as np


def generate_array_digest(array: np.ndarray) -> str:
    """
    Generate a reasonably unique string to identify an array.
    """
    return str(hash(array.data.tobytes()))
