import numpy as np
from hashlib import sha1


def generate_array_digest(array: np.ndarray) -> str:
    """
    Generate a reasonably unique string to identify an array's contents.
    """
    return sha1(array).hexdigest()
