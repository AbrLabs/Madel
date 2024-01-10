from typing import Tuple

import numpy as np
from numpy.random import randint


def generate_data(
    size: int = 10, features_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    data = np.random.randint(0, 2, (size, features_size))
    result = np.zeros((size, size), dtype=int)

    relations = randint(0, 2, size=(size, size))
    np.fill_diagonal(relations, 1)

    result[np.triu_indices(size, k=0)] = relations[np.triu_indices(size, k=0)]
    result = result + result.T - np.diag(result.diagonal())

    return data, result