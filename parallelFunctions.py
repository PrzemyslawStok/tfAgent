import multiprocessing as mp
import numpy as np
from time import time


def howmany_within_range(row, minimum, maximum):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count


if __name__ == '__main__':
    # Prepare data
    np.random.RandomState(100)
    arr = np.random.randint(0, 10, size=[200000, 5])
    data = arr.tolist()

    print("Number of processors: ", mp.cpu_count())

    results = []

    for row in data:
        results.append(howmany_within_range(row, minimum=4, maximum=8))

    print(results[:10])

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(howmany_within_range, [(row, 4, 5) for row in data])
    pool.close()

    print(results[:10])
