import time
from loguru import logger


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.info(f"{method.__qualname__} Took {int((te - ts) * 1000)}ms")
        return result

    return timed
