# %%

from collections import defaultdict
import time
import functools


counts = defaultdict(int)
times = defaultdict(float)

def timeit(name):
    """
    Decorator to time a function.
    Example usage:
    @timeit('my_func')
    def my_func():
        ...
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            result = f(*args, **kwargs)
            end = time.monotonic()
            counts[name] += 1
            times[name] += end - start
            return result
        return wrapper
    return decorator
