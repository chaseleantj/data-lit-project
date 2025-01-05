import time

def time_it(func):
    """
    Decorator to print the time taken for a function to execute.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

