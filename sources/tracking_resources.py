import os
import tracemalloc
import time

def track_resources(func, *args, **kwargs):
    tracemalloc.start()
    start_time = time.time()

    result = func(*args, **kwargs)

    duration = time.time() - start_time
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return result, duration, peak_ram / (1024 * 1024)  # Convert to MB
