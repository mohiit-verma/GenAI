import time, contextlib
from typing import Iterator

@contextlib.contextmanager
def timed(label: str) -> Iterator[None]:
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{label}] {dt:.2f}s")