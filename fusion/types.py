from queue import Queue
from typing import Any, Union

WorkerQueueT = Queue[tuple[int, int, int]]
PyArrowFilterT = Union[list[tuple[Any]], list[list[tuple[Any]]]]
