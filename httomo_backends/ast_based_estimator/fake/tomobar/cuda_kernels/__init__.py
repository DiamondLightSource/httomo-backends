from typing import List, Tuple
from ...cupy import RawModule


def load_cuda_module(
    file: str, name_expressions: List[str] = None, options: Tuple[str] = tuple()
) -> RawModule:
    return RawModule(options, "", name_expressions)
