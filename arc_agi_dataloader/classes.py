from dataclasses import dataclass
from typing import List

@dataclass
class GridSample:
    input: List[List[int]]
    output: List[List[int]]
@dataclass
class EpisodicGridSample:
    train: List[GridSample]
    test: List[GridSample]


