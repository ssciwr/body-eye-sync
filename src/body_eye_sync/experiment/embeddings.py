"""Storage for the best-K embedding tables the pipeline stages collect.

A video keeps body-appearance and face embeddings per track. The tables differ
only in which columns identify a vector, so the on-disk form is shared: every
column is written as it comes, except ``embedding``, which becomes a
fixed-size ``float16`` list.
"""

from __future__ import annotations

import heapq
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


class TopK:
    """Keeps the best-K embeddings per group as ``float16``, ranked by score.

    ``group`` is the identity the vectors are kept per -- a video's ``track_id``
    -- and ``index`` records where each one came from, a frame number.
    """

    def __init__(self, k: int, columns: list[str]) -> None:
        self._k = k
        self._columns = columns
        self._heaps: dict[int, list] = {}
        self._counter = itertools.count()

    def add(self, group: int, index: int, score: float, embedding) -> None:
        if self._k <= 0 or embedding is None:
            return
        vec = np.asarray(embedding, dtype=np.float16)
        # A per-group min-heap keyed by score keeps the K highest scoring; the
        # counter breaks score ties so the arrays are never compared.
        item = (float(score), next(self._counter), int(index), vec)
        heap = self._heaps.setdefault(int(group), [])
        if len(heap) < self._k:
            heapq.heappush(heap, item)
        elif item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def to_frame(self) -> pd.DataFrame | None:
        """Best-first table of the kept embeddings, or ``None`` if none were kept."""
        rows = []
        for group, heap in self._heaps.items():
            for score, _, index, vec in sorted(heap, key=lambda x: x[0], reverse=True):
                rows.append((group, index, score, vec))
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=self._columns)
        group_column, index_column, score_column = self._columns[:3]
        return frame.astype(
            {group_column: "int64", index_column: "int64", score_column: "float32"}
        )


def write_embeddings(path: str | Path, table: pd.DataFrame) -> None:
    """Write a best-K embeddings table as fixed-size ``float16`` vectors."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    matrix = np.stack(table["embedding"].to_numpy())
    _, dim = matrix.shape
    values = pa.array(matrix.reshape(-1), type=pa.float16())
    columns = {
        name: pa.array(table[name].to_numpy())
        for name in table.columns
        if name != "embedding"
    }
    columns["embedding"] = pa.FixedSizeListArray.from_arrays(values, dim)
    pq.write_table(pa.table(columns), str(path))


def read_embeddings(path: str | Path) -> pd.DataFrame:
    """Load an embeddings file back into a table of ``float16`` vectors."""
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pandas()
