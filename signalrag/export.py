"""Export SignalRAG data to various formats (Parquet, CSV, DataFrame)."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .embeddings.store import MessageStore


def to_dataframe(store: MessageStore | None = None, *, include_vectors: bool = True) -> pd.DataFrame:
    """Export the vector store to a pandas DataFrame.

    Compatible with Apple's Embedding Atlas widget:
        from embedding_atlas.widget import EmbeddingAtlasWidget
        widget = EmbeddingAtlasWidget(df, text="text", vector="vector")
    """
    store = store or MessageStore()
    table = store.table.to_arrow()

    df = table.to_pandas()

    if not include_vectors and "vector" in df.columns:
        df = df.drop(columns=["vector"])

    return df


def to_parquet(
    output_path: str | Path,
    *,
    store: MessageStore | None = None,
    include_vectors: bool = True,
) -> Path:
    """Export the vector store to a Parquet file.

    Compatible with Apple's Embedding Atlas CLI:
        embedding-atlas output.parquet --text text --vector vector
    """
    output_path = Path(output_path)
    df = to_dataframe(store=store, include_vectors=include_vectors)
    df.to_parquet(output_path, index=False)
    return output_path


def to_csv(
    output_path: str | Path,
    *,
    store: MessageStore | None = None,
) -> Path:
    """Export metadata (without vectors) to CSV."""
    output_path = Path(output_path)
    df = to_dataframe(store=store, include_vectors=False)
    df.to_csv(output_path, index=False)
    return output_path
