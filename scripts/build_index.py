"""Build the full SignalRAG vector index."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signalrag.embeddings.indexer import Indexer


def main():
    indexer = Indexer()
    stats = indexer.full_index(min_length=5, progress_fn=print)
    print("\nFinal stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
