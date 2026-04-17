"""
Data Pipeline
==============
Minimal dataset and data loading for next-token prediction training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from classical.tokenizer import CharTokenizer


class TextDataset:
    """Windowed next-token prediction dataset."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = np.array(token_ids, dtype=np.int64)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


class DataLoader:
    """Yields batches of (input, target) from a TextDataset."""

    def __init__(self, dataset: TextDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            xs, ys = [], []
            for i in batch_idx:
                x, y = self.dataset[i]
                xs.append(x)
                ys.append(y)
            yield np.array(xs), np.array(ys)

    def __len__(self) -> int:
        return max(0, len(self.dataset) // self.batch_size)


def load_corpus(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def prepare_data(
    text: str,
    seq_len: int = 16,
    batch_size: int = 4,
    val_fraction: float = 0.1,
    tokenizer: CharTokenizer | None = None,
) -> tuple[DataLoader, DataLoader, CharTokenizer]:
    if tokenizer is None:
        tokenizer = CharTokenizer.from_text(text)
    token_ids = tokenizer.encode(text)

    split = max(1, int(len(token_ids) * (1 - val_fraction)))
    train_ds = TextDataset(token_ids[:split], seq_len)
    val_ds = TextDataset(token_ids[split:], seq_len)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
