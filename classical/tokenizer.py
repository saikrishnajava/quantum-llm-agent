"""
Character-Level Tokenizer
==========================
Lightweight tokenizer with special tokens, save/load support,
and zero external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]


class CharTokenizer:
    """Character-level tokenizer with special token support."""

    def __init__(self, vocab: dict[str, int] | None = None):
        if vocab is not None:
            self.char_to_id = dict(vocab)
        else:
            self.char_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in chars:
            if ch not in vocab:
                vocab[ch] = len(vocab)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    @property
    def pad_id(self) -> int:
        return self.char_to_id["<PAD>"]

    @property
    def unk_id(self) -> int:
        return self.char_to_id["<UNK>"]

    def encode(self, text: str, add_special: bool = False) -> list[int]:
        ids = [self.char_to_id.get(c, self.unk_id) for c in text]
        if add_special:
            ids = [self.char_to_id["<BOS>"]] + ids + [self.char_to_id["<EOS>"]]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        special_ids = set(range(len(SPECIAL_TOKENS))) if skip_special else set()
        return "".join(
            self.id_to_char.get(int(i), "?") for i in ids if int(i) not in special_ids
        )

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.char_to_id, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)
