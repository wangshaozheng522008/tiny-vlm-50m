"""Byte-level BPE tokenizer with the special tokens TinyVLM expects.

A 16k vocabulary keeps the token embedding table at ~6M parameters, which matters
when the whole quantised model has to fit in 50 MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
IMAGE_TOKEN = "<image>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, IMAGE_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

PAD_ID = 0
IMAGE_ID = 1
BOS_ID = 2
EOS_ID = 3
UNK_ID = 4


class TinyTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 16384,
        save_path: str | Path | None = None,
    ) -> "TinyTokenizer":
        tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        wrapper = cls(tok)
        if save_path is not None:
            wrapper.save(save_path)
        return wrapper

    # ------------------------------------------------------------------ io

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TinyTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    # ------------------------------------------------------------------ api

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = self._tok.encode(text).ids
        return [BOS_ID, *ids, EOS_ID] if add_special_tokens else ids

    def encode_batch(self, texts: Sequence[str], add_special_tokens: bool = True) -> list[list[int]]:
        return [self.encode(t, add_special_tokens) for t in texts]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
