"""Chinese-first ByteLevel BPE tokenizer (24,576 tokens including specials).

Layout of the vocabulary, in id order:

    0 <pad>   1 <bos>   2 <eos>   3 <unk>
    4 <|system|>   5 <|user|>   6 <|assistant|>   7 <|end_of_turn|>
    8..263        the 256 byte-level symbols (guaranteed by `initial_alphabet`)
    264..24575    BPE merges learned from the corpus

The first 256 byte symbols are always present, so every UTF-8 string is
encodable and `decode(encode(x)) == x` holds for arbitrary text. The special
tokens are listed in a fixed order, which is what pins their ids.

Chat format (identical for training and inference):

    <bos><|system|>\\n{system}<|end_of_turn|>\\n
        <|user|>\\n{user}<|end_of_turn|>\\n
        <|assistant|>\\n{reply}<|end_of_turn|>\\n
        ...

Only assistant content and its trailing `<|end_of_turn|>` are supervised.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_OF_TURN_TOKEN = "<|end_of_turn|>"

SPECIAL_TOKENS = [
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    UNK_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    END_OF_TURN_TOKEN,
]

SPECIAL_TOKEN_IDS = {
    "pad": 0,
    "bos": 1,
    "eos": 2,
    "unk": 3,
    "system": 4,
    "user": 5,
    "assistant": 6,
    "end_of_turn": 7,
}

PAD_ID = SPECIAL_TOKEN_IDS["pad"]
BOS_ID = SPECIAL_TOKEN_IDS["bos"]
EOS_ID = SPECIAL_TOKEN_IDS["eos"]
UNK_ID = SPECIAL_TOKEN_IDS["unk"]
SYSTEM_ID = SPECIAL_TOKEN_IDS["system"]
USER_ID = SPECIAL_TOKEN_IDS["user"]
ASSISTANT_ID = SPECIAL_TOKEN_IDS["assistant"]
END_OF_TURN_ID = SPECIAL_TOKEN_IDS["end_of_turn"]

DEFAULT_SYSTEM = "你是一个有帮助的中文助手。"
TARGET_VOCAB_SIZE = 24576


@dataclass
class ChatEncoding:
    ids: list[int]
    labels: list[int]  # -100 where the token is not supervised

    def __len__(self) -> int:
        return len(self.ids)


class MiniLLMTokenizer:
    def __init__(self, tokenizer: Tokenizer, ignore_index: int = -100):
        self._tok = tokenizer
        self.ignore_index = ignore_index
        # Runtime experiment metadata, deliberately excluded from vocab JSON.
        self.text_policy = "identity-v1"

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = TARGET_VOCAB_SIZE,
        save_path: str | Path | None = None,
    ) -> "MiniLLMTokenizer":
        """Train from scratch and refuse to silently produce a smaller vocabulary."""
        tok = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
            min_frequency=2,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        wrapper = cls(tok)
        actual = wrapper.vocab_size
        if actual != vocab_size:
            raise RuntimeError(
                f"tokenizer training produced {actual} tokens but {vocab_size} were requested; "
                "the corpus is too small or too repetitive for this vocabulary size. "
                "Supply more text or lower --vocab-size explicitly."
            )
        wrapper.validate_layout()
        if save_path is not None:
            wrapper.save(save_path)
        return wrapper

    def validate_layout(self) -> None:
        """Special ids are part of the model contract; verify them after training/loading."""
        vocab = self._tok.get_vocab()
        for name, expected in SPECIAL_TOKEN_IDS.items():
            token = dict(
                pad=PAD_TOKEN, bos=BOS_TOKEN, eos=EOS_TOKEN, unk=UNK_TOKEN,
                system=SYSTEM_TOKEN, user=USER_TOKEN,
                assistant=ASSISTANT_TOKEN, end_of_turn=END_OF_TURN_TOKEN,
            )[name]
            actual = vocab.get(token)
            if actual != expected:
                raise ValueError(f"special token {token!r} has id {actual}, expected {expected}")
        if self.vocab_size < 256 + len(SPECIAL_TOKENS):
            raise ValueError("vocabulary is missing byte-level symbols")

    # ------------------------------------------------------------------ io

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: str | Path, ignore_index: int = -100) -> "MiniLLMTokenizer":
        wrapper = cls(Tokenizer.from_file(str(path)), ignore_index=ignore_index)
        wrapper.validate_layout()
        return wrapper

    # ------------------------------------------------------------------ properties

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def fingerprint(self) -> str:
        """Hash of vocabulary, merges and preprocessing (specials included)."""
        canonical = json.dumps(json.loads(self._tok.to_str()), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def special_ids(self) -> dict[str, int]:
        return dict(SPECIAL_TOKEN_IDS)

    def assert_vocab_size(self, expected: int) -> None:
        if self.vocab_size != expected:
            raise ValueError(
                f"tokenizer has {self.vocab_size} tokens but the model expects {expected}"
            )

    # ------------------------------------------------------------------ text

    def prepare_text(self, text: str, *, category: str = "auto") -> str:
        """Apply the experiment policy with the caller's declared content type.

        Category is input metadata, not a property inferable from the vocabulary.
        Unlabelled text uses the existing auto heuristic; callers must pass code
        for source text that the heuristic cannot recognize.
        """
        from .text_policy import apply_text_policy
        return apply_text_policy(text, self.text_policy, category=category)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False,
               category: str = "auto") -> list[int]:
        ids = self._tok.encode(self.prepare_text(text, category=category)).ids
        if add_bos:
            ids = [BOS_ID, *ids]
        if add_eos:
            ids = [*ids, EOS_ID]
        return ids

    def encode_document(self, text: str, *, category: str = "auto") -> list[int]:
        """One pretraining document: content followed by EOS (no BOS)."""
        return self.encode(text, add_bos=False, add_eos=True, category=category)

    def encode_batch(self, texts: Sequence[str], *, add_bos: bool = False,
                     add_eos: bool = False, categories: Sequence[str] | None = None) -> list[list[int]]:
        if categories is not None and len(categories) != len(texts):
            raise ValueError("encode_batch categories must align with texts")
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos,
                            category=categories[i] if categories is not None else "auto") for i,t in enumerate(texts)]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------ chat

    def _segment(self, text: str, *, category: str = "auto") -> list[int]:
        """Encode a chat segment on its own.

        Segments are tokenised separately so that marker boundaries can never be
        merged into a neighbouring word by BPE — the same split is used at
        training and inference time, which is what keeps the two formats equal.
        """
        return self.encode(text, category=category)

    def chat_encoding(
        self,
        messages: Sequence[dict],
        *,
        system: str | None = DEFAULT_SYSTEM,
        add_generation_prompt: bool = False,
        supervise_all_assistant_turns: bool = True,
        category: str = "auto",
        system_category: str = "auto",
    ) -> ChatEncoding:
        """Tokenise a conversation and mark the tokens that carry loss.

        `messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.
        A system message inside `messages` overrides the `system` argument; the
        default system prompt is emitted when neither is present, so the format
        is stable even for system-less data.
        """
        if not isinstance(messages, Sequence) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                raise ValueError("each message needs 'role' and 'content'")
            if message["role"] not in ("system", "user", "assistant"):
                raise ValueError(f"unknown role {message['role']!r}")

        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            system_category = messages[0].get("category", system_category)
            messages = messages[1:]
        if not messages:
            raise ValueError("conversation has no user/assistant turns")

        ids: list[int] = [BOS_ID]
        labels: list[int] = [self.ignore_index]
        ignore = self.ignore_index

        if system is not None and system != "":
            ids += [SYSTEM_ID, *self._segment("\n" + system, category=system_category), END_OF_TURN_ID, *self._segment("\n")]
            labels += [ignore] * (len(ids) - len(labels))

        assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        last_assistant = assistant_indices[-1] if assistant_indices else -1

        for index, message in enumerate(messages):
            role_id = USER_ID if message["role"] == "user" else ASSISTANT_ID
            content = self._segment("\n" + message["content"], category=message.get("category", category))
            supervised = message["role"] == "assistant" and (
                supervise_all_assistant_turns or index == last_assistant
            )
            ids.append(role_id)
            labels.append(ignore)
            ids += content
            # Labels are aligned 1:1 with input ids; the model shifts internally
            # in `MiniLLM.loss_terms()`, so a supervised label must be the real
            # target token, never a placeholder such as PAD (0).
            labels += content if supervised else [ignore] * len(content)
            ids.append(END_OF_TURN_ID)
            labels.append(END_OF_TURN_ID if supervised else ignore)
            separator = self._segment("\n")
            ids += separator
            labels += [ignore] * len(separator)

        if add_generation_prompt:
            ids += [ASSISTANT_ID, *self._segment("\n")]
            labels += [ignore] * (len(ids) - len(labels))
        assert len(ids) == len(labels)
        return ChatEncoding(ids=ids, labels=labels)

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        *,
        system: str | None = DEFAULT_SYSTEM,
        add_generation_prompt: bool = False,
        category: str = "auto",
        system_category: str = "auto",
    ) -> list[int]:
        """Inference-side ids; always exercised by the same code path as SFT."""
        return self.chat_encoding(
            messages, system=system, add_generation_prompt=add_generation_prompt,
            category=category, system_category=system_category,
        ).ids

    # ------------------------------------------------------------------ metrics

    def efficiency_report(self, samples: dict[str, Sequence[str]]) -> dict[str, float]:
        """Token-per-character / per-word efficiency used by the tokenizer tests.

        Keys: `zh_tokens_per_char`, `en_tokens_per_word`, `code_compression`
        (characters per token) and `roundtrip_ok`.
        """
        report: dict[str, float] = {}
        if samples.get("zh"):
            chars = sum(len(t) for t in samples["zh"])
            tokens = sum(len(self.encode(t)) for t in samples["zh"])
            report["zh_tokens_per_char"] = tokens / max(chars, 1)
        if samples.get("en"):
            words = sum(len(t.split()) for t in samples["en"])
            tokens = sum(len(self.encode(t)) for t in samples["en"])
            report["en_tokens_per_word"] = tokens / max(words, 1)
        if samples.get("code"):
            chars = sum(len(t) for t in samples["code"])
            tokens = sum(len(self.encode(t)) for t in samples["code"])
            report["code_compression"] = chars / max(tokens, 1)
        roundtrip = all(
            self.decode(self.encode(t), skip_special_tokens=True) == t
            for texts in samples.values()
            for t in texts
        )
        report["roundtrip_ok"] = 1.0 if roundtrip else 0.0
        return report


def normalise_text(text: str) -> str:
    """NFKC + newline cleanup used before training the tokenizer."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def iter_texts(paths: Iterable[str | Path], text_key: str = "text") -> Iterable[str]:
    """Yield `text` from JSONL files or plain-text files, one document at a time."""
    for path in paths:
        path = Path(path)
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            if path.suffix == ".jsonl":
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = record.get(text_key) if isinstance(record, dict) else None
                    if isinstance(text, str) and text.strip():
                        yield text
            else:
                buffer: list[str] = []
                for line in handle:
                    if line.strip():
                        buffer.append(line)
                    elif buffer:
                        yield "".join(buffer)
                        buffer = []
                if buffer:
                    yield "".join(buffer)
