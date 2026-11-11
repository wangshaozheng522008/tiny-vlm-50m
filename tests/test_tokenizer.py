"""Tokenizer contract: fixed special ids, roundtrip, chat template, efficiency."""

from __future__ import annotations

from pathlib import Path

import pytest

from minillm.tokenizer import (
    ASSISTANT_ID,
    ASSISTANT_TOKEN,
    BOS_ID,
    END_OF_TURN_ID,
    END_OF_TURN_TOKEN,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKEN_IDS,
    SYSTEM_ID,
    UNK_ID,
    USER_ID,
    MiniLLMTokenizer,
)

CHINESE = "人工智能是机器学习的重要组成部分，需要大量的数据和计算资源。"
ENGLISH = "Machine learning models need data, compute and careful evaluation."
CODE = "def add(a, b):\n    return a + b\n"
MIXED = "MiniLLM 是一个 87M 参数的中文语言模型，支持 INT4 量化。\n"


def test_special_token_ids_are_fixed(tiny_tokenizer):
    assert tiny_tokenizer.special_ids == SPECIAL_TOKEN_IDS
    assert [PAD_ID, BOS_ID, EOS_ID, UNK_ID] == [0, 1, 2, 3]
    assert [SYSTEM_ID, USER_ID, ASSISTANT_ID, END_OF_TURN_ID] == [4, 5, 6, 7]
    vocab = tiny_tokenizer._tok.get_vocab()
    assert vocab[ASSISTANT_TOKEN] == ASSISTANT_ID
    assert vocab[END_OF_TURN_TOKEN] == END_OF_TURN_ID


def test_vocab_size_and_byte_alphabet(tiny_tokenizer):
    assert tiny_tokenizer.vocab_size == 1024
    # every byte is a token, so no string can fall back to <unk>
    assert tiny_tokenizer.encode("") == []
    exotic = "🙂‍↔️\u200b\x7f"
    ids = tiny_tokenizer.encode(exotic)
    assert tiny_tokenizer.decode(ids) == exotic
    assert UNK_ID not in ids


@pytest.mark.parametrize("text", [CHINESE, ENGLISH, CODE, MIXED, "", "  ", "a\n\nb"])
def test_roundtrip(tiny_tokenizer, text):
    ids = tiny_tokenizer.encode(text)
    assert tiny_tokenizer.decode(ids) == text
    assert all(0 <= token < tiny_tokenizer.vocab_size for token in ids)


def test_encode_document_appends_eos_only(tiny_tokenizer):
    ids = tiny_tokenizer.encode_document(CHINESE)
    assert ids[-1] == EOS_ID
    assert BOS_ID not in ids
    assert ids[: len(tiny_tokenizer.encode(CHINESE))] == tiny_tokenizer.encode(CHINESE)


def test_insufficient_corpus_raises_instead_of_shrinking():
    with pytest.raises(RuntimeError, match="too small"):
        MiniLLMTokenizer.train(["短文本。"], vocab_size=24576)


def test_assert_vocab_size(tiny_tokenizer):
    tiny_tokenizer.assert_vocab_size(1024)
    with pytest.raises(ValueError, match="expects 24576"):
        tiny_tokenizer.assert_vocab_size(24576)


def test_fingerprint_survives_save_and_load(tiny_tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(path)
    loaded = MiniLLMTokenizer.load(path)
    assert loaded.fingerprint == tiny_tokenizer.fingerprint
    assert loaded.encode(MIXED) == tiny_tokenizer.encode(MIXED)
    assert loaded.special_ids == tiny_tokenizer.special_ids


def test_chat_template_supervises_assistant_only(tiny_tokenizer):
    messages = [
        {"role": "system", "content": "你是助手。"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"},
    ]
    encoding = tiny_tokenizer.chat_encoding(messages)
    ignored = tiny_tokenizer.ignore_index
    supervised = [token for token, label in zip(encoding.ids, encoding.labels) if label != ignored]
    assert supervised, "assistant turn must carry loss"
    assert supervised[-1] == END_OF_TURN_ID, "the closing marker is supervised"
    text = tiny_tokenizer.decode(supervised, skip_special_tokens=True)
    assert "你好" in text
    assert "你是助手" not in text and "人工智能" not in text

    # structural tokens, system prompt and user turn are never supervised
    for token, label in zip(encoding.ids, encoding.labels):
        if token in (BOS_ID, SYSTEM_ID, USER_ID, ASSISTANT_ID):
            assert label == ignored
    assert encoding.ids[0] == BOS_ID
    decoded = tiny_tokenizer.decode(encoding.ids)
    assert decoded.startswith("你是助手。\n你好\n你好，有什么可以帮你？") or "你好" in decoded


def test_chat_template_roles_and_generation_prompt(tiny_tokenizer):
    messages = [{"role": "user", "content": "写一句诗"}]
    prompt = tiny_tokenizer.chat_encoding(messages, system=None, add_generation_prompt=True)
    assert prompt.ids[-1] != ASSISTANT_ID
    assert ASSISTANT_ID in prompt.ids
    # the generation prompt ends with `<|assistant|>\n`, mirrored by SFT formatting
    assistant_at = len(prompt.ids) - 1 - len(tiny_tokenizer._segment("\n"))
    assert prompt.ids[assistant_at] == ASSISTANT_ID
    assert all(label == tiny_tokenizer.ignore_index for label in prompt.labels)

    with_system = tiny_tokenizer.chat_encoding(messages, system="你是诗人。")
    assert SYSTEM_ID in with_system.ids


def test_last_assistant_turn_only(tiny_tokenizer):
    messages = [
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
        {"role": "user", "content": "第二个问题"},
        {"role": "assistant", "content": "第二个回答"},
    ]
    all_turns = tiny_tokenizer.chat_encoding(messages, supervise_all_assistant_turns=True)
    last_turn = tiny_tokenizer.chat_encoding(messages, supervise_all_assistant_turns=False)
    assert sum(l != tiny_tokenizer.ignore_index for l in all_turns.labels) > sum(
        l != tiny_tokenizer.ignore_index for l in last_turn.labels
    )
    supervised = tiny_tokenizer.decode(
        [t for t, l in zip(last_turn.labels and last_turn.ids, last_turn.labels)
         if l != tiny_tokenizer.ignore_index],
        skip_special_tokens=True,
    )
    assert "第二个回答" in supervised
    assert "第一个回答" not in supervised


def test_chat_template_rejects_bad_input(tiny_tokenizer):
    with pytest.raises(ValueError):
        tiny_tokenizer.chat_encoding([])
    with pytest.raises(ValueError):
        tiny_tokenizer.chat_encoding([{"role": "tool", "content": "x"}])
    with pytest.raises(ValueError):
        tiny_tokenizer.chat_encoding([{"role": "user"}])


def test_efficiency_report_shape(tiny_tokenizer):
    report = tiny_tokenizer.efficiency_report(
        {"zh": [CHINESE], "en": [ENGLISH], "code": [CODE]}
    )
    assert report["roundtrip_ok"] == 1.0
    assert 0 < report["zh_tokens_per_char"] < 3
    assert 0 < report["en_tokens_per_word"] < 10
    assert report["code_compression"] > 0.2


REAL_CORPUS = Path("data/minimind/pretrain_t2t_mini.jsonl")


@pytest.mark.skipif(not REAL_CORPUS.exists(), reason="real corpus not available")
def test_efficiency_on_real_chinese(tmp_path):
    """A tokenizer trained on real text must beat byte-level encoding."""
    from minillm.data import stream_documents

    texts = []
    for text in stream_documents([REAL_CORPUS]):
        texts.append(text)
        if len(texts) >= 4000:
            break
    tokenizer = MiniLLMTokenizer.train(texts, vocab_size=4096)
    sample = "人工智能的发展离不开算力、算法和数据，模型规模越大，训练成本越高。"
    report = tokenizer.efficiency_report({"zh": [sample], "code": [CODE]})
    assert report["zh_tokens_per_char"] < 1.2, report
    assert report["roundtrip_ok"] == 1.0


# ------------------------------------------------------------------ P0: labels

SENTINEL_USER = "UUUSERQUESTION"
SENTINEL_ASSISTANT = "AAASSISTANTREPLY"


def test_chat_labels_are_the_real_next_tokens(tiny_tokenizer):
    encoding = tiny_tokenizer.chat_encoding([
        {"role": "system", "content": "SSSYSTEMPROMPT"},
        {"role": "user", "content": SENTINEL_USER},
        {"role": "assistant", "content": SENTINEL_ASSISTANT},
    ])
    ids, labels = encoding.ids, encoding.labels
    ignore = tiny_tokenizer.ignore_index
    assert len(ids) == len(labels)
    for token, label in zip(ids, labels):
        assert label in (ignore, token), "supervised labels must equal the input token"
    supervised = [t for t, l in zip(ids, labels) if l != ignore]
    assert supervised, "assistant turn must be supervised"
    assert supervised[-1] == END_OF_TURN_ID, "the terminator is a real target"
    assert PAD_ID not in supervised and all(t != PAD_ID for t in supervised)
    assert any(t != PAD_ID for t in supervised), "labels were all collapsed to PAD"


def test_chat_supervises_exactly_the_assistant_segment(tiny_tokenizer):
    encoding = tiny_tokenizer.chat_encoding([
        {"role": "system", "content": "SSSYSTEMPROMPT"},
        {"role": "user", "content": SENTINEL_USER},
        {"role": "assistant", "content": SENTINEL_ASSISTANT},
    ])
    ignore = tiny_tokenizer.ignore_index
    content = tiny_tokenizer._segment("\n" + SENTINEL_ASSISTANT)
    supervised = sum(1 for label in encoding.labels if label != ignore)
    assert supervised == len(content) + 1, "content tokens plus the closing marker"
    allowed = set(content) | {END_OF_TURN_ID}
    for token, label in zip(encoding.ids, encoding.labels):
        if label == ignore:
            continue
        assert token in allowed, f"token {token} is not part of the assistant answer"
    supervised_ids = [t for t, l in zip(encoding.ids, encoding.labels) if l != ignore]
    assert supervised_ids[: len(content)] == content


def test_chat_system_and_user_tokens_are_ignored(tiny_tokenizer):
    encoding = tiny_tokenizer.chat_encoding([
        {"role": "system", "content": "SSSYSTEMPROMPT"},
        {"role": "user", "content": SENTINEL_USER},
        {"role": "assistant", "content": SENTINEL_ASSISTANT},
    ])
    ignore = tiny_tokenizer.ignore_index
    system_tokens = set(tiny_tokenizer._segment("\n" + "SSSYSTEMPROMPT"))
    user_tokens = set(tiny_tokenizer._segment("\n" + SENTINEL_USER))
    answer_tokens = set(tiny_tokenizer._segment("\n" + SENTINEL_ASSISTANT))
    for token, label in zip(encoding.ids, encoding.labels):
        inprompt = token in system_tokens or token in user_tokens
        if inprompt and token not in answer_tokens:
            assert label == ignore, "prompt tokens must not be trained on"
    supervised_text = tiny_tokenizer.decode(
        [t for t, l in zip(encoding.ids, encoding.labels) if l != ignore],
        skip_special_tokens=True,
    )
    assert SENTINEL_ASSISTANT in supervised_text
    assert SENTINEL_USER not in supervised_text
    assert "SSSYSTEMPROMPT" not in supervised_text


def test_chat_multi_turn_labels_are_aligned(tiny_tokenizer):
    encoding = tiny_tokenizer.chat_encoding([
        {"role": "user", "content": "第一个问题"},
        {"role": "assistant", "content": "第一个回答"},
        {"role": "user", "content": "第二个问题"},
        {"role": "assistant", "content": "第二个回答"},
    ])
    ignore = tiny_tokenizer.ignore_index
    for token, label in zip(encoding.ids, encoding.labels):
        assert label in (ignore, token)
    supervised = [t for t, l in zip(encoding.ids, encoding.labels) if l != ignore]
    assert supervised.count(END_OF_TURN_ID) == 2, "each answer terminator is supervised"
    text = tiny_tokenizer.decode(supervised, skip_special_tokens=True)
    assert "第一个回答" in text and "第二个回答" in text


def test_last_turn_only_labels_are_aligned(tiny_tokenizer):
    encoding = tiny_tokenizer.chat_encoding(
        [
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": "第一个回答"},
            {"role": "user", "content": "第二个问题"},
            {"role": "assistant", "content": "第二个回答"},
        ],
        supervise_all_assistant_turns=False,
    )
    ignore = tiny_tokenizer.ignore_index
    for token, label in zip(encoding.ids, encoding.labels):
        assert label in (ignore, token)
    supervised = [t for t, l in zip(encoding.ids, encoding.labels) if l != ignore]
    text = tiny_tokenizer.decode(supervised, skip_special_tokens=True)
    assert "第二个回答" in text and "第一个回答" not in text
    assert supervised.count(END_OF_TURN_ID) == 1


def test_chat_generation_prompt_appends_assistant_marker(tiny_tokenizer):
    messages = [{"role": "user", "content": "你好"}]
    plain = tiny_tokenizer.chat_encoding(messages)
    prompted = tiny_tokenizer.chat_encoding(messages, add_generation_prompt=True)
    assert prompted.ids[: len(plain.ids)] == plain.ids, "same template as SFT"
    assert prompted.ids[len(plain.ids)] == ASSISTANT_ID
    assert all(label == tiny_tokenizer.ignore_index for label in prompted.labels)
