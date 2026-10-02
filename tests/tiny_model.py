"""A tiny BERT classifier built on the fly, so model tests need no download."""

from __future__ import annotations

import pytest

VOCAB_WORDS = ("great", "bad", "product", "it", "love", "awful", "service", "the", "was", "food")
LABELS = {0: "negative", 1: "positive"}


def build_tiny_model(directory, *, multi_label: bool = False, vocab_words=VOCAB_WORDS) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *vocab_words]
    vocab_file = directory / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file))

    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
        problem_type="multi_label_classification" if multi_label else None,
    )
    torch.manual_seed(0)
    model = transformers.BertForSequenceClassification(config)
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
