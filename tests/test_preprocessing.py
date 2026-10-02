import pytest

from review_classifier.preprocessing import (
    MENTION_PLACEHOLDER,
    URL_PLACEHOLDER,
    normalize_text,
)


def test_collapses_and_strips_whitespace():
    assert normalize_text("  great \n\n  product\t ") == "great product"


def test_applies_nfkc_normalisation():
    # Full-width letters and the "ﬁ" ligature become plain ASCII.
    assert normalize_text("\uff27\uff32\uff25\uff21\uff34 ﬁt") == "GREAT fit"


def test_removes_control_characters():
    assert normalize_text("good\x00 val\x07ue") == "good value"


def test_keeps_emoji_sequences_intact():
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"  # uses zero-width joiners
    assert normalize_text(f"love it {family}") == f"love it {family}"


@pytest.mark.parametrize(
    "raw",
    ["see https://example.com/item?id=1", "see http://x.co", "see www.example.org/page"],
)
def test_replaces_urls(raw):
    assert normalize_text(raw) == f"see {URL_PLACEHOLDER}"


def test_replaces_mentions_but_not_email_addresses():
    text = "@shop_uk thanks, reply to me@example.com"
    assert normalize_text(text) == f"{MENTION_PLACEHOLDER} thanks, reply to me@example.com"


def test_unicode_normalisation_can_be_disabled():
    assert normalize_text("\uff27 \ufb01", normalize_unicode=False) == "\uff27 \ufb01"


def test_whitespace_normalisation_can_be_disabled():
    assert normalize_text("  a \n\n b ", normalize_whitespace=False) == "  a \n\n b "


def test_control_characters_are_removed_even_in_raw_mode():
    raw = dict(
        normalize_unicode=False,
        normalize_whitespace=False,
        replace_urls=False,
        replace_mentions=False,
    )
    assert normalize_text("a\x00b c", **raw) == "ab c"


def test_replacements_can_be_disabled():
    text = "@shop see https://example.com"
    assert normalize_text(text, replace_urls=False, replace_mentions=False) == text


def test_is_idempotent():
    text = "  @shop  \uff27\uff32\uff25\uff21\uff34   https://a.b/c \x00 "
    once = normalize_text(text)
    assert normalize_text(once) == once


@pytest.mark.parametrize("raw", ["", "   ", "\n\t", "\x00\x01"])
def test_blank_input_becomes_empty_string(raw):
    assert normalize_text(raw) == ""
