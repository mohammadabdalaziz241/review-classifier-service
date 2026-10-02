"""Text preprocessing applied before inference.

Which steps run is decided per checkpoint (see serving_config.py), because
serving must reproduce the preprocessing the model was trained with; a
mismatch silently costs accuracy. The built-in default follows the convention
of Twitter-trained RoBERTa models (URLs -> "http", user mentions -> "@user").
"""

from __future__ import annotations

import re
import unicodedata

from .serving_config import PreprocessingConfig

URL_PLACEHOLDER = "http"
MENTION_PLACEHOLDER = "@user"

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# "@name" only when not preceded by a word character, so e-mail addresses survive.
_MENTION_RE = re.compile(r"(?<![\w@])@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_control_chars(text: str) -> str:
    # Remove control characters (category Cc) other than whitespace. Format
    # characters (Cf) such as the zero-width joiner are kept because they are
    # part of emoji sequences that carry sentiment.
    return "".join(ch for ch in text if ch.isspace() or unicodedata.category(ch) != "Cc")


def normalize_text(
    text: str,
    *,
    normalize_unicode: bool = True,
    normalize_whitespace: bool = True,
    replace_urls: bool = True,
    replace_mentions: bool = True,
) -> str:
    """Return a cleaned copy of ``text``. The function is idempotent.

    Control characters are always removed. Every other step can be turned off;
    with all of them off, ordinary text reaches the tokenizer unchanged.
    """
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    text = _strip_control_chars(text)
    if replace_urls:
        text = _URL_RE.sub(URL_PLACEHOLDER, text)
    if replace_mentions:
        text = _MENTION_RE.sub(MENTION_PLACEHOLDER, text)
    if normalize_whitespace:
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def preprocess(text: str, config: PreprocessingConfig) -> str:
    return normalize_text(text, **config.as_dict())
