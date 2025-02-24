from typing import Callable, Optional, Literal
import re

from .utils import split_with_regex, merge_splits
from ..base import Snippet


_whitespace_pattern = re.compile(r"\s+")
_non_whitespace_separators = [  # from https://github.com/isaacus-dev/semchunk
    ".",
    "?",
    "!",
    "*",  # Sentence terminators.
    ";",
    ",",
    "(",
    ")",
    "[",
    "]",
    "“",
    "”",
    "‘",
    "’",
    "'",
    '"',
    "`",  # Clause separators.
    ":",
    "—",
    "…",  # Sentence interrupters.
    "/",
    "\\",
    "–",
    "&",
    "-",  # Word joiners.
]


def len_word(text: str) -> int:
    return len(text.split())


def _default_separators(text):
    """Get the default separators based on the text"""
    whitespaces = _whitespace_pattern.findall(text)
    if not whitespaces:
        return _non_whitespace_separators + [""]

    return (
        list(sorted(set(whitespaces), key=lambda x: len(x), reverse=True))
        + _non_whitespace_separators
        + [""]
    )


def chunk_by_characters(
    text: str | Snippet,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_fn: Callable[[str], int] = len_word,
    separators: Optional[list[str]] = None,
    keep_separator: bool | Literal["start", "end"] = "start",
    is_separator_regex: bool = False,
    return_type: Literal["str", "snippet"] = "str",
) -> list[str]:
    """Chunk the text recursively based on a list of characters.

    Args:
        text: text to split
        chunk_size: maximum size of chunks to return
        chunk_overlap: overlap between chunks
        length_fn: function to measure text length
        separators: list of separators to use for splitting, tries each one in order
        keep_separator: whether to keep separator in chunks and where to place it
        is_separator_regex: whether separators are regex patterns

    Returns:
        list of text chunks
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )

    separators = separators or _default_separators(text)

    def _split_text(text: str, separators: list[str]) -> list[str]:
        """Recursive implementation of text splitting."""
        # find the first separator that appears in text
        separator = separators[-1]
        _separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            _sep = sep if is_separator_regex else re.escape(sep)
            if sep == "":
                separator = sep
                break
            if re.search(_sep, text):
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # split text using the selected separator
        _separator = separator if is_separator_regex else re.escape(separator)
        splits = split_with_regex(text, _separator, keep_separator=keep_separator)
        print(splits, _separator, new_separators)

        final_chunks = []
        good_splits = []
        _separator = "" if keep_separator else separator

        # process each split
        for split in splits:
            if length_fn(split) < chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = merge_splits(
                        good_splits, _separator, chunk_size, chunk_overlap, length_fn
                    )
                    final_chunks.extend(merged_text)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_chunks = _split_text(split, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged_text = merge_splits(
                good_splits, _separator, chunk_size, chunk_overlap, length_fn
            )
            final_chunks.extend(merged_text)

        return final_chunks

    return _split_text(text, separators)


__all__ = ["chunk_by_characters"]
