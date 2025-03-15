import re
from typing import Callable, Literal, Optional

from ..base import BaseOperation, Chunk, ChunkGroup
from .utils import merge_splits, split_with_regex, word_len

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


class ChunkByCharacters(BaseOperation):

    _len_fns = {
        "len": len,
        "word_len": word_len,
    }

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_fn: Callable[[str], int] | None | str = word_len,
        separators: Optional[list[str]] = None,
        keep_separator: bool | Literal["start", "end"] = "start",
        is_separator_regex: bool = False,
        **kwargs,
    ) -> ChunkGroup:
        """Chunk the text recursively based on a list of characters.

        Args:
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

        if isinstance(chunk, Chunk):
            chunk = ChunkGroup([chunk])

        separators = separators or _default_separators(chunk[0].text)
        if isinstance(length_fn, str):
            length = ChunkByCharacters._len_fns[length_fn]
        elif length_fn is None:
            length = word_len
        else:
            length = length_fn

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

            final_chunks = []
            good_splits = []
            _separator = "" if keep_separator else separator

            # process each split
            for split in splits:
                if length(split) < chunk_size:
                    good_splits.append(split)
                else:
                    if good_splits:
                        merged_text = merge_splits(
                            good_splits, _separator, chunk_size, chunk_overlap, length
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
                    good_splits, _separator, chunk_size, chunk_overlap, length
                )
                final_chunks.extend(merged_text)

            return final_chunks

        result = ChunkGroup()
        for c in chunk:
            splitted_texts = _split_text(c.text, separators)
            if len(splitted_texts) == 1:
                # nothing to split, skip
                result.append(c)
                continue

            splitted_chunks = [
                Chunk(
                    mimetype="text/plain",
                    content=text,
                    text=text,
                    metadata=c.metadata,
                    origin=c.origin,
                    parent=c.parent,
                )
                for text in splitted_texts
            ]

            for r in splitted_chunks:
                for h in c.history:
                    r.history.append(h)
                r.history.append(
                    cls.name(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        keep_separator=keep_separator,
                        is_separator_regex=is_separator_regex,
                    )
                )
                result.append(r)

        if chunk.store:
            result.attach_store(chunk.store)

        return result
