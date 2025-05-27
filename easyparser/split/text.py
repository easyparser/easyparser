import copy
import re
from typing import Callable, Literal, Optional

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType
from easyparser.mime import MimeType

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
        chunks: Chunk | ChunkGroup,
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

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

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

        output = ChunkGroup()
        for ch in chunks:
            if isinstance(ch.content, str) and not ch.content:
                output.append(ch)
                continue

            if ch.ctype == CType.Root:
                continue  # Skip root chunks

            separators = separators or _default_separators(ch.text)
            splitted_texts = _split_text(ch.content, separators)
            if len(splitted_texts) == 1:
                # nothing to split, skip
                output.append(ch)
                continue

            # Record history
            history = copy.deepcopy(ch.history)
            history.append(
                cls.name(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    keep_separator=keep_separator,
                    is_separator_regex=is_separator_regex,
                )
            )

            splitted_chunks = [
                Chunk(
                    mimetype=MimeType.text,
                    content=text,
                    origin=ch.origin,
                    parent=ch,
                    history=history,
                )
                for text in splitted_texts
            ]

            # Next and prev intra chunks
            for idx, _c in enumerate(splitted_chunks[1:], start=1):
                _c.prev = splitted_chunks[idx - 1]
                splitted_chunks[idx - 1].next = _c

            # Return the first chunk
            output.append(splitted_chunks[0])

        return output


class ChunkJsonString(BaseOperation):
    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_fn: Callable[[str], int] | None | str = word_len,
        **kwargs,
    ):
        return ChunkByCharacters.run(
            chunks,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_fn=length_fn,
            separators=["},", "],", "}", "]", ",", " "],
            keep_separator="end",
            is_separator_regex=False,
            **kwargs,
        )
