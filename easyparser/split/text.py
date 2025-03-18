import copy
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

        output = ChunkGroup()
        for group_root, chunks in chunk.iter_groups():
            result = []
            for ch in chunks:
                if not ch.text:
                    result.append(ch)
                    continue

                splitted_texts = _split_text(ch.text, separators)
                if len(splitted_texts) == 1:
                    # nothing to split, skip
                    result.append(ch)
                    continue

                # Note that the resulting chunks are split from the original chunk
                metadata = copy.deepcopy(ch.metadata)
                metadata["split"] = True

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
                        mimetype=ch.mimetype,
                        content=ch.content,
                        text=text,
                        origin=ch.origin,
                        parent=ch.parent,  # the parent will be here to fast traveling
                        metadata=metadata,
                        history=history,
                    )
                    for text in splitted_texts
                ]

                # The last chunk will inherit the child
                splitted_chunks[-1].child = ch.child

                # The first chunk will inherit the parent
                if ch.parent:
                    ch.parent.child = splitted_chunks[0]

                # Next and prev intra chunks
                for idx, _c in enumerate(splitted_chunks[1:], start=1):
                    _c.prev = splitted_chunks[idx - 1]
                    splitted_chunks[idx - 1].next = _c

                # Outside next and prev
                if ch.prev:
                    splitted_chunks[0].prev = ch.prev
                    ch.prev.next = splitted_chunks[0]
                if ch.next:
                    splitted_chunks[-1].next = ch.next
                    ch.next.prev = splitted_chunks[-1]

                result.extend(splitted_chunks)

            output.add_group(ChunkGroup(root=group_root, chunks=result))

        if chunk.store:
            output.attach_store(chunk.store)

        return output
